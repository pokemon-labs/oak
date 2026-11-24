"""Speed options for the perfect-info trainer (2026-11-22, docs/pi_speed.md),
each behind a Config flag in ataraxos/pi/train.py so it can be ablated; the
defaults there keep the original code path.

  - NumpyBuffer (`--buffer numpy`): train.Buffer's data, built with numpy in
    `take` instead of a Python loop over rows in every `add`.
    tests/py/check_pi_buffer.py checks it gives train.Buffer's Data.
  - AsyncActor (`--actor async|graphs`): pinned host buffers, non-blocking
    copies, and `launch` / `collect` so the CPU can step one env while the
    GPU runs the other's forward. `graphs`: the forward and the sampling as
    one CUDA graph at the env's row count (padded with row 0 if fewer).
  - Envs (`--pingpong`): the battles split over two env objects, so
    `collect` on one overlaps the other's step; it looks like one env to the
    trainer's Exp3 and stats code.
The buffer and the actor take the model's row columns and inputs (pi's or
pi4's, ataraxos/pi/train.py's `Model`, 2026-11-24).
"""

import numpy as np
import torch

from ataraxos.pi.net import CHOICES, legal_logits

ROW_COLUMNS = ("mon", "hp", "active", "policy", "n", "choices")
INPUTS = ("mon", "hp", "active", "policy", "n")


class NumpyBuffer:
    """train.Buffer with the trajectories found in `take`: a row is in
    player p's trajectory of its (battle, game) when p has 2+ choices; a
    trajectory is done when its game has ended (and, with at_cap "drop", not
    at the cap). `games()` counts the ended games not dropped (train.Buffer
    counts done trajectories / 2: the same but for a game where one player
    never chose)."""

    def __init__(self, at_cap="draw", gather=False, columns=ROW_COLUMNS):
        """`gather` (the trainer's --take_gather): `take` concatenates only
        the index columns and gathers each other column's needed rows from
        the stored parts, not every row of every column first. `columns`:
        the row columns kept (the model's)."""
        if at_cap not in ("draw", "drop"):
            raise ValueError(at_cap)
        self.columns = tuple(columns)
        self.at_cap = at_cap
        self.gather = gather
        self.parts = []
        self.size = 0
        self.ends = []
        self.n_games = 0

    def end(self, obs):
        if len(obs["end_battle"]) == 0:
            return
        capped = obs["end_capped"].astype(bool)
        self.ends.append((obs["end_battle"].astype(np.int64), obs["end_game"].astype(np.int64),
                          obs["end_result"].astype(np.int64), capped))
        self.n_games += int(len(capped) if self.at_cap == "draw" else (~capped).sum())

    def add(self, obs, picks, logp, value, extra=None):
        """`extra`: per-row columns kept alongside (search in training's
        searched, post, swdl; the actor's q_exp, q_pick: pass them on every
        add or none)."""
        R = len(obs["n"])
        if R == 0:
            return
        part = {k: obs[k] for k in self.columns}
        part.update(pick=picks, logp=logp, value=value, battle=obs["battle"], game=obs["game"])
        if extra:
            part.update(extra)
        self.parts.append(part)
        self.size += R

    def games(self):
        return self.n_games

    def _col(self, k, idx, offsets):
        """Column k at the sorted global rows idx, gathered part by part."""
        part = np.searchsorted(offsets, idx, side="right") - 1
        cuts = np.searchsorted(part, np.arange(len(self.parts) + 1))
        pieces = [self.parts[j][k][idx[cuts[j]:cuts[j + 1]] - offsets[j]]
                  for j in range(len(self.parts)) if cuts[j + 1] > cuts[j]]
        if pieces:
            return np.concatenate(pieces)
        return self.parts[0][k][:0]

    def take(self):
        if self.gather:
            return self._take_gather()
        from ataraxos.pi.train import Data

        cat = {k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}
        key = (cat["battle"].astype(np.int64) << 32) | cat["game"].astype(np.int64)
        if self.ends:
            eb, eg, er, ec = (np.concatenate(x) for x in zip(*self.ends))
            ekey = (eb << 32) | eg
            o = np.argsort(ekey, kind="stable")
            ekey, er, ec = ekey[o], er[o], ec[o]
            pos = np.minimum(np.searchsorted(ekey, key), len(ekey) - 1)
            ended = ekey[pos] == key
            result, capped = er[pos], ec[pos]
        else:
            ended = np.zeros(len(key), bool)
            result, capped = np.zeros(len(key), np.int64), np.zeros(len(key), bool)
        fin = ended & ~(capped & (self.at_cap == "drop"))
        n = cat["n"]
        r0, r1 = np.flatnonzero(fin & (n[:, 0] >= 2)), np.flatnonzero(fin & (n[:, 1] >= 2))
        allr = np.concatenate([r0, r1])
        allp = np.concatenate([np.zeros(len(r0), np.int64), np.ones(len(r1), np.int64)])
        o = np.lexsort((allr, allp, key[allr]))  # by game, then player, then time
        rows, side = allr[o], allp[o]
        gk = key[rows]
        new = np.r_[True, (gk[1:] != gk[:-1]) | (side[1:] != side[:-1])] if len(rows) else np.zeros(0, bool)
        start = np.flatnonzero(new)
        length = np.diff(np.r_[start, len(rows)]).astype(np.int64)
        player = side[start]
        reward = result[rows[start]] * np.where(player == 0, 1, -1)
        used, inv = np.unique(rows, return_inverse=True)
        value = cat["value"][rows]
        value = np.where(side[:, None] == 1, value[:, ::-1], value)
        data = Data(row=inv, player=player, side=side, n=n[rows, side], choices=cat["choices"][rows, side],
                    logp=cat["logp"][rows, side], pick=cat["pick"][rows, side], value=np.ascontiguousarray(value),
                    start=start.astype(np.int64), length=length, reward=reward.astype(np.int64),
                    rows={k: cat[k][used] for k in self.columns}, truncated=np.zeros(len(length), bool))
        if "q_exp" in cat:  # the deciding player's Q (its own W/D/L) under the data policy and at its pick
            data.q_exp = np.ascontiguousarray(cat["q_exp"][rows, side])
            data.q_pick = np.ascontiguousarray(cat["q_pick"][rows, side])
        if "searched" in cat:  # the deciding player's: its posterior, the W/D/L from its side
            data.searched = cat["searched"][rows].astype(bool)
            data.post = cat["post"][rows, side]
            sw = cat["swdl"][rows]
            data.swdl = np.ascontiguousarray(np.where(side[:, None] == 1, sw[:, ::-1], sw))
        keep = np.flatnonzero(~ended)  # the unfinished games' rows (a dropped game's rows go)
        self.parts = [{k: v[keep] for k, v in cat.items()}] if len(keep) else []
        self.size = len(keep)
        self.ends = []
        self.n_games = 0
        return data


    def _take_gather(self):
        """take's Data from the index columns and gathered rows (`gather`)."""
        from ataraxos.pi.train import Data

        offsets = np.cumsum([0] + [len(p["n"]) for p in self.parts])[:-1]
        cat = {k: np.concatenate([p[k] for p in self.parts]) for k in ("battle", "game", "n")}
        key = (cat["battle"].astype(np.int64) << 32) | cat["game"].astype(np.int64)
        if self.ends:
            eb, eg, er, ec = (np.concatenate(x) for x in zip(*self.ends))
            ekey = (eb << 32) | eg
            o = np.argsort(ekey, kind="stable")
            ekey, er, ec = ekey[o], er[o], ec[o]
            pos = np.minimum(np.searchsorted(ekey, key), len(ekey) - 1)
            ended = ekey[pos] == key
            result, capped = er[pos], ec[pos]
        else:
            ended = np.zeros(len(key), bool)
            result, capped = np.zeros(len(key), np.int64), np.zeros(len(key), bool)
        fin = ended & ~(capped & (self.at_cap == "drop"))
        n = cat["n"]
        r0, r1 = np.flatnonzero(fin & (n[:, 0] >= 2)), np.flatnonzero(fin & (n[:, 1] >= 2))
        allr = np.concatenate([r0, r1])
        allp = np.concatenate([np.zeros(len(r0), np.int64), np.ones(len(r1), np.int64)])
        o = np.lexsort((allr, allp, key[allr]))  # by game, then player, then time
        rows, side = allr[o], allp[o]
        gk = key[rows]
        new = np.r_[True, (gk[1:] != gk[:-1]) | (side[1:] != side[:-1])] if len(rows) else np.zeros(0, bool)
        start = np.flatnonzero(new)
        length = np.diff(np.r_[start, len(rows)]).astype(np.int64)
        player = side[start]
        reward = result[rows[start]] * np.where(player == 0, 1, -1)
        used, inv = np.unique(rows, return_inverse=True)
        U = {k: self._col(k, used, offsets) for k in self.parts[0] if k not in ("battle", "game")}
        value = U["value"][inv]
        value = np.where(side[:, None] == 1, value[:, ::-1], value)
        data = Data(row=inv, player=player, side=side, n=n[rows, side], choices=U["choices"][inv, side],
                    logp=U["logp"][inv, side], pick=U["pick"][inv, side], value=np.ascontiguousarray(value),
                    start=start.astype(np.int64), length=length, reward=reward.astype(np.int64),
                    rows={k: U[k] for k in self.columns}, truncated=np.zeros(len(length), bool))
        if "q_exp" in U:
            data.q_exp = np.ascontiguousarray(U["q_exp"][inv, side])
            data.q_pick = np.ascontiguousarray(U["q_pick"][inv, side])
        if "searched" in U:  # the deciding player's: its posterior, the W/D/L from its side
            data.searched = U["searched"][inv].astype(bool)
            data.post = U["post"][inv, side]
            sw = U["swdl"][inv]
            data.swdl = np.ascontiguousarray(np.where(side[:, None] == 1, sw[:, ::-1], sw))
        keep = np.flatnonzero(~ended)  # the unfinished games' rows (a dropped game's rows go)
        if len(keep):
            self.parts = [{k: (cat[k][keep] if k in cat else self._col(k, keep, offsets)) for k in self.parts[0]}]
        else:
            self.parts = []
        self.size = len(keep)
        self.ends = []
        self.n_games = 0
        return data


class Envs:
    """Several SelfPlay envs as one for the trainer's team code (weights,
    probs, counts, names, stats). Env i's battles are offset by the battles
    before it (`shift`), so the buffer sees distinct battle ids."""

    def __init__(self, envs, sizes):
        self.envs = envs
        self.offsets = np.cumsum([0] + list(sizes))[:-1].astype(np.uint32)

    def shift(self, i, obs):
        off = int(self.offsets[i])
        if off == 0:
            return obs
        obs = dict(obs)
        obs["battle"] = obs["battle"] + np.uint32(off)
        obs["end_battle"] = obs["end_battle"] + np.uint32(off)
        return obs

    def set_team_weights(self, w):
        for e in self.envs:
            e.set_team_weights(w)

    @property
    def team_probs(self):
        return self.envs[0].team_probs

    @property
    def team_names(self):
        return self.envs[0].team_names

    @property
    def team_counts(self):
        return np.sum([np.asarray(e.team_counts, np.int64) for e in self.envs], 0)

    @property
    def stats(self):
        out = {}
        for e in self.envs:
            for k, v in e.stats.items():
                out[k] = out.get(k, 0) + v
        return out


def pi_logits(net, x):
    """pi's (ataraxos/pi/net.py) forward on the actor's inputs: (logits [R,
    2, CHOICES] over the legal choices, value logits [R, 3], no Q)."""
    value, policy = net(x["mon"].long(), x["hp"], x["active"])
    logits, _ = legal_logits(policy, x["policy"], x["n"])
    return logits, value, None


def sample_q(logp, pick, q):
    """The actor's Q columns: Q's W/D/L (softmax of q [R, 2, C, 3]) summed
    over the data policy and at the pick, each [R, 2, 3]."""
    P = torch.softmax(q.float(), -1)
    q_exp = (logp.exp()[..., None] * P).sum(-2)
    q_pick = P.gather(2, pick.long()[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
    return q_exp, q_pick


class AsyncActor:
    """The trainer's Actor as `launch(obs, slot)` (copies in, the forward,
    copies out, all queued on the current stream) and `collect(handle)`
    (waits for that launch only): picks u8 [R, 2], log-probs f32 [R, 2,
    CHOICES], value logits f32 [R, 3], and with `q` the Q columns (q_exp,
    q_pick [R, 2, 3]) else None, as Actor.act. `inputs` and `logits` are
    the model's (pi's by default). One slot per obs in
    flight (two with --pingpong). `graphs`: the forward and the sampling
    replayed as one CUDA graph of `capacity` rows (the env's battles);
    sampling is Gumbel-max then (exact; torch.multinomial isn't used in a
    capture). The graph reads the net's parameters in place, so the
    learner's updates show up without a recapture. `compile`: the net,
    legal logits and log-softmax through torch.compile (inside the graph with
    `graphs`); the sampling stays eager, so the graph's RNG is torch's own."""

    def __init__(self, net, device, amp, capacity, slots=1, graphs=False, gumbel=False, compile=False,
                 inputs=INPUTS, logits=pi_logits, q=False):
        self.net, self.dev, self.amp = net, torch.device(device), amp
        self.inputs, self.logits_fn, self.q = tuple(inputs), logits, q
        self.cap, self.graphs = capacity, graphs
        self.gumbel = gumbel or graphs
        self._logp = self._logp_eager if not compile else torch.compile(
            self._logp_eager, mode="max-autotune-no-cudagraphs", dynamic=False if graphs else None)
        cuda = self.dev.type == "cuda"
        self.pin_in, self.pin_out, self.events = [], [], []
        self.din = None
        for _ in range(slots):
            self.pin_in.append(None)
            self.pin_out.append(self._alloc_out("cpu", pin=cuda))
            self.events.append(torch.cuda.Event() if cuda else None)
        self.dout = self._alloc_out(self.dev) if graphs else None
        self.graph = None

    def _alloc_out(self, device, pin=False):
        C = self.cap
        out = dict(pick=torch.empty((C, 2), dtype=torch.uint8, device=device, pin_memory=pin),
                   logp=torch.empty((C, 2, CHOICES), dtype=torch.float32, device=device, pin_memory=pin),
                   value=torch.empty((C, 3), dtype=torch.float32, device=device, pin_memory=pin))
        if self.q:
            for k in ("q_exp", "q_pick"):
                out[k] = torch.empty((C, 2, 3), dtype=torch.float32, device=device, pin_memory=pin)
        return out

    def _alloc_in(self, obs, device, pin=False):
        return {k: torch.empty((self.cap,) + obs[k].shape[1:], dtype=torch.from_numpy(obs[k][:0]).dtype,
                               device=device, pin_memory=pin) for k in self.inputs}

    def _logp_eager(self, *xs):
        with torch.autocast(self.dev.type, torch.bfloat16, enabled=self.amp and self.dev.type == "cuda",
                            cache_enabled=False):
            logits, value, q = self.logits_fn(self.net, dict(zip(self.inputs, xs)))
        logp = torch.log_softmax(logits.float(), -1)
        return logp, value.float(), (q.float() if q is not None else None)

    def _forward(self, x):
        logp, value, q = self._logp(*(x[k] for k in self.inputs))
        if self.gumbel:
            u = torch.rand_like(logp).clamp_(1e-10, 1 - 1e-7)
            pick = (logp - torch.log(-torch.log(u))).argmax(-1)
        else:
            pick = torch.multinomial(logp.exp().view(-1, CHOICES), 1).view(-1, 2)
        out = dict(pick=pick.to(torch.uint8), logp=logp, value=value)
        if self.q:
            out["q_exp"], out["q_pick"] = sample_q(logp, pick, q)
        return out

    def _capture(self):
        x = self.din
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._forward(x)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"), torch.no_grad():
            for k, v in self._forward(x).items():
                self.dout[k].copy_(v)

    @torch.no_grad()
    def launch(self, obs, slot=0):
        R = len(obs["n"])
        if R > self.cap:
            raise ValueError(f"{R} rows, capacity {self.cap}")
        cuda = self.dev.type == "cuda"
        if self.pin_in[slot] is None:
            self.pin_in[slot] = self._alloc_in(obs, "cpu", pin=cuda)
        if self.din is None:
            self.din = self._alloc_in(obs, self.dev)
        pin = self.pin_in[slot]
        # the previous launch in this slot was collected, so its copies are done
        for k in self.inputs:
            pin[k][:R].copy_(torch.from_numpy(obs[k]))
            if self.graphs and R < self.cap:
                pin[k][R:] = pin[k][:1]
        rows = self.cap if self.graphs else R
        for k in self.inputs:
            self.din[k][:rows].copy_(pin[k][:rows], non_blocking=True)
        out = self.pin_out[slot]
        if self.graphs:
            if self.graph is None:
                self._capture()
            self.graph.replay()
            src = self.dout
        else:
            src = self._forward({k: v[:R] for k, v in self.din.items()})
        for k in out:
            out[k][:R].copy_(src[k][:R], non_blocking=True)
        if self.events[slot] is not None:
            self.events[slot].record()
        return slot, R

    def collect(self, handle):
        slot, R = handle
        if self.events[slot] is not None:
            self.events[slot].synchronize()
        out = self.pin_out[slot]
        extra = {k: out[k][:R].numpy().copy() for k in ("q_exp", "q_pick")} if self.q else None
        return out["pick"][:R].numpy().copy(), out["logp"][:R].numpy().copy(), out["value"][:R].numpy().copy(), extra

    def act(self, obs):
        return self.collect(self.launch(obs))
