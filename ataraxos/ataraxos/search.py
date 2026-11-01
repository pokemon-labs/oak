"""Ataraxos's test-time search (the paper's, arXiv 2511.07312; the user's
design for Gen 1, 2026-11-01): at a root decision of ours, `worlds` history
samples (src/history_sampler.hpp) and, in each, `reps` rollouts per legal
choice (src/rollouts.hpp), worlds * reps * n in all; `budget` sets worlds
as budget / (reps * n) when `worlds` is 0. A rollout plays our choice,
then both sides' nets for `depth` turns, and scores our next decision with
the value head, P(win) - P(loss) (a finished game is +1 / 0 / -1). q is
each choice's mean, and the move is drawn from the closed-form update

    pi_search ∝ [e^q rho^alpha pi^beta]^(1 / (alpha + beta))

(rho the training magnet, pi the net's policy at the root).

The nets' caches: ours is the real game's, copied from the actor that
played the root (after it ran on the root's summary, so a rollout's first
decision of ours is forced and not run again); the foe's is its rebuilt
history, run once per world (`prefill`) and copied to that world's
rollouts. Many roots search together: every decision of the rollouts is
one batch per step.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from ataraxos._ext import ataraxos_env
from ataraxos.nets.causal_actor import CausalActor
from ataraxos.nets.causal_net import summarize_grouped
from ataraxos.nets.policy import pad_chunks
from ataraxos.rl.train import magnet


@dataclass
class SearchConfig:
    worlds: int = 0      # history samples per root; 0: from the budget
    reps: int = 2        # rollouts per (world, choice)
    depth: int = 10      # turns before the leaf
    budget: int = 128    # rollouts per root, with worlds 0
    alpha: float = 0.002  # the magnet's coefficient
    beta: float = 0.02    # the net policy's
    magnet: str = "attack_switch"


def win_minus_loss(value_logits: torch.Tensor) -> torch.Tensor:
    """P(win) - P(loss) from the value head's (loss, draw, win) logits."""
    p = torch.softmax(value_logits.float(), -1)
    return p[:, 2] - p[:, 0]


def update(q, logp, choices, n, cfg: SearchConfig):
    """The closed-form update's log-probs [CHOICES] for one root: q [n], the
    net's log-probs [CHOICES] and the root's choices [CHOICES]."""
    legal = torch.arange(len(choices), device=logp.device) < n
    rho = magnet(choices[None].long(), legal[None], cfg.magnet)[0]
    qq = torch.zeros_like(logp)
    qq[:n] = torch.as_tensor(q, dtype=logp.dtype, device=logp.device)
    x = (qq + cfg.alpha * rho + cfg.beta * logp.masked_fill(~legal, 0)) / (cfg.alpha + cfg.beta)
    return torch.log_softmax(x.masked_fill(~legal, -1e9), -1)


class Search:
    """`teams`: the history sampler's pool (Showdown export files); `ratio`
    its rank-decay prior."""

    def __init__(self, net, teams, device, cfg: SearchConfig = SearchConfig(), threads=8, amp=True,
                 ratio=0.1, seed=0):
        self.net, self.device, self.cfg, self.amp = net, torch.device(device), cfg, amp
        self.r = ataraxos_env.Rollouts(list(teams), net.c.window - 1, threads=threads, state=net.c.state,
                                       ratio=ratio)
        self.actor = None
        self.rng = np.random.default_rng(seed)
        self.stats = {"roots": 0, "rollouts": 0, "steps": 0, "differs": 0}
        self.times = {"sample": 0.0, "prefill": 0.0, "rollouts": 0.0, "update": 0.0}

    def worlds(self, n):
        c = self.cfg
        return c.worlds or max(1, c.budget // (c.reps * n))

    def _actor(self, rollouts):
        if self.actor is None or self.actor.slots < 2 * rollouts:
            self.actor = None
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
            self.actor = CausalActor(self.net, max(rollouts, 64), self.device, amp=self.amp)
        return self.actor

    @torch.no_grad()
    def prefill(self, actor, keys, lay):
        """Runs the foe's summaries of each world into its cache at `keys`
        [W] (world w's first rollout's foe key): summary j of a world at
        position t0 + j."""
        net, dev, R = self.net, self.device, actor.R
        summaries, off = lay["summaries"], lay["offsets"]
        count = np.diff(summaries)
        t0 = torch.from_numpy(lay["t0"]).to(dev)
        keys_t = torch.from_numpy(keys).to(dev)
        actor.game[keys_t] = 0
        actor.t[keys_t] = t0
        tokens = torch.from_numpy(lay["tokens"])
        for j in range(int(count.max()) if len(count) else 0):
            w = np.flatnonzero(count > j)
            s = summaries[w] + j  # each world's summary j
            k = torch.from_numpy(keys[w]).to(dev)
            o = torch.from_numpy(np.stack([off[s], off[s + 1]], 1).reshape(-1))
            # the chunks of rows s, as pad_chunks reads consecutive offsets
            lens = (o[1::2] - o[0::2])
            flat = torch.cat([tokens[a:b] for a, b in zip(o[0::2].tolist(), o[1::2].tolist())]) \
                if lens.sum() else tokens[:0]
            offs = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lens, 0)])
            chunks = pad_chunks(flat, offs, net.c.max_lines)
            lengths = np.minimum(lens.numpy(), net.c.max_lines)
            request = torch.from_numpy(lay["request"][s]).to(dev)
            t = actor.t[k]
            with torch.autocast(dev.type, dtype=actor.dtype, enabled=actor.amp):
                x = summarize_grouped(net, chunks, lengths, request, actor.k.dtype)
                x = x + net.pos[t.clamp(max=net.c.max_pos - 1)]
                valid = torch.arange(R, device=dev)[None] < t[:, None]
                for l, layer in enumerate(net.layers):
                    x, kl, vl = layer.step(x, actor.k[l, k], actor.v[l, k], valid)
                    actor.k[l, k, t % R] = kl.to(actor.k.dtype)
                    actor.v[l, k, t % R] = vl.to(actor.v.dtype)
            actor.t[k] = t + 1

    def run(self, env, real: CausalActor, obs, logp, seed=None, truth=False, check=None):
        """Searches decisions `obs` (rows of the SelfPlay `env`'s output, all
        ours, the real actor having just run on them; `logp` [D, CHOICES]
        its log-probs) and returns (picks u8 [D], q list, search log-probs
        [D, CHOICES]). `truth`: every world has the foe's true sets. `check`
        (tests, a dict): fills in "ours", the copied cache rerun on the
        root summary against `logp` (max abs), and "foe", per root whose
        rollouts start with a foe decision: (root, the mean of their first
        foe log-probs, that decision's request, choices and n)."""
        c = self.cfg

        def add(r, i, seed):
            r.add_selfplay(env, int(obs["battle"][i]), int(obs["player"][i]), self.worlds(int(obs["n"][i])),
                           c.reps, c.depth, seed, truth)

        keys = obs["battle"].astype(np.int64) * 2 + obs["player"]
        return self.search(add, real, keys, obs["player"], obs, logp, seed, check)

    def run_client(self, client, real: CausalActor, key, obs, logp, seed=None):
        """One search at the ladder Client's request: `obs` its one decision,
        which the real actor ran at cache key `key`."""
        c = self.cfg

        def add(r, i, seed):
            r.add_client(client, self.worlds(int(obs["n"][0])), c.reps, c.depth, seed)

        return self.search(add, real, np.array([key], np.int64), np.array([client.viewer()]), obs, logp, seed)

    @torch.no_grad()
    def search(self, add, real, keys, players, obs, logp, seed=None, check=None):
        """`add(rollouts, i, seed)` adds root i; `keys` [D] the roots' keys
        in the real actor's cache, `players` [D] their sides."""
        D = len(obs["n"])
        seed = int(self.rng.integers(1 << 62)) if seed is None else seed
        r = self.r
        tic = time.perf_counter()

        def lap(what):
            nonlocal tic
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            self.times[what] += now - tic
            tic = now

        r.clear()
        for i in range(D):
            add(r, i, seed + i)
        lay = r.layout()
        N = r.rollouts
        lap("sample")
        actor = self._actor(N)
        dev = self.device
        world_of = lay["world"].astype(np.int64)
        root_of = lay["root"].astype(np.int64)[world_of]
        players = np.asarray(players, np.int64)
        us = players[root_of]
        idx = np.arange(N, dtype=np.int64)
        ours = torch.from_numpy(idx * 2 + us).to(dev)
        foes = torch.from_numpy(idx * 2 + 1 - us).to(dev)
        # ours: the real game's cache
        src = torch.from_numpy(np.asarray(keys, np.int64)[root_of]).to(real.device)
        actor.k[:, ours] = real.k[:, src].to(dev, actor.k.dtype)
        actor.v[:, ours] = real.v[:, src].to(dev, actor.v.dtype)
        actor.t[ours] = real.t[src].to(dev)
        actor.game[ours] = 0
        # the foe's: each world's history, then copied to its rollouts
        _, first = np.unique(world_of, return_index=True)
        first_keys = idx[first] * 2 + 1 - us[first]
        self.prefill(actor, first_keys, lay)
        fk = torch.from_numpy(first_keys[world_of]).to(dev)
        actor.k[:, foes] = actor.k[:, fk]
        actor.v[:, foes] = actor.v[:, fk]
        actor.t[foes] = actor.t[fk]
        actor.game[foes] = 0
        if check is not None:
            # the root summary again, on each root's first rollout's copy
            _, root_first = np.unique(root_of, return_index=True)
            again = dict(obs)
            again["battle"] = idx[root_first].astype(np.uint32)
            again["player"] = us[root_first].astype(np.uint8)
            again["game"] = np.zeros(D, np.uint32)
            k = torch.from_numpy(idx[root_first] * 2 + us[root_first]).to(dev)
            t = actor.t[k].clone()
            actor.t[k] = t - 1
            _, lp_again, _ = actor.act(again)
            actor.t[k] = t
            legal = torch.arange(logp.shape[1], device=logp.device)[None] < torch.from_numpy(obs["n"]).to(logp.device)[:, None]
            check["ours"] = float((lp_again.to(logp.device) - logp).abs().masked_fill(~legal, 0).max())
        lap("prefill")

        out = r.start(seed)
        steps = 0
        first = check is not None
        while len(out["n"]):
            picks, lp_r, value = actor.act(out)
            if first:
                first = False
                foe_rows = np.flatnonzero(out["player"] != players[out["root"]])
                check["foe"] = []
                for i in range(D):
                    rows = foe_rows[out["root"][foe_rows] == i]
                    if len(rows):
                        lq = lp_r[torch.from_numpy(rows).to(dev)].mean(0)
                        check["foe"].append((i, lq, out["request"][rows[0]], out["choices"][rows[0]], out["n"][rows[0]]))
            out = r.step(picks, win_minus_loss(value).cpu().numpy().astype(np.float32))
            steps += 1
        assert r.done
        lap("rollouts")

        picks = np.zeros(D, np.uint8)
        qs, lps = [], torch.empty_like(logp)
        choices = torch.from_numpy(np.ascontiguousarray(obs["choices"])).to(logp.device)
        for i in range(D):
            q = r.q(i)
            lp = update(q, logp[i], choices[i], int(obs["n"][i]), self.cfg)
            picks[i] = torch.multinomial(lp.exp(), 1).item()
            qs.append(q)
            lps[i] = lp
        lap("update")
        s = self.stats
        s["roots"] += D
        s["rollouts"] += N
        s["steps"] += steps
        s["differs"] += int(lay["differs"].sum())
        return picks, qs, lps
