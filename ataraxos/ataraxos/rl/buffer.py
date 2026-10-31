"""Self-play trajectories (draft, 2026-10-23), stored per game.

Each player's game is one trajectory: its decisions in order, then the
game's result. An iteration trains on the games that ended during it (the
user's, 2026-10-23: iterate until K games end); a game still going carries
over, so its early decisions came from older weights, which the ratio clip
and the KL to the data policy are for.

The store is columnar: each step's decisions are appended as arrays, and a
trajectory is a list of global decision ids. `take` gathers the finished
trajectories and compacts the store to the open ones. A game ended by the
turn cap goes by `at_cap` (the user's, 2026-10-26): "drop" never trains on
it, "draw" keeps it as the tie the env reports, "bootstrap" keeps it with
no result, its last decision's target its own value.

A decision's value is three logits, (loss, draw, win) (the causal net's,
2026-10-27); with the env's `state` on, each decision's bookkeeping state is
kept too.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

COLUMNS_SAVED = ("request", "choices", "n", "pick", "logp", "value")


@dataclass
class Data:
    """Finished trajectories, each contiguous: trajectory g is decisions
    start[g] : start[g] + length[g]."""
    tokens: np.ndarray    # [T, COLUMNS], each decision's chunk (at most max_lines)
    offsets: np.ndarray   # [N + 1]
    request: np.ndarray   # [N, REQUEST_FIELDS]
    choices: np.ndarray   # [N, CHOICES]
    n: np.ndarray         # [N]
    pick: np.ndarray      # [N]
    logp: np.ndarray      # [N, CHOICES] the data policy's
    value: np.ndarray     # [N, 3] the data policy's value logits, (loss, draw, win)
    start: np.ndarray     # [G]
    length: np.ndarray    # [G]
    reward: np.ndarray    # [G] the player's: 1, 0, -1
    player: np.ndarray    # [G] 0 P1, 1 P2
    state: np.ndarray = None  # [N, STATE_FIELDS], with the env's `state`
    truncated: np.ndarray = None  # [G] bool: capped under "bootstrap", `reward` unused


AT_CAP = ("drop", "draw", "bootstrap")


class Buffer:
    def __init__(self, max_lines: int, at_cap: str = "draw", sides: int = 2):
        """`sides`: the trajectories a game gives, 1 when only one side's
        decisions are added (the exploiter run's)."""
        if at_cap not in AT_CAP:
            raise ValueError(f"at_cap must be one of {AT_CAP}, not {at_cap!r}")
        self.max_lines = max_lines
        self.at_cap = at_cap
        self.sides = sides
        self.parts = []          # per step: dict of arrays
        self.size = 0            # decisions stored
        self.open = {}           # (battle, player) -> (game, [ids])
        self.done = []           # (ids, reward, player, truncated)
        self.columns = COLUMNS_SAVED  # plus "state" once a decision has one

    def end(self, obs):
        """Closes the games that ended in this env call. Call before `add`.
        A game ended by the turn cap goes by `at_cap`; dropped, its
        decisions are never trained on and it doesn't count."""
        for b, g, r, c in zip(obs["end_battle"].tolist(), obs["end_game"].tolist(), obs["end_result"].tolist(),
                              obs["end_capped"].tolist()):
            for p, sign in ((0, 1), (1, -1)):
                t = self.open.pop((b, p), None)
                if t is not None and t[0] == g and t[1] and not (c and self.at_cap == "drop"):
                    self.done.append((t[1], sign * r, p, bool(c) and self.at_cap == "bootstrap"))

    def add(self, obs, pick, logp, value):
        D = len(obs["n"])
        if D == 0:
            return
        off = obs["offsets"]
        lengths = np.minimum(off[1:] - off[:-1], self.max_lines)
        first = off[1:] - lengths
        idx = np.repeat(first - np.r_[0, np.cumsum(lengths)[:-1]], lengths) + np.arange(lengths.sum())
        part = dict(tokens=obs["tokens"][idx], lengths=lengths, request=obs["request"], choices=obs["choices"],
                    n=obs["n"], pick=pick, logp=logp.cpu().numpy(), value=value.cpu().numpy())
        if "state" in obs:
            part["state"] = obs["state"]
            self.columns = COLUMNS_SAVED + ("state",)
        self.parts.append(part)
        ids = range(self.size, self.size + D)
        for i, b, p, g in zip(ids, obs["battle"].tolist(), obs["player"].tolist(), obs["game"].tolist()):
            t = self.open.get((b, p))
            if t is None or t[0] != g:
                t = (g, [])
                self.open[(b, p)] = t
            t[1].append(i)
        self.size += D

    def games(self) -> int:
        """Finished games (`sides` trajectories each) not yet taken."""
        return len(self.done) // self.sides

    def _columns(self):
        cat = {k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}
        cat["offsets"] = np.r_[0, np.cumsum(cat.pop("lengths"))].astype(np.int64)
        return cat

    def _gather(self, cat, ids):
        off = cat["offsets"]
        lengths = off[ids + 1] - off[ids]
        idx = np.repeat(off[ids] - np.r_[0, np.cumsum(lengths)[:-1]], lengths) + np.arange(lengths.sum())
        out = {k: cat[k][ids] for k in self.columns}
        out["tokens"] = cat["tokens"][idx]
        out["lengths"] = lengths
        return out

    def take(self) -> Data:
        """The finished trajectories; the store keeps only the open ones."""
        cat = self._columns()
        ids = np.array([i for t, _, _, _ in self.done for i in t], dtype=np.int64)
        length = np.array([len(t) for t, _, _, _ in self.done], dtype=np.int64)
        reward = np.array([r for _, r, _, _ in self.done], dtype=np.int64)
        player = np.array([p for _, _, p, _ in self.done], dtype=np.int64)
        truncated = np.array([u for _, _, _, u in self.done], dtype=bool)
        g = self._gather(cat, ids)
        data = Data(tokens=g["tokens"], offsets=np.r_[0, np.cumsum(g["lengths"])].astype(np.int64),
                    request=g["request"], choices=g["choices"], n=g["n"], pick=g["pick"], logp=g["logp"],
                    value=g["value"], start=np.r_[0, np.cumsum(length)[:-1]].astype(np.int64),
                    length=length, reward=reward, player=player, state=g.get("state"), truncated=truncated)
        # Compact: renumber the open trajectories' decisions from 0.
        keep = np.array([i for _, t in self.open.values() for i in t], dtype=np.int64)
        if len(keep):
            k = self._gather(cat, keep)
            self.parts = [dict(tokens=k["tokens"], lengths=k["lengths"],
                               **{c: k[c] for c in self.columns})]
            new = {old: j for j, old in enumerate(keep.tolist())}
            self.open = {key: (game, [new[i] for i in t]) for key, (game, t) in self.open.items()}
        else:
            self.parts = []
        self.size = len(keep)
        self.done = []
        return data


def _bootstrap(V, data: Data):
    """A truncated trajectory (capped under "bootstrap") ends on its last
    decision's own value, not a result: the value at the cap itself isn't
    computed, since the env sends no decision there."""
    if data.truncated is None or not data.truncated.any():
        return
    g = torch.from_numpy(np.flatnonzero(data.truncated))
    last = torch.from_numpy(data.length[data.truncated])
    V[g, last] = V[g, last - 1]


@torch.no_grad()
def returns_and_advantages(data: Data, td_lambda: float, gae_lambda: float):
    """Stratego's buffer.process_data, per trajectory, over (loss, draw, win)
    distributions: rewards terminal only, gamma 1. A step's target is the
    next step's distribution, or the result's one-hot at the last step (a
    truncated trajectory's own distribution there); returns are TD(td_lambda)
    over the distribution, advantages GAE(gae_lambda) over the expected
    outcome P(win) - P(loss), the paper's, so its 0.01 filter threshold keeps
    its scale. Returns (returns [N, 3], advantages [N])."""
    G, T = len(data.length), int(data.length.max())
    N = len(data.n)
    pos = np.arange(N) - np.repeat(data.start, data.length)
    row = np.repeat(np.arange(G), data.length)
    V = torch.zeros(G, T + 1, 3)
    V[row, pos] = torch.softmax(torch.from_numpy(data.value).float(), -1)
    V[torch.arange(G), torch.from_numpy(data.length)] = F.one_hot(torch.from_numpy(data.reward + 1), 3).float()
    _bootstrap(V, data)
    valid = torch.zeros(G, T, dtype=torch.bool)
    valid[row, pos] = True
    delta = (V[:, 1:] - V[:, :-1]) * valid[..., None]  # [G, T, 3]
    e = delta[..., 2] - delta[..., 0]
    ret = torch.zeros(G, T, 3)
    adv = torch.zeros(G, T)
    acc, sacc = torch.zeros(G, 3), torch.zeros(G)
    for t in range(T - 1, -1, -1):
        nxt = valid[:, t + 1].float() if t + 1 < T else torch.zeros(G)
        acc = delta[:, t] + td_lambda * acc * nxt[:, None]
        sacc = e[:, t] + gae_lambda * sacc * nxt
        ret[:, t] = acc
        adv[:, t] = sacc
    ret = ret + V[:, :-1]
    return ret[row, pos], adv[row, pos]


def minibatches(data: Data, keep: np.ndarray, size: int, rng):
    """Shuffled groups of whole trajectories, each with about `size` kept
    decisions. Yields arrays of trajectory indices."""
    kept = np.add.reduceat(keep.astype(np.int64), data.start) if len(data.start) else np.zeros(0)
    group, count = [], 0
    for g in rng.permutation(len(data.length)):
        if kept[g] == 0:
            continue
        group.append(g)
        count += kept[g]
        if count >= size:
            yield np.array(group)
            group, count = [], 0
    if group:
        yield np.array(group)


def split(data: Data, group: np.ndarray, keep: np.ndarray, size: int) -> list:
    """A minibatch's trajectories `group`, in order, cut into micro-batches
    of whole trajectories with about `size` kept decisions each (one
    trajectory with more is its own); `size` 0 keeps the group whole.
    Every micro-batch has a kept decision."""
    if size <= 0:
        return [group]
    kept = np.add.reduceat(keep.astype(np.int64), data.start)[group]
    out, part, count = [], [], 0
    for g, k in zip(group.tolist(), kept.tolist()):
        if k == 0:
            continue
        part.append(g)
        count += k
        if count >= size:
            out.append(np.array(part))
            part, count = [], 0
    if part:
        out.append(np.array(part))
    return out
