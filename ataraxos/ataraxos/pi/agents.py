"""Agents for the perfect-info env (src/pi_env.hpp) and `match`, as
ataraxos/eval/agents.py's. An agent's `act(obs, side)` picks for the given
side ([R] 0 or 1) of every row: an index into that side's choices (0 for a
side with one choice)."""

import numpy as np
import torch

from ataraxos._ext import ataraxos_env, ataraxos_pi
from ataraxos.pi.net import CHOICES, PiNet, legal_logits, make_net

MOVE, SWITCH = 1, 2
SPECIAL = 8  # libpkmn's Type: Fire and after are special in Gen 1
N_PP = 6  # PP buckets in the move index (src/pi_env.hpp)


def tensors(obs, device):
    t = lambda k, dt: torch.from_numpy(obs[k]).to(device=device, dtype=dt, non_blocking=True)
    return t("mon", torch.long), t("hp", torch.float), t("active", torch.float)


class Agent:
    def act(self, obs, side) -> np.ndarray:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, side):
        n = np.maximum(obs["n"][np.arange(len(side)), side], 1)
        return (self.rng.random(len(side)) * n).astype(np.uint8)


class NetAgent(Agent):
    """Samples the net's policy (argmax with `greedy`)."""

    def __init__(self, net: PiNet, device, amp=True, greedy=False):
        self.net, self.dev, self.amp, self.greedy = net, torch.device(device), amp, greedy

    @torch.no_grad()
    def act(self, obs, side):
        with torch.autocast(self.dev.type, torch.bfloat16, enabled=self.amp and self.dev.type == "cuda"):
            _, policy = self.net(*tensors(obs, self.dev))
        x, _ = legal_logits(policy, torch.from_numpy(obs["policy"]).to(self.dev),
                            torch.from_numpy(obs["n"]).to(self.dev))
        x = x[torch.arange(len(side), device=self.dev), torch.from_numpy(side).to(self.dev)]
        pick = x.argmax(-1) if self.greedy else torch.multinomial(torch.softmax(x, -1), 1)[:, 0]
        return pick.to(torch.uint8).cpu().numpy()


class MaxDamage(Agent):
    """ataraxos's MaxDamage with perfect information: power x accuracy x
    STAB x effectiveness x our attacking stat / the foe's base defending
    stat, Gen 1's physical / special split by type. Our active's stat is its
    real one (the active features), a benched Pokémon's its base stat (the
    features carry no bench stats); the foe is its real active. Base power
    0 or 1 scores 0. It attacks with the best scored move (ties at random),
    else a random choice; a switch alone brings in the Pokémon whose best
    move scores highest."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        d = ataraxos_env.game_data()
        self.bp = d["move_bp"].astype(np.float64)
        self.move_type = d["move_type"]
        self.acc = d["move_accuracy"].astype(np.float64)
        self.types = d["species_types"]
        self.base = d["species_stats"].astype(np.float64)
        self.chart = d["type_chart"].astype(np.float64) / 10

    def _score(self, move_index, species, atk, spc, foe):
        if move_index < 0:
            return 0.0
        move = move_index // N_PP + 1
        bp = self.bp[move]
        if bp <= 1:
            return 0.0
        t = self.move_type[move]
        special = t >= SPECIAL
        stab = 1.5 if t in self.types[species] else 1.0
        if foe > 0:
            ft = self.types[foe]
            eff = self.chart[t, ft[0]] * (self.chart[t, ft[1]] if ft[1] != ft[0] else 1.0)
            defense = self.base[foe, 4 if special else 2]
        else:
            eff, defense = 1.0, 100.0
        acc = self.acc[move] / 100 if self.acc[move] > 0 else 1.0
        return bp * acc * stab * eff * (spc if special else atk) / (2 * defense + 5)

    def act(self, obs, side):
        R = len(side)
        picks = np.zeros(R, np.uint8)
        for i in range(R):
            s = int(side[i])
            n = int(obs["n"][i, s])
            if n <= 1:
                continue
            ch = obs["choices"][i, s, :n]
            kind, slot = ch & 3, ch >> 2
            mons = obs["mon"][i, s]
            foe = int(obs["mon"][i, 1 - s, 0, 0])
            moves = np.flatnonzero(kind == MOVE)
            if len(moves):
                sp = int(mons[0, 0])
                atk, spc = obs["active"][i, s, 0] * 999, obs["active"][i, s, 3] * 999
                scores = np.array([1e-9 if slot[c] == 0 else
                                   self._score(int(mons[0, 2 + slot[c] - 1]), sp, atk, spc, foe) for c in moves])
                if scores.max() > 0:
                    picks[i] = moves[self.rng.choice(np.flatnonzero(scores == scores.max()))]
                else:
                    picks[i] = self.rng.integers(n)
                continue
            scores = []
            for c in range(n):
                m = mons[int(slot[c]) - 1]
                sp = int(m[0])
                if kind[c] != SWITCH or sp <= 0:
                    scores.append(0.0)
                    continue
                atk, spc = self.base[sp, 1], self.base[sp, 4]
                scores.append(max(self._score(int(m[2 + k]), sp, atk, spc, foe) for k in range(4)))
            scores = np.array(scores)
            picks[i] = self.rng.choice(np.flatnonzero(scores == scores.max()))
        return picks


def load_net(path, device, weights="ema") -> PiNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = make_net(ck["config"]["size"]).to(device).eval()
    net.load_state_dict(ck[weights])
    return net


def make_agent(name, device, seed=0, amp=True):
    if name == "random":
        return RandomAgent(seed)
    if name == "maxdamage":
        return MaxDamage(seed)
    return NetAgent(load_net(name, device), device, amp)


def match(env_factory, a: Agent, b: Agent, battles: int, max_steps=100_000):
    """`a` against `b`: `a` is P1 in the even battles and P2 in the odd
    ones; each battle's first game counts. Returns a's (wins, ties, losses),
    a capped game a tie."""
    env = env_factory(battles)
    result = np.full(battles, -2, dtype=np.int64)
    obs = env.reset()
    for step in range(max_steps + 1):
        for bt, g, r in zip(obs["end_battle"], obs["end_game"], obs["end_result"]):
            if g == 0:
                result[bt] = r if bt % 2 == 0 else -r
        if (result != -2).all() or step == max_steps:
            break
        side_a = (obs["battle"] % 2).astype(np.int64)
        picks = np.zeros((len(side_a), 2), np.uint8)
        r = np.arange(len(side_a))
        picks[r, side_a] = a.act(obs, side_a)
        picks[r, 1 - side_a] = b.act(obs, 1 - side_a)
        obs = env.step(picks)
    done = result[result != -2]
    return int((done == 1).sum()), int((done == 0).sum()), int((done == -1).sum())


__all__ = ["Agent", "RandomAgent", "NetAgent", "MaxDamage", "make_agent", "match", "load_net", "CHOICES",
           "ataraxos_pi"]
