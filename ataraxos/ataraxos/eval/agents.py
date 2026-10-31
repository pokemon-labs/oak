"""Agents and matches (2026-10-25): anything that picks from what a Showdown
player sees.

An agent gets the env's decisions (`act(obs)`, obs as ataraxos/_ext/ataraxos_env.cpp
gives it, or a subset of one: battle, game, player, tokens, offsets,
request, choices, n) and returns one pick per decision, an index into its
choices. That is the net's whole input: its view of the log as tokens and
the request, which src/client.hpp gives on Showdown exactly as the env does
in training (tests/cpp/check_client.cpp). So every agent here reads only what a
Showdown player can, and the same agent can play the ladder once a driver
feeds it the Client's output.

- NetAgent: a CausalPolicy, through the causal actor's key/value cache
  (ataraxos/nets/causal_actor.py).
- RandomAgent: uniform over the legal choices.
- MaxDamage: the move with the highest expected damage score against the
  foe's active Pokémon, from the request and the log alone (below).

`match(env_factory, a, b, battles, seed)` plays `a` against `b`: `a` is P1
in the even battles and P2 in the odd ones; each battle's first game
counts. `make_agent(name, ...)` turns an anchor name into an agent:
"random", "maxdamage", or a checkpoint path.
"""


import numpy as np
import torch

from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.nets.causal_actor import make_actor  # noqa: E402
from ataraxos.nets.causal_net import make_net  # noqa: E402
from ataraxos.nets.policy import SIZES  # noqa: E402

# request field layout (src/tokens.hpp's encode)
HEADER, ACTIVE_MOVE, SIDE = 7, 4, 14
SIDE0 = HEADER + 4 * ACTIVE_MOVE
F_SPECIES, F_ACTIVE, F_ATK, F_DEF, F_SPE, F_SPC, F_MOVES = 0, 5, 6, 7, 8, 9, 10
# token columns (src/tokens.hpp's Column)
KIND, ACTOR_COL, SPECIES_COL = 0, 2, 6
SPECIAL = 8  # libpkmn's Type: Fire and after are special in Gen 1
MOVE, SWITCH = 1, 2  # libpkmn's choice type


def subset(obs, rows):
    """The decisions `rows` (a numpy index) of an env output, ends dropped."""
    off = obs["offsets"]
    lengths = off[rows + 1] - off[rows]
    idx = np.repeat(off[rows] - np.cumsum(np.r_[0, lengths[:-1]]), lengths) + np.arange(lengths.sum())
    out = {k: obs[k][rows] for k in ("battle", "game", "player", "request", "choices", "n", "state") if k in obs}
    out["tokens"] = obs["tokens"][idx]
    out["offsets"] = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    return out


class Agent:
    """`start(battles)` before a match; `act(obs)` -> picks u8 [D]."""

    def start(self, battles: int):
        pass

    def act(self, obs) -> np.ndarray:
        raise NotImplementedError


class NetAgent(Agent):
    def __init__(self, net, device, amp=True, greedy=False):
        self.net, self.device, self.amp, self.greedy = net, device, amp, greedy

    def start(self, battles):
        self.actor = make_actor(self.net, battles, self.device, amp=self.amp)

    def act(self, obs):
        return self.actor.act(obs, greedy=self.greedy)[0]


class RandomAgent(Agent):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return (self.rng.random(len(obs["n"])) * obs["n"]).astype(np.uint8)


class MaxDamage(Agent):
    """The usual first benchmark for Pokémon bots. A move's score is

        power x accuracy x STAB x effectiveness x our attacking stat
          / the foe's base defending stat

    with Gen 1's split by the move's type (special types use Special
    against Special). Power is libpkmn's base power; fixed-damage, counter
    and OHKO moves (base power 0 or 1) and status moves score 0. The foe is
    the last species the log switched in on its side, tracked per battle and
    game; its set is never guessed. It attacks with the best scored move,
    ties broken at random; with nothing scored, or no move to pick, it takes
    a random choice, except that a switch alone (a faint or a forced switch)
    brings in the Pokémon whose best move scores highest. Boosts, crits and
    screens are ignored."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        d = ataraxos_env.game_data()
        self.bp = d["move_bp"].astype(np.float64)
        self.move_type = d["move_type"]
        self.acc = d["move_accuracy"].astype(np.float64)
        self.types = d["species_types"]
        self.base = d["species_stats"].astype(np.float64)
        self.chart = d["type_chart"].astype(np.float64) / 10
        self.switch_kinds = (d["ARG_SWITCH"],)
        self.foe = {}  # (battle, player) -> (game, species)

    def start(self, battles):
        self.foe = {}

    def _track(self, obs):
        off, tok = obs["offsets"], obs["tokens"]
        for i, (b, p, g) in enumerate(zip(obs["battle"].tolist(), obs["player"].tolist(), obs["game"].tolist())):
            key = (b, p)
            old = self.foe.get(key)
            species = old[1] if old is not None and old[0] == g else 0
            t = tok[off[i]:off[i + 1]]
            # ident 1 + (side << 3 | position): the foe's side is 1
            hit = np.isin(t[:, KIND], self.switch_kinds) & (((t[:, ACTOR_COL] - 1) >> 3) == 1)
            if hit.any():
                species = int(t[np.flatnonzero(hit)[-1], SPECIES_COL])
            self.foe[key] = (g, species)

    def _score(self, move, species, atk, spc, foe):
        if move <= 0:
            return 0.0
        bp = self.bp[move]
        if bp <= 1:
            return 0.0
        t = self.move_type[move]
        special = t >= SPECIAL
        own = self.types[species]
        stab = 1.5 if t in own else 1.0
        if foe > 0:
            ft = self.types[foe]
            eff = self.chart[t, ft[0]] * (self.chart[t, ft[1]] if ft[1] != ft[0] else 1.0)
            defense = self.base[foe, 4 if special else 2]
        else:
            eff, defense = 1.0, 100.0
        acc = self.acc[move] / 100 if self.acc[move] > 0 else 1.0
        return bp * acc * stab * eff * (spc if special else atk) / (2 * defense + 5)

    def _best(self, req_side, foe):
        sp = int(req_side[F_SPECIES])
        if sp <= 0:
            return 0.0
        return max(self._score(int(req_side[F_MOVES + k]), sp, float(req_side[F_ATK]), float(req_side[F_SPC]), foe)
                   for k in range(4))

    def act(self, obs):
        self._track(obs)
        D = len(obs["n"])
        picks = np.zeros(D, np.uint8)
        for i in range(D):
            n = int(obs["n"][i])
            ch = obs["choices"][i, :n]
            kind, slot = ch & 3, ch >> 2
            req = obs["request"][i]
            sides = req[SIDE0:].reshape(6, SIDE)
            foe = self.foe[(int(obs["battle"][i]), int(obs["player"][i]))][1]
            active = sides[np.flatnonzero(sides[:, F_ACTIVE])[0]] if sides[:, F_ACTIVE].any() else sides[0]
            moves = np.flatnonzero(kind == MOVE)
            if len(moves):
                scores = np.zeros(len(moves))
                for j, c in enumerate(moves):
                    k = int(slot[c])
                    if k == 0:  # Struggle: the only move there is
                        scores[j] = 1e-9
                        continue
                    m = int(req[HEADER + (k - 1) * ACTIVE_MOVE])
                    scores[j] = self._score(m, int(active[F_SPECIES]), float(active[F_ATK]), float(active[F_SPC]), foe)
                if scores.max() > 0:
                    best = np.flatnonzero(scores == scores.max())
                    picks[i] = moves[self.rng.choice(best)]
                    continue
                picks[i] = self.rng.integers(n)
                continue
            # switches only: the one whose best move scores highest
            scores = np.array([self._best(sides[int(slot[c]) - 1], foe) for c in range(n)])
            best = np.flatnonzero(scores == scores.max())
            picks[i] = self.rng.choice(best)
        return picks


def load_policy(path, device, weights="ema"):
    """A checkpoint's weights ("ema" or "net") as a CausalPolicy."""
    ck = torch.load(path, map_location=device, weights_only=False)
    net = make_net(SIZES[ck["config"]["size"]]).to(device).eval()
    net.load_state_dict(ck[weights])
    return net


def make_agent(name, device, seed=0, amp=True):
    """An anchor by name: "random", "maxdamage", or a checkpoint path (its
    EMA weights)."""
    if name == "random":
        return RandomAgent(seed)
    if name == "maxdamage":
        return MaxDamage(seed)
    return NetAgent(load_policy(name, device), device, amp)


def match(env_factory, a: Agent, b: Agent, battles: int, max_steps=100_000):
    """`a` against `b` over `battles` battles: `a` plays P1 in the even
    battles and P2 in the odd ones, and each battle's first game counts, so
    long games aren't undercounted. Returns `a`'s (wins, ties, losses);
    a game ended by the env's turn cap is a tie."""
    env = env_factory(battles)
    a.start(battles)
    b.start(battles)
    result = np.full(battles, -2, dtype=np.int64)  # a's: 1 win, 0 tie, -1 loss
    obs = env.reset()  # a game with no decision to send can end here already
    for step in range(max_steps + 1):
        for bt, g, r in zip(obs["end_battle"], obs["end_game"], obs["end_result"]):
            if g == 0:
                result[bt] = r if bt % 2 == 0 else -r
        if (result != -2).all() or step == max_steps:
            break
        mine = (obs["battle"] % 2) == obs["player"]
        picks = np.zeros(len(obs["n"]), np.uint8)
        for rows, agent in ((np.flatnonzero(mine), a), (np.flatnonzero(~mine), b)):
            if len(rows):
                picks[rows] = agent.act(subset(obs, rows))
        obs = env.step(picks)
    done = result[result != -2]
    return int((done == 1).sum()), int((done == 0).sum()), int((done == -1).sum())
