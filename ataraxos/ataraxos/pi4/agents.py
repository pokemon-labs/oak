"""pi4's evals (2026-11-24, docs/pi4.md): pi4 against anything, in rby4's env
(ataraxos_env's SelfPlay with `state`), each side reading its own input from
the same battles: pi4 its perfect-info rows (`pi4_obs(False)`, one per
decision), a pi1-3 net oak's battle encoding (`pi_obs`, as
ataraxos/eval/pi_h2h.py), and a Showdown-view agent (rby4, maxdamage,
random: ataraxos/eval/agents.py) its tokens and request. Policy only, every
net sampling its raw policy.

`match(env_factory, a, b, battles)`: `a` plays P1 in the even battles and P2
in the odd ones; each battle's first game counts; a's (wins, ties, losses).
"""

import numpy as np
import torch

from ataraxos.eval import agents as showdown
from ataraxos.pi import agents as pi
from ataraxos.pi.net import SIZES as PI_SIZES
from ataraxos.pi4.net import is_pi4, load_net

PI4_INPUTS = ("state", "request", "choices", "n")
PI_INPUTS = ("mon", "hp", "active", "policy", "n")


class Pi4Agent:
    """A pi4 net: samples the deciding player's policy."""

    def __init__(self, net, device, amp=True):
        self.net, self.dev, self.amp = net, torch.device(device), amp

    def start(self, battles):
        pass

    @torch.no_grad()
    def act(self, env, obs, rows):
        o = env.pi4_obs(False)
        x = [torch.from_numpy(np.ascontiguousarray(o[k][rows])).to(self.dev) for k in PI4_INPUTS]
        with torch.autocast(self.dev.type, torch.bfloat16, enabled=self.amp and self.dev.type == "cuda"):
            _, logits = self.net(*x)
        side = torch.from_numpy(obs["player"][rows].astype(np.int64)).to(self.dev)
        lg = logits.float()[torch.arange(len(rows), device=self.dev), side]
        return torch.multinomial(torch.softmax(lg, -1), 1)[:, 0].to(torch.uint8).cpu().numpy()


class PiAgent:
    """A pi1-3 net (ataraxos/pi/net.py) on oak's battle encoding."""

    def __init__(self, agent):
        self.agent = agent

    def start(self, battles):
        pass

    def act(self, env, obs, rows):
        po = env.pi_obs()
        sub = {k: np.ascontiguousarray(po[k][rows]) for k in PI_INPUTS}
        return self.agent.act(sub, obs["player"][rows].astype(np.int64))


class ShowdownAgent:
    """ataraxos/eval/agents.py's: reads the tokens and request."""

    def __init__(self, agent):
        self.agent = agent

    def start(self, battles):
        self.agent.start(battles)

    def act(self, env, obs, rows):
        return self.agent.act(showdown.subset(obs, rows))


def make_agent(name, device, seed=0, amp=True):
    """"random", "maxdamage", or a checkpoint (its EMA weights): pi4, pi1-3
    or a Showdown-view net, by its config's size."""
    if name in ("random", "maxdamage"):
        return ShowdownAgent(showdown.make_agent(name, device, seed, amp))
    size = torch.load(name, map_location="cpu", weights_only=False, mmap=True)["config"]["size"]
    if is_pi4(size):
        return Pi4Agent(load_net(name, device), device, amp)
    if size in PI_SIZES:
        return PiAgent(pi.NetAgent(pi.load_net(name, device), device, amp))
    return ShowdownAgent(showdown.make_agent(name, device, seed, amp))


def match(env_factory, a, b, battles, max_steps=100_000):
    env = env_factory(battles)
    a.start(battles)
    b.start(battles)
    result = np.full(battles, -2, dtype=np.int64)
    obs = env.reset()
    for step in range(max_steps + 1):
        for bt, g, r in zip(obs["end_battle"], obs["end_game"], obs["end_result"]):
            if g == 0 and result[bt] == -2:
                result[bt] = r if bt % 2 == 0 else -r
        if (result != -2).all() or step == max_steps:
            break
        mine = (obs["battle"] % 2) == obs["player"]
        picks = np.zeros(len(obs["n"]), np.uint8)
        for rows, agent in ((np.flatnonzero(mine), a), (np.flatnonzero(~mine), b)):
            if len(rows):
                picks[rows] = agent.act(env, obs, rows)
        obs = env.step(picks)
    done = result[result != -2]
    return int((done == 1).sum()), int((done == 0).sum()), int((done == -1).sum())


def eval_env(seed, teams, max_turns, threads):
    from ataraxos._ext import ataraxos_env
    return lambda b: ataraxos_env.SelfPlay(b, seed, teams=list(teams), max_turns=max_turns, threads=threads,
                                           state=True)
