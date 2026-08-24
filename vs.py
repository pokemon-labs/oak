import oak
import oak.search

import argparse
from collections import defaultdict

import subprocess

parser = argparse.ArgumentParser(
    description="VS",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--teams",
    type=str,
)


def add_agent_args(parser, prefix):
    parser.add_argument(
        f"--{prefix}eval",
        type=str,
    )
    parser.add_argument(
        f"--{prefix}budget",
        type=str,
    )
    parser.add_argument(
        f"--{prefix}bandit",
        type=str,
    )
    parser.add_argument(
        f"--{prefix}policy-mode",
        type=str,
    )

add_agent_args(parser, "")
add_agent_args(parser, "p1-")
add_agent_args(parser, "p2-")

class Agent:
    def __init__(args, prefix):
        self.eval = None
        self.budget = None
        self.bandit = None
        self.policy_mode = None

        self.eval = args[f"{prefix}eval"] or args.eval or None
        # repeat

        for _ in ["eval", "budget", "bandit", "policy_mode"]:
            if self.__dict__[_] is None:
                assert False, f"{_} not intialized" 

def do_game(args, ):

    teams: list[str] = # read args.teams

    p1_team = random.sample(teams)
    p2_team = random.sample(teams)
    battle, durations, result = oak.parse_battle(f"{p1_team} | {p2_team}")

    while (not result_kind(result)):

        p1_output = oak.search.run(
            battle, durations, P1.budget, P1.bandit, oak.search.Node(), 
        )

        p2_output = oak.search.run(

        )

        p1_choice = p1_output.p1.choices[
            np.sample(weights=oak.search.get_policy(args.p1_policy_mode), 1)[0]
        ]
        p2_choice = p1_output.p1.choices[
            np.sample(weights=oak.search.get_policy(args.p1_policy_mode), 1)[0]
        ]

    

