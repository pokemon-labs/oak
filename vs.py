# import oak
# import oak.search

# import argparse
# from collections import defaultdict

# import subprocess

# parser = argparse.ArgumentParser(
#     description="VS",
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# )

# parser.add_argument(
#     "--teams",
#     type=str,
# )


# def add_agent_args(parser, prefix):
#     parser.add_argument(
#         f"--{prefix}eval",
#         type=str,
#     )
#     parser.add_argument(
#         f"--{prefix}budget",
#         type=str,
#     )
#     parser.add_argument(
#         f"--{prefix}bandit",
#         type=str,
#     )
#     parser.add_argument(
#         f"--{prefix}policy-mode",
#         type=str,
#     )

# add_agent_args(parser, "")
# add_agent_args(parser, "p1-")
# add_agent_args(parser, "p2-")

# class Agent:
#     def __init__(args, prefix):
#         self.eval = None
#         self.budget = None
#         self.bandit = None
#         self.policy_mode = None

#         self.eval = args[f"{prefix}eval"] or args.eval or None
#         # repeat

#         for _ in ["eval", "budget", "bandit", "policy_mode"]:
#             if self.__dict__[_] is None:
#                 assert False, f"{_} not intialized"

# def do_game(args, ):

#     teams: list[str] = oak.load_teams()

#     p1_team = random.sample(teams)
#     p2_team = random.sample(teams)
#     battle, durations, result = oak.parse_battle(f"{p1_team} | {p2_team}")

#     while (not result_kind(result)):

#         p1_output = oak.search.run(
#             battle, durations, P1.budget, P1.bandit, oak.search.Node(),
#         )

#         p2_output = oak.search.run(

#         )

#         p1_choice = p1_output.p1.choices[
#             np.sample(weights=oak.search.get_policy(args.p1_policy_mode), 1)[0]
#         ]
#         p2_choice = p1_output.p1.choices[
#             np.sample(weights=oak.search.get_policy(args.p1_policy_mode), 1)[0]
#         ]
import oak
import oak.search

import oldoak
import oldoak.search

import random
import numpy as np

network_path = "/home/user/battle-2026-08-23-12:55:38/200.battle.net"
budget = "400ms"
bandit = "pucb-1.0"

teams_path = "/home/user/teams1500"
teams = []
with open(teams_path) as f:
    for x in f:
        teams.append(x.strip())

network = oak.search.Network()
network.read_parameters(network_path)
network.quantize()

result_types = {}
result_types[1] = 0
result_types[2] = 0
result_types[3] = 0

def play_game(p1_team, p2_team, p1_s1_cache, p1_s2_cache, agent) -> int:
    battle, durations, result = oak.parse_battle(f"{p1_team} | {p2_team}")
    result_type = 0
    while result_type == 0:
        p1_choices, p2_choices = oak.choices(battle, result)

        p1_output = oak.search.run(
            battle,
            durations,
            oak.search.parse_budget(budget),
            bandit=oak.search.parse_bandit(bandit),
            heap=oak.search.Node(),
            eval=network,
            output=oak.search.Output(),
            p1_cache=p1_s1_cache,
            p2_cache=p1_s2_cache,
        )

        p2_output = oldoak.search.search(
            oldoak.Battle(battle.bytes()),
            oldoak.Durations(durations.bytes()),
            result,
            oldoak.search.Heap(),
            agent,
        )

        print(p1_output.iterations)
        print(p2_output.iterations)

        p1_index = np.argmax(p1_output.p1.empirical)
        p2_index = np.argmax(p2_output.p2_empirical)

        p1_choice = p1_choices[p1_index]
        p2_choice = p2_choices[p2_index]

        result = oak.update(battle, durations, p1_choice, p2_choice)

        result_type = oak.result_type(result)

    return result_type



while True:

    p1_team = random.choice(teams)
    p2_team = random.choice(teams)

    battle, durations, result = oak.parse_battle(f"{p1_team} | {p2_team}")

    p1_s1_cache = oak.search.SideCache()
    p1_s2_cache = oak.search.SideCache()
    p1_s1_cache.quantize(network)
    p1_s2_cache.quantize(network)
    for index in range(6):
        p1_s1_cache.precompute(network, battle.side(0), index)
        p1_s2_cache.precompute(network, battle.side(1), index)

    agent = oldoak.search.Agent()
    agent.eval = "/home/user/apple.old-battle.net"
    agent.bandit = bandit
    agent.budget = budget

    # print(result_type)
    result_types[play_game(p1_team, p2_team, p1_s1_cache, p1_s2_cache, agent)] += 1
    result_types[play_game(p2_team, p1_team, p1_s2_cache, p1_s1_cache, agent)] += 1
    print(result_types)