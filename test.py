import oak
import oak.search
import time
import numpy as np
import random

p1 = "jynx blizzard lovelykiss psychic rest; chansey icebeam sing softboiled thunderbolt; cloyster blizzard clamp explosion hyperbeam; rhydon bodyslam earthquake rockslide substitute; starmie blizzard recover thunderbolt thunderwave; tauros blizzard bodyslam earthquake hyperbeam"
p2 = "alakazam psychic recover seismictoss thunderwave; chansey reflect seismictoss softboiled thunderwave; exeggutor explosion psychic sleeppowder stunspore; lapras blizzard hyperbeam sing thunderbolt; snorlax bodyslam earthquake hyperbeam selfdestruct; tauros blizzard bodyslam earthquake hyperbeam"
battle, durations, result = oak.parse_battle(f"{p1} | {p2}")


def timed(f, *args, **kwargs):
    start = time.perf_counter()
    result = f(*args, **kwargs)
    print(f"{(time.perf_counter() - start) * 1000:.3f} ms")
    return result


def test_search():
    iterations = 2**10  
    network = oak.search.Network(2)
    network.resize(
        pokemon_hidden=2**8,
        pokemon_out=30,
        active_hidden=2**8,
        active_out=54,
        moves_hidden=2**8,
        moves_out=25,
        main_hidden=32,
        value_hidden=32,
        policy_hidden=32,
    )
    network.initialize(seed=random.randint(0, 2**64 - 1))  
    # network.read_parameters("/home/user/battle-2026-08-10-16:02:00/300.battle.net")
    ucb = oak.search.UCB(c=1.0)
    pucb = oak.search.PUCB(c=1.0)
    budget = oak.search.Iterations(iterations)
    p1_cache = oak.search.SideCache()
    p2_cache = oak.search.SideCache()
    for index in range(6):
        p1_cache.precompute(network, battle.side(0), index)
        p2_cache.precompute(network, battle.side(1), index)

    # network.quantize()
    # p1_cache.quantize(network)
    # p2_cache.quantize(network)

    output = timed(
        oak.search.run,
        battle,
        durations,
        budget,
        bandit=oak.search.PUCB(c=1.0),
        heap=oak.search.Node(),
        eval=network,
        p1_cache=p1_cache,
        p2_cache=p2_cache,
    )

    for side in [output.p1, output.p2]:
        print(side.empirical)
        print(side.nash)


# def code():

#     battle, durations, result = oak.parse_battle(s)

#     trajectory = oak.train.Trajectory(battle)
#     trajectory.empirical_matrix = True

#     while (not oak.result_type(result)):
#         output = oak.search.run(battle, ...)
#         i = np.sample(weights=output.p1_empirical)[0]
#         j = np.sample(weights=output.p2_empirical)[0]
#         c1 = output.p1_choices[i]
#         c2 = output.p2_choices[j]
#         trajectory.update(c1, c2, output)

#         result = oak.update(battle, durations, c1, c2)

#     b = trajectory.bytes()


def count(
    pokemon_hidden=2**9,
    pokemon_out=30,
    active_hidden=2**9,
    active_out=54,
    moves_hidden=2**9,
    moves_out=25,
    main_hidden=64,
    value_hidden=32,
    policy_hidden=64,
):
    return (
        pokemon_hidden * (pokemon_out + (151 + 50 + 15))
        + active_hidden * (50 + active_out)
        + moves_hidden * (6 * 160 + moves_out)
        + main_hidden
        * (main_hidden + (12 * (pokemon_out + moves_out) + 2 * active_out))
        + main_hidden * (value_hidden + 2 * policy_hidden)
        + value_hidden
        + 2 * policy_hidden * (151 + 160)
    )


test_search()
# print(count())    
