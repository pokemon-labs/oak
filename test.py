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
    iterations = 2**20
    node = oak.search.Node()
    network = oak.search.Network()
    network.resize(
        pokemon_hidden=128,
        pokemon_out=40,
        active_hidden=128,
        active_out=60,
        moves_hidden=128,
        moves_out=40,
        main_hidden=64,
        value_hidden=64,
        policy_hidden=64,
    )
    network.initialize(seed=random.randint(0, 2**64-1))
    fp = oak.search.PokeEngine()
    ucb = oak.search.UCB(c=1.0)
    budget = oak.search.Iterations(iterations)
    p1_cache = oak.search.SideCache()
    p2_cache = oak.search.SideCache()
    for index in range(6):
        p1_cache.precompute(network, battle.side(0), index)
        p2_cache.precompute(network, battle.side(1), index)

    # output_fp = timed(
    #     oak.search.search,
    #     battle,
    #     durations,
    #     budget,
    #     params=ucb,
    #     heap=node,
    #     eval=fp,
    # )
    output_net = timed(
        oak.search.search,
        battle,
        durations,
        budget,
        params=ucb,
        heap=node,
        eval=network,
    )
    output_net = timed(
        oak.search.search,
        battle,
        durations,
        budget,
        params=ucb,
        heap=node,
        eval=network,
        p1_cache=p1_cache,
        p2_cache=p2_cache,
    )

def bar():
    node = oak.search.Node()
    network = oak.search.Network()
    network.resize(
        pokemon_hidden=128,
        pokemon_out=40,
        active_hidden=128,
        active_out=60,
        moves_hidden=128,
        moves_out=40,
        main_hidden=64,
        value_hidden=64,
        policy_hidden=64,
    )
    network.initialize(seed=random.randint(0, 2**64-1))
    ucb = oak.search.UCB(c=1.0)
    budget = oak.search.Iterations(2**10)
    fp = oak.search.PokeEngine()


    output = oak.search.search(battle, durations, budget, params=ucb, heap=node, eval=fp)
    print(output.iterations)
    print(output.visit_matrix)

test_search()