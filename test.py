import oak
import oak.search
import time
import numpy as np
import random


p1 = "jynx blizzard lovelykiss psychic rest; chansey icebeam sing softboiled thunderbolt; cloyster blizzard clamp explosion hyperbeam; rhydon bodyslam earthquake rockslide substitute; starmie blizzard recover thunderbolt thunderwave; tauros blizzard bodyslam earthquake hyperbeam"
p2 = "alakazam psychic recover seismictoss thunderwave; chansey reflect seismictoss softboiled thunderwave; exeggutor explosion psychic sleeppowder stunspore; lapras blizzard hyperbeam sing thunderbolt; snorlax bodyslam earthquake hyperbeam selfdestruct; tauros blizzard bodyslam earthquake hyperbeam"
battle, durations, result = oak.parse_battle(f"{p1} | {p2}")

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

    output = oak.search.search(battle, durations, budget, params=ucb, heap=node, eval=network)
    print(output.iterations)
    print(output.visit_matrix)

# def foo():
#     for i in range(1000):
#         print(i)
#         network = Network()
#         network.resize(
#             pokemon_hidden=128,
#             pokemon_out=40,
#             active_hidden=128,
#             active_out=60,
#             moves_hidden=128,
#             moves_out=40,
#             main_hidden=64,
#             value_hidden=64,
#             policy_hidden=64,
#         )
#         network.initialize(seed=random.randint(0, 2**64-1))

#         p1_cache = SideCache()
#         for index in range(6):
#             p1_cache.precompute(network, battle.side(0), index)

#         # t0 = time.perf_counter()
#         p1_embedding = network.forward_side(battle.side(0), durations.get(0))
#         # print(f"no cache: {(time.perf_counter() - t0) * 1000:.3f} ms")
#         # print(p1_embedding[:10], len(p1_embedding))
#         # t0 = time.perf_counter()
#         p1_embedding_cache = network.forward_side(battle.side(0), durations.get(0), p1_cache)
#         # print(f"with cache: {(time.perf_counter() - t0) * 1000:.3f} ms")
#         # print(p1_embedding_cache[:10], len(p1_embedding_cache))


#         diff = (p1_embedding == p1_embedding_cache)
#         print(diff.shape)
#         # idx = np.flatnonzero(~diff)[0]
#         # print(idx)
#         assert all(diff), f"{diff}"

bar()