import oak
import oak.search
import oak.train    
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


# def test_search():
iterations = 2**20
network = oak.search.Network(2)
# network.resize(
#     pokemon_hidden=2**8,
#     pokemon_out=30,
#     active_hidden=2**8,
#     active_out=54,
#     moves_hidden=2**8,
#     moves_out=25,
#     main_hidden=32,
#     value_hidden=32,
#     policy_hidden=32,
# )
# network.initialize(seed=random.randint(0, 2**64 - 1))  
network.read_parameters("/home/user/rl-2026-08-17-16:38:18/nets/150.battle.net")

p1_cache = oak.search.SideCache()
p2_cache = oak.search.SideCache()
outputs = []
for index in range(6):
    p1_cache.precompute(network, battle.side(0), index)
    p2_cache.precompute(network, battle.side(1), index)

network.quantize()
p1_cache.quantize(network)
p2_cache.quantize(network)
# for b in [oak.search.UCB(1.0), oak.search.PUCB(1.0), oak.search.UCB1(1.0), oak.search.Exp3(1.0, 0.1), oak.search.PExp3(1.0, 0.1)][::-1]:
for b in [oak.search.PExp3(1.0, 0.1), oak.search.Exp3(1.0, 0.1)][::-1]:
    try:
        output = timed(
            oak.search.run,
            battle,
            durations,
            oak.search.Iterations(iterations),
            bandit=b,
            heap=oak.search.Node(),
            eval=network,
            p1_cache=p1_cache,
            p2_cache=p2_cache,
        )   
        outputs.append(output)
    except:
        print("FOO")
    for s, side in zip([output.p1, output.p2], [battle.side(0), battle.side(1)]):
        print("___")
        for i in range(s.k):
            print(oak.choice_label(side, s.choices[i]), s.empirical[i], s.nash[i])


# test_search()
