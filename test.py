import oak
from oak.search import *
import numpy
s = "starmie recover | snorlax bodyslam"
network = Network()
network.zero_initialize()
network.resize(128, 40, 128, 60, 128, 40, 64, 64, 64)
network.initialize(10)
cache = SideCache()

battle, durations, result = oak.parse_battle(s)

cache.precompute(network, battle.side(0), 0)

foo = cache.pokemon_embedding(0, 749, 40)
print(foo)