import oak
from oak.search import *
s = "starmie recover | snorlax bodyslam"
network = Network()
# network.zero_initialize()
network.read_parameters("/home/user/pacnet")
cache = SideCache()

battle, durations, result = oak.parse_battle(s)

cache.precompute(network, battle.side(0), 0)

foo = cache.pokemon_embedding(1, 1, 59)
# print(foo)