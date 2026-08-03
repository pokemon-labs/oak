import oak
import random 
from collections import defaultdict
s = "snorlax bodyslam | starmie psychic"

def check_seeds():
    hp_dict = defaultdict(lambda : 0)
    for _ in range(10000):
        b, d, r = oak.parse_battle(s)
        b.rng = random.randint(0, 2**64 - 1)
        
        lax = b.side(0).stored()
        actions = oak.Actions()
        x = actions.get(1)
        y = actions.get(1)
        # x.critical_hit = 2
        # y.critical_hit = 0
        # x.psywave = 15
        # x.metronome = 15
        x.damage = 255
        y.damage = 255
        # x.hit = 0
        # x.critical_hit = 0

        r = oak.update(b, d, actions, 5, 5)

        # print(actions.get(0))
        percent = int(lax.hp / lax.stats().hp * 100)
        hp_dict[percent] += 1

    foo = [(key, value) for key, value in hp_dict.items()]
    foo.sort(key=lambda x : x[0])
    print(foo)

    print(foo[0][1]/foo[1][1])

check_seeds()