# constexpr uint8_t hp48(uint16_t hp, uint16_t maxhp) {
#   return static_cast<uint8_t>(1 + (uint32_t(hp - 1) * 49) / (maxhp - 1));
# }

import math

def hp50(hp, max):
    return int(1 + (hp - 1) * 49 // (max - 1))

# max = 300
# for hp in range(0, max + 1):

#     print(f"{hp} {max} {hp50(hp, max)}")
def check(max):
    for b in range(1, 51):
        hp  = b * max // 50
        assert b == hp50(hp, max), f"{b} {hp} {hp50(hp, max)}"

check(714)