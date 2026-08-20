#pragma once

#include <encode/old-battle/battle.h>

/*

Over a given battle only 240 different embeddings are possible for each slot.
Species, Moves, Stats do not change, only Status and Moves, and the encoding
only checks if PP is available. So there are 15 Status and 16 Move possibilities

There is a cache what stores these embeddings. This code quickly determines
which one it is and returns a u8 key.

This way the NN only needs inference the active embeddings for each side. It can
just lookup the stored embeddings.
*/

namespace Encode {

namespace OldBattle {

constexpr uint8_t pp_byte(uint64_t v) noexcept {
  uint64_t t = (v >> 8) & 0x00FF00FF00FF00FFull;
  uint8_t r = 0;
  r |= (!!((t >> 0) & 0xFFu)) << 0;
  r |= (!!((t >> 16) & 0xFFu)) << 1;
  r |= (!!((t >> 32) & 0xFFu)) << 2;
  r |= (!!((t >> 48) & 0xFFu)) << 3;
  return r;
}

// the PP keys just have to be unique and fit inside 4 bits
consteval bool check_pp_byte() {
  bool success = true;

  std::array<int, 16> keys{};

  for (auto m = 0; m < 16; ++m) {
    const uint8_t a = m & 1;
    const uint8_t b = m & 2;
    const uint8_t c = m & 4;
    const uint8_t d = m & 8;
    std::array<uint8_t, 8> moves{0, a, 0, b, 0, c, 0, d};
    auto v = std::bit_cast<uint64_t>(moves);
    auto key = pp_byte(v);
    if (key >= 16) {
      return false;
    }
    keys[m] = key;
  }

  std::sort(keys.begin(), keys.end());

  for (auto i = 0; i < 15; ++i) {
    if (keys[i] == keys[i + 1]) {
      return false;
    }
  }

  return true;
}

static_assert(check_pp_byte());

inline uint8_t pokemon_key(const PKMN::Pokemon &pokemon, const uint8_t sleep) {
  uint8_t key = pp_byte(std::bit_cast<uint64_t>(pokemon.moves));
  if (static_cast<bool>(pokemon.status)) {
    key |= (Status::get_status_index(pokemon.status, sleep) + 1) << 4;
  }
  return key;
}

// TODO move set embeddings

inline constexpr uint8_t ceil_log2_u8(uint8_t x) {
  return (uint8_t)(31 - __builtin_clz((2 * x + 1)));
}

inline consteval bool is_bucket_step(const auto i) {
  return (ceil_log2_u8(i) + 1) == ceil_log2_u8(i + 1);
}

// This determines all values since its an increasing function
static_assert(ceil_log2_u8(0) == 0);
static_assert(is_bucket_step(0));
static_assert(is_bucket_step(1));
static_assert(is_bucket_step(3));
static_assert(is_bucket_step(7));
static_assert(is_bucket_step(15));
static_assert(is_bucket_step(31));
static_assert(ceil_log2_u8(61) == 6);

inline constexpr uint16_t pp_index(uint64_t bytes) {
  constexpr auto base = ceil_log2_u8(61) + 1;
  uint16_t index = 0;
  for (auto i = 0; i < 4; ++i) {
    index *= base;
    index += ceil_log2_u8(bytes & 255);
    bytes = bytes >> 16;
  }
  return index;
}

} // namespace Battle

} // namespace Encode