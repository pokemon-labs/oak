#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace PKMN {

namespace Data {
enum class Status : std::underlying_type_t<std::byte> {
  None = 0b00000000,
  Poison = 0b00001000,
  Burn = 0b00010000,
  Freeze = 0b00100000,
  Paralysis = 0b01000000,
  Toxic = 0b10001000,
  Sleep1 = 0b00000001,
  Sleep2 = 0b00000010,
  Sleep3 = 0b00000011,
  Sleep4 = 0b00000100,
  Sleep5 = 0b00000101,
  Sleep6 = 0b00000110,
  Sleep7 = 0b00000111,
  Rest1 = 0b10000001,
  Rest2 = 0b10000010,
  // Seemingly only possible mid update
  Rest3 = 0b10000011,
};

constexpr bool is_sleep(const auto status) {
  return static_cast<uint8_t>(status) & 7;
}

constexpr bool self(const auto status) {
  return static_cast<uint8_t>(status) & 0b10000000;
}

constexpr Status sleep(const auto n) {
  assert((n & 7) == n);
  return static_cast<Status>(n);
}
constexpr Status rest(const auto n) {
  assert((n & 3) == n);
  return static_cast<Status>(0b10000000 | n);
}
} // namespace Data

} // namespace PKMN