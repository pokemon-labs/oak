#pragma once

#include <encode/battle/battle.h>

namespace Encode::Battle::Key {

constexpr auto n_hp = 50;
constexpr auto n_status = Status::n_dim;
constexpr auto n_pokemon = n_hp * n_status;

constexpr uint8_t hp48(uint16_t hp, uint16_t maxhp) {
  return static_cast<uint8_t>(1 + (uint32_t(hp - 1) * 49) / (maxhp - 1));
}

constexpr auto get_key(const PKMN::Pokemon &pokemon, uint8_t sleep) {
  uint16_t key = hp48(pokemon.hp, pokemon.stats.hp) * Status::n_dim +
                 Status::get_status_index(pokemon.status, sleep);
  return key;
}

using Types = uint8_t;
using Duration = uint16_t;
using ActiveKey = std::tuple<PKMN::Stats, Types, PKMN::Volatiles, Duration>;

constexpr auto get_key(const PKMN::ActivePokemon &active,
                       const PKMN::Duration &duration) {
  ActiveKey key{};
  std::get<0>(key) = active.stats;
  std::get<1>(key) = active.types;
  auto vol = active.volatiles;
  vol.set_confusion_left(0);
  vol.set_attacks(0);
  vol.set_state(0);
  vol.set_disable_left(0);
  std::get<2>(key) = vol;
  const auto *halves = reinterpret_cast<const uint16_t *>(&duration);
  auto d = halves[1];
  d >>= 2;
  std::get<3>(key) = d;
  return key;
}

constexpr auto n_moves = 7 * 7 * 7 * 7;

uint16_t get_key(const std::array<PKMN::MoveSlot, 4> &moves) {
  uint16_t key = 0;
  for (const auto [id, pp] : moves) {
    key *= 7;
    auto b = ceil_log2_u8(pp);
    key += b;
  }
  return key;
}

auto get_key_active(const std::array<PKMN::MoveSlot, 4> &moves) {
  auto key = moves;
  for (auto &ms : key) {
    ms.pp = ceil_log2_u8(ms.pp);
  }
  return key;
}

} // namespace Encode::Battle::Key