#pragma once

#include <libpkmn/pkmn.h>

namespace Client {

using Moves = std::array<PKMN::MoveSlot, 4>;
inline bool compare_moves(const Moves &_public, const Moves &truth,
                          std::string &reason) {
  for (const auto &ms : _public) {
    if (ms.id == PKMN::Data::Move::None) {
      continue;
    }
    const auto x =
        std::find_if(truth.begin(), truth.end(), [ms](const auto &x) {
          return (x.id == ms.id) && (x.pp == ms.pp);
        });
    if (x == truth.end()) {
      reason = PKMN::move_string(ms.id) + " not found in true moveset";
      return false;
    }
  }
  return true;
}

inline bool compare_volatiles(const PKMN::Volatiles &a,
                              const PKMN::Volatiles &b, std::string &reason) {
#define CMP(field)                                                             \
  do {                                                                         \
    if (a.field() != b.field()) {                                              \
      reason = #field " mismatch";                                             \
      return false;                                                            \
    }                                                                          \
  } while (0)

  // compare each field besides secret info: confusion_left, attacks,
  // disable_left if field mismatch, write the reason and return false,
  // otherwise return true

  // bitfields
  CMP(bide);
  CMP(thrashing);
  CMP(multi_hit);
  CMP(flinch);
  CMP(charging);
  CMP(binding);
  CMP(invulnerable);
  CMP(confusion);
  CMP(mist);
  CMP(focus_energy);
  CMP(substitute);
  CMP(recharging);
  CMP(rage);
  CMP(leech_seed);
  CMP(toxic);
  CMP(light_screen);
  CMP(reflect);
  CMP(transform);
  // secret/ignored fields (explicitly skipped):
  // confusion_left, attacks, disable_left
  // CMP(state); TODO wtf is this
  // CMP(substitute_hp); We don't know this
  CMP(transform_species);
  // CMP(disable_move); TODO
  CMP(toxic_counter);

#undef CMP

  reason = "equal";
  return true;
}

inline bool compare_boosts(const PKMN::Boosts &_public,
                           const PKMN::Boosts &truth, std::string &reason) {
  if (_public.atk() != truth.atk()) {
    reason = "boosts atk";
    return false;
  }
  if (_public.def() != truth.def()) {
    reason = "boosts def";
    return false;
  }
  if (_public.spc() != truth.spc()) {
    reason = "boosts spc";
    return false;
  }
  if (_public.spe() != truth.spe()) {
    reason = "boosts spe";
    return false;
  }
  if (_public.acc() != truth.acc()) {
    reason = "boosts acc";
    return false;
  }
  if (_public.eva() != truth.eva()) {
    reason = "boosts eva";
    return false;
  }
  return true;
}

inline bool compare_stats(const PKMN::Stats &_public_stats,
                          const PKMN::Stats &truth, std::string &reason) {
  if (_public_stats.hp != truth.hp) {
    reason = "stats hp";
    return false;
  }
  if (_public_stats.atk != truth.atk) {
    reason = "stats atk";
    return false;
  }
  if (_public_stats.def != truth.def) {
    reason = "stats def";
    return false;
  }
  if (_public_stats.spe != truth.spe) {
    reason = "stats spe";
    return false;
  }
  if (_public_stats.spc != truth.spc) {
    reason = "stats spc";
    return false;
  }
  return true;
}

inline bool compare_active(const PKMN::ActivePokemon &_public,
                           const PKMN::ActivePokemon &truth,
                           std::string &reason) {
  if (!compare_moves(_public.moves, truth.moves, reason)) {
    return false;
  }
  if (!compare_volatiles(_public.volatiles, truth.volatiles, reason)) {
    return false;
  }
  if (!compare_boosts(_public.boosts, truth.boosts, reason)) {
    return false;
  }
  if (_public.types != truth.types) {
    reason = "active types";
    return false;
  }
  if (_public.species != truth.species) {
    reason = "active species";
    return false;
  }
  if (!compare_stats(_public.stats, truth.stats, reason)) {
    return false;
  }
  return true;
}

inline bool compare_pokemon(const PKMN::Pokemon &_public,
                            const PKMN::Pokemon &truth, std::string &reason) {
  if (!compare_stats(_public.stats, truth.stats, reason)) {
    return false;
  }
  if (!compare_moves(_public.moves, truth.moves, reason)) {
    return false;
  }
  const auto normalize_status = [](PKMN::Status status) {
    if (PKMN::Data::is_sleep(status) && !PKMN::Data::self(status)) {
      return PKMN::Data::Status::Sleep1;
    } else {
      return status;
    }
  };
  if (normalize_status(_public.status) != normalize_status(truth.status)) {
    reason = "stored status";
    return false;
  }
  if (_public.species != truth.species) {
    reason = "stored species";
    return false;
  }
  if (_public.types != truth.types) {
    reason = "stored types";
    return false;
  }
  if (_public.level != truth.level) {
    reason = "stored level";
    return false;
  }
  // TODO different behaviour for p1/p2
  // if (_public.hp != truth.hp) {
  //   reason = "stored hp";
  // }
  return true;
}

using PokemonSleep = std::pair<PKMN::Pokemon, uint8_t>;
using Bench = std::array<PokemonSleep, 6>;
inline bool compare_pokemon_sleep(const PokemonSleep &_public,
                                  const PokemonSleep &truth,
                                  std::string &reason) {
  if (!compare_pokemon(_public.first, truth.first, reason)) {
    return false;
  }
  if (_public.second != truth.second) {
    reason = "mismatched sleep duration";
    return false;
  }
  return true;
}

Bench get_bench(const PKMN::Side &side, const PKMN::Duration &duration) {
  Bench bench{};
  for (auto slot = 1; slot <= 6; ++slot) {
    bench[slot - 1] = {side.get(slot), duration.sleep(slot - 1)};
  }
  return bench;
}

inline bool compare_bench(const Bench &_public, const Bench &truth,
                          std::string &reason) {

  if (!compare_pokemon_sleep(_public[0], truth[0], reason)) {
    return false;
  }
  for (auto slot = 2; slot <= 6; ++slot) {
    const auto &ps = _public[slot - 1];
    if (ps.first.species == PKMN::Data::Species::None) {
      continue;
    }
    const auto matching =
        std::find_if(truth.begin() + 1, truth.end(), [ps](const auto &x) {
          return x.first.species == ps.first.species;
        });
    if (matching == truth.end()) {
      reason =
          "Could not match _public's " + PKMN::species_string(ps.first.species);
      return false;
    }
    if (!compare_pokemon_sleep(ps, *matching, reason)) {
      return false;
    }
  }
  return true;
}

inline bool compare_side(const PKMN::Side &_public_side,
                         const PKMN::Duration &_public_duration,
                         const PKMN::Side &truth_side,
                         const PKMN::Duration &truth_duration,
                         std::string &reason) {
  if (!compare_active(_public_side.active, truth_side.active, reason)) {
    return false;
  }
  auto _public_bench = get_bench(_public_side, _public_duration);
  auto truth_bench = get_bench(truth_side, truth_duration);
  if (!compare_bench(_public_bench, truth_bench, reason)) {
    return false;
  }
  // TODO last_ stuff_public_bench
  return true;
}

inline bool compare_battles(const PKMN::Battle &_public_battle,
                            const PKMN::Durations &_public_durations,
                            const PKMN::Battle &truth_battle,
                            const PKMN::Durations &truth_durations,
                            std::string &reason) {
  if (!compare_side(_public_battle.sides[0], _public_durations.get(0),
                    truth_battle.sides[0], truth_durations.get(0), reason)) {
    return false;
  }
  if (!compare_side(_public_battle.sides[1], _public_durations.get(1),
                    truth_battle.sides[1], truth_durations.get(1), reason)) {
    return false;
  }
  return true;
}

} // namespace Client