#pragma once

#include <libpkmn/pkmn.h>

namespace PKMN::Client {

using PKMN::Data::Effect;
using PKMN::Data::Move;
using PKMN::Data::Species;

#define COMPARE_PROPERTY(a, b, field, suffix)                                  \
  do {                                                                         \
    if (a.field() != b.field()) {                                              \
      reason += #field suffix " mismatch";                                     \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define COMPARE_FIELD(a, b, field, suffix)                                     \
  do {                                                                         \
    if (a.field != b.field) {                                                  \
      reason += #field suffix " mismatch";                                     \
      return false;                                                            \
    }                                                                          \
  } while (0)

using Moves = std::array<PKMN::MoveSlot, 4>;
using PokemonSleep = std::pair<PKMN::Pokemon, uint8_t>;
using Bench = std::array<PokemonSleep, 6>;

struct Options {
  int max_hp_diff;
};

inline bool compare_moves(const Moves &client, const Moves &truth,
                          std::string prefix, std::string &reason,
                          const Options &options) {
  for (const auto &ms : client) {
    if (ms.id == PKMN::Data::Move::None) {
      continue;
    }
    const auto x =
        std::find_if(truth.begin(), truth.end(), [ms](const auto &x) {
          // we can't determine binding move pp from client alone
          return (x.id == ms.id) &&
                 (PKMN::Data::move_data(ms.id).effect == Effect::Binding ||
                  x.pp == ms.pp);
        });
    if (x == truth.end()) {
      reason += prefix + PKMN::move_string(ms.id) + ": " +
                std::to_string(ms.pp) + " not found in true moveset ";
      return false;
    }
  }
  return true;
}

inline bool compare_volatiles(
                              const PKMN::Volatiles &a,
                              const PKMN::Volatiles &b, 
                              std::string &reason,
                              const Options &options) {
  COMPARE_PROPERTY(a, b, bide, "");
  COMPARE_PROPERTY(a, b, thrashing, "");
  COMPARE_PROPERTY(a, b, multi_hit, "");
  // flinch
  COMPARE_PROPERTY(a, b, charging, "");
  // we deal with binding separately since it can end privately
  // COMPARE_PROPERTY(a, b, binding, "");
  COMPARE_PROPERTY(a, b, invulnerable, "");
  COMPARE_PROPERTY(a, b, confusion, "");
  COMPARE_PROPERTY(a, b, mist, "");
  COMPARE_PROPERTY(a, b, focus_energy, "");
  COMPARE_PROPERTY(a, b, substitute, "");
  COMPARE_PROPERTY(a, b, recharging, "");
  COMPARE_PROPERTY(a, b, rage, "");
  COMPARE_PROPERTY(a, b, leech_seed, "");
  COMPARE_PROPERTY(a, b, toxic, "");
  COMPARE_PROPERTY(a, b, light_screen, "");
  COMPARE_PROPERTY(a, b, reflect, "");
  COMPARE_PROPERTY(a, b, transform, "");
  // confusion_left
  // attacks
  // state
  // sub_hp

  // if (a.transform_species == 0 && b.transform_species == 0) {
  //   // pass
  // } else {
  //   struct ID {
  //     int side;
  //     int slot;
  //     ID(auto ts) {
  //       auto x = static_cast<uint8_t>(ts);
  //       side = ((x & 0b1000) >> 3);
  //       slot = x & 0b111;
  //     }
  //     auto get_species(const PKMN::Battle& battle) {
  //       return battle.sides[side].get(slot).species;
  //     }
  //   };
  //   ID x{a.transform_species()};
  //   ID y{b.transform_species()};
  // }

  // disable_left
  COMPARE_PROPERTY(a, b, toxic_counter, "");
  return true;
}

inline bool compare_boosts(const PKMN::Boosts &client,
                           const PKMN::Boosts &truth, std::string &reason,
                           const Options &options) {
  COMPARE_PROPERTY(client, truth, atk, " boost");
  COMPARE_PROPERTY(client, truth, def, " boost");
  COMPARE_PROPERTY(client, truth, spc, " boost");
  COMPARE_PROPERTY(client, truth, spe, " boost");
  COMPARE_PROPERTY(client, truth, acc, " boost");
  COMPARE_PROPERTY(client, truth, eva, " boost");
  return true;
}

inline bool compare_stats(const PKMN::Stats &client_stats,
                          const PKMN::Stats &truth, std::string &reason,
                          const Options &options) {
  COMPARE_FIELD(client_stats, truth, hp, " stat");
  COMPARE_FIELD(client_stats, truth, atk, " stat");
  COMPARE_FIELD(client_stats, truth, def, " stat");
  COMPARE_FIELD(client_stats, truth, spe, " stat");
  COMPARE_FIELD(client_stats, truth, spc, " stat");
  return true;
}

inline bool compare_active(const PKMN::ActivePokemon &client,
                           const PKMN::ActivePokemon &truth,
                           std::string &reason, const Options &options) {
  if (!compare_volatiles(client.volatiles, truth.volatiles, reason, options)) {
    return false;
  }
  if (!compare_moves(client.moves, truth.moves, "active ", reason, options)) {
    return false;
  }
  if (!compare_stats(client.stats, truth.stats, reason, options)) {
    return false;
  }
  if (!compare_boosts(client.boosts, truth.boosts, reason, options)) {
    return false;
  }
  if (client.types != truth.types) {
    reason += "active types";
    return false;
  }
  if (client.species != truth.species) {
    reason += "active species";
    return false;
  }
  return true;
}

bool is_slept(PKMN::Status status) {
  return PKMN::Data::is_sleep(status) && !PKMN::Data::self(status);
}

auto normalize_status(PKMN::Status status) {
  if (is_slept(status)) {
    return PKMN::Data::Status::Sleep1;
  } else {
    return status;
  }
}

inline bool compare_pokemon(const PKMN::Pokemon &client,
                            const PKMN::Pokemon &truth, std::string &reason,
                            const Options &options) {
  if (!compare_stats(client.stats, truth.stats, reason, options)) {
    return false;
  }
  if (!compare_moves(client.moves, truth.moves, "stored ", reason, options)) {
    return false;
  }
  if (normalize_status(client.status) != normalize_status(truth.status)) {
    reason += "stored status";
    return false;
  }
  if (client.species != truth.species) {
    reason += "stored species";
    return false;
  }
  if (client.types != truth.types) {
    reason += "stored types";
    return false;
  }
  if (client.level != truth.level) {
    reason += "stored level";
    return false;
  }
  auto diff =
      client.hp > truth.hp ? client.hp - truth.hp : truth.hp - client.hp;
  if (diff > options.max_hp_diff) {
    reason += "stored hp " + std::to_string(client.hp) + " " +
              std::to_string(truth.hp) +
              " exceeds max_diff: " + std::to_string(options.max_hp_diff);
    return false;
  }
  return true;
}

inline bool compare_pokemon_sleep(const PokemonSleep &client,
                                  const PokemonSleep &truth,
                                  std::string &reason, const Options &options) {
  const bool client_fainted = (client.first.hp == 0);
  const bool truth_fainted = (truth.first.hp == 0);
  if (client_fainted != truth_fainted) {
    reason += "faint";
    return false;
  }
  if (!truth_fainted) {
    if (!compare_pokemon(client.first, truth.first, reason, options)) {
      return false;
    }
    if (is_slept(client.first.status) || is_slept(truth.first.status)) {
      if (client.second != truth.second) {
        reason += "mismatched sleep duration";
        return false;
      }
    }
  }
  return true;
}

Bench get_bench(const PKMN::Side &side, const PKMN::Duration &duration) {
  Bench bench{};
  for (auto slot = 1; slot <= 6; ++slot) {
    const auto i = side.order[slot - 1];
    if (i == 0) {
      bench[slot - 1] = {};
    } else {
      bench[slot - 1] = {side.get(slot), duration.sleep(slot - 1)};
    }
  }
  return bench;
}

inline bool compare_bench(const Bench &client, const Bench &truth,
                          std::string &reason, const Options &options) {

  if (!compare_pokemon_sleep(client[0], truth[0], reason, options)) {
    return false;
  }
  for (auto slot = 2; slot <= 6; ++slot) {
    const auto &ps = client[slot - 1];
    if (ps.first.species == PKMN::Data::Species::None) {
      continue;
    }
    const auto matching =
        std::find_if(truth.begin() + 1, truth.end(), [ps](const auto &x) {
          return x.first.species == ps.first.species;
        });
    if (matching == truth.end()) {
      reason += "Could not match client's " +
                PKMN::species_string(ps.first.species) + " at slot " +
                std::to_string(slot);
      return false;
    }
    if (!compare_pokemon_sleep(ps, *matching, reason, options)) {
      return false;
    }
  }
  return true;
}

inline bool compare_side(const PKMN::Side &client_side,
                         const PKMN::Duration &client_duration,
                         const PKMN::Side &truth_side,
                         const PKMN::Duration &truth_duration,
                         std::string &reason, const Options &options) {
  const auto client_fainted = client_side.stored().hp == 0;
  const auto truth_fainted = client_side.stored().hp == 0;
  if (client_fainted != truth_fainted) {
    reason += "stored faint mismatch";
    return false;
  }
  if (!truth_fainted) {
    if (!compare_active(client_side.active, truth_side.active, reason,
                        options)) {
      return false;
    }
    COMPARE_PROPERTY(client_duration, truth_duration, confusion, " duration");
    COMPARE_PROPERTY(client_duration, truth_duration, attacking, " duration");
    COMPARE_PROPERTY(client_duration, truth_duration, disable, " duration");
    // COMPARE_PROPERTY(client_duration, truth_duration, binding, " duration");

    if (truth_side.active.volatiles.binding() || truth_duration.binding()) {
      if (truth_side.active.volatiles.binding() !=
          client_side.active.volatiles.binding()) {
        reason += "volatiles binding";
        return false;
      }
      if (truth_duration.binding() != client_duration.binding()) {
        reason += "duration binding";
        return false;
      }
    }
  }
  const auto client_bench = get_bench(client_side, client_duration);
  const auto truth_bench = get_bench(truth_side, truth_duration);
  if (!compare_bench(client_bench, truth_bench, reason, options)) {
    return false;
  }
  // TODO
  if (client_side.last_used_move != truth_side.last_used_move) {
    reason += " last_used_move";
    return false;
  }
  // Data::Move last_used_move;
  return true;
}

inline bool compare_battles(const PKMN::Battle &client_battle,
                            const PKMN::Durations &client_durations,
                            const PKMN::Battle &truth_battle,
                            const PKMN::Durations &truth_durations,
                            std::string &reason) {
  auto p1 = Options{.max_hp_diff = 0};
  auto p2 = Options{.max_hp_diff = 15};
  for (auto i = 0; i < 2; ++i) {
    if (!compare_side(client_battle.sides[i], client_durations.get(i),
                      truth_battle.sides[i], truth_durations.get(i), reason,
                      i ? p2 : p1)) {
      return false;
    }
  }
  // TODO
  // std::array<MoveDetails, 2> last_moves;
  return true;
}

} // namespace PKMN::Client