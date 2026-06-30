#pragma once

#include <libpkmn/pkmn.h>

namespace PKMN::Client {

using PKMN::Data::Effect;
using PKMN::Data::Move;
using PKMN::Data::Species;

#define COMPARE_PROPERTY(key, a, b, property, suffix)                          \
  do {                                                                         \
    if (!static_cast<bool>(key.property())) {                                  \
      if (a.property() != b.property()) {                                      \
        reason += #property suffix " mismatch";                                \
        return false;                                                          \
      }                                                                        \
    } else {                                                                   \
    }                                                                          \
  } while (0)

#define COMPARE_FIELD(key, a, b, field, suffix)                                \
  do {                                                                         \
    if (!static_cast<bool>(key.field)) {                                       \
      if (a.field != b.field) {                                                \
        reason += #field suffix " mismatch";                                   \
        return false;                                                          \
      }                                                                        \
    } else {                                                                   \
    }                                                                          \
  } while (0)

using Moves = std::array<PKMN::MoveSlot, 4>;
using PokemonSleep = std::pair<PKMN::Pokemon, uint8_t>;
using Bench = std::array<PokemonSleep, 6>;

inline bool compare_moves(const Moves &client, const Moves &truth,
                          std::string prefix, const Moves &key,
                          std::string &reason) {
  for (auto i = 0; i < 4; ++i) {
    const auto &ms = client[i];
    const auto &k = key[i];
    if (ms.id == PKMN::Data::Move::None) {
      continue;
    }
    const auto x =
        std::find_if(truth.begin(), truth.end(), [ms, k](const auto &x) {
          // we can't determine binding move pp from client alone
          return (static_cast<bool>(k.id) || (x.id == ms.id)) &&
                 (k.pp ||
                  PKMN::Data::move_data(ms.id).effect == Effect::Binding ||
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

inline bool compare_volatiles(int player, const PKMN::Battle &client_battle,
                              const PKMN::Battle &truth_battle,
                              const PKMN::Volatiles &key, std::string &reason) {
  const auto &a = client_battle.sides[player].active.volatiles;
  const auto &b = truth_battle.sides[player].active.volatiles;
  COMPARE_PROPERTY(key, a, b, bide, "");
  COMPARE_PROPERTY(key, a, b, thrashing, "");
  COMPARE_PROPERTY(key, a, b, multi_hit, "");
  // flinch
  COMPARE_PROPERTY(key, a, b, charging, "");
  // we disable it with the key since it can end privately
  COMPARE_PROPERTY(key, a, b, binding, "");
  COMPARE_PROPERTY(key, a, b, invulnerable, "");
  COMPARE_PROPERTY(key, a, b, confusion, "");
  COMPARE_PROPERTY(key, a, b, mist, "");
  COMPARE_PROPERTY(key, a, b, focus_energy, "");
  COMPARE_PROPERTY(key, a, b, substitute, "");
  COMPARE_PROPERTY(key, a, b, recharging, "");
  COMPARE_PROPERTY(key, a, b, rage, "");
  COMPARE_PROPERTY(key, a, b, leech_seed, "");
  COMPARE_PROPERTY(key, a, b, toxic, "");
  COMPARE_PROPERTY(key, a, b, light_screen, "");
  COMPARE_PROPERTY(key, a, b, reflect, "");
  COMPARE_PROPERTY(key, a, b, transform, "");
  // confusion_left
  // attacks
  if (!key.bide() && a.bide()) {
    // b then also has bide set
    const auto state_diff =
        a.state() > b.state() ? a.state() - b.state() : b.state() - a.state();
    if (state_diff > key.state()) {
      reason += "state " + std::to_string(a.state()) + " " +
                std::to_string(b.state());
      return false;
    }
  }
  if (a.substitute_hp() < b.substitute_hp()) {
    reason += "substitute hp smaller";
    return false;
  }
  const auto sub_hp_diff = a.substitute_hp() - b.substitute_hp();
  if (sub_hp_diff > key.substitute_hp()) {
    reason += "substitute hp " + std::to_string(a.substitute_hp()) + " " +
              std::to_string(b.substitute_hp()) +
              " exceeds max_diff: " + std::to_string(key.substitute_hp());
    return false;
  }

  if (key.transform() ||
      (a.transform_species() == 0 && b.transform_species() == 0)) {
    // pass
  } else {
    struct ID {
      int side;
      int slot;
      ID(uint8_t ts) {
        auto x = static_cast<uint8_t>(ts);
        side = ((x & 0b1000) >> 3);
        slot = x & 0b111;
      }
      auto get_species(const PKMN::Battle &battle) {
        return battle.sides[side].pokemon[slot - 1].species;
      }
    };
    auto client_species = ID{a.transform_species()}.get_species(client_battle);
    auto truth_species = ID{b.transform_species()}.get_species(truth_battle);
    if (client_species != truth_species) {
      reason +=
          "tranform species mismatch: " + PKMN::species_string(client_species) +
          " " + PKMN::species_string(truth_species);
      return false;
    }
  }
  // disable_left
  COMPARE_PROPERTY(key, a, b, toxic_counter, "");
  return true;
}

inline bool compare_boosts(const PKMN::Boosts &client,
                           const PKMN::Boosts &truth, const PKMN::Boosts &key,
                           std::string &reason) {
  COMPARE_PROPERTY(key, client, truth, atk, " boost");
  COMPARE_PROPERTY(key, client, truth, def, " boost");
  COMPARE_PROPERTY(key, client, truth, spc, " boost");
  COMPARE_PROPERTY(key, client, truth, spe, " boost");
  COMPARE_PROPERTY(key, client, truth, acc, " boost");
  COMPARE_PROPERTY(key, client, truth, eva, " boost");
  return true;
}

inline bool compare_stats(const PKMN::Stats &client_stats,
                          const PKMN::Stats &truth, const PKMN::Stats &key,
                          std::string &reason) {
  COMPARE_FIELD(key, client_stats, truth, hp, " stat");
  COMPARE_FIELD(key, client_stats, truth, atk, " stat");
  COMPARE_FIELD(key, client_stats, truth, def, " stat");
  COMPARE_FIELD(key, client_stats, truth, spe, " stat");
  COMPARE_FIELD(key, client_stats, truth, spc, " stat");
  return true;
}

inline bool compare_active(int player, const PKMN::Battle &client_battle,
                           const PKMN::Battle &truth_battle,
                           const PKMN::ActivePokemon &key,
                           std::string &reason) {
  const auto &client = client_battle.sides[player].active;
  const auto &truth = truth_battle.sides[player].active;
  if (!compare_volatiles(player, client_battle, truth_battle, key.volatiles,
                         reason)) {
    return false;
  }
  if (!compare_moves(client.moves, truth.moves, "active ", key.moves, reason)) {
    return false;
  }
  if (!compare_stats(client.stats, truth.stats, key.stats, reason)) {
    return false;
  }
  if (!compare_boosts(client.boosts, truth.boosts, key.boosts, reason)) {
    return false;
  }
  if (!key.types && client.types != truth.types) {
    reason += "active types";
    return false;
  }
  if (!static_cast<bool>(key.species) && client.species != truth.species) {
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
                            const PKMN::Pokemon &truth,
                            const PKMN::Pokemon &key, std::string &reason) {
  if (!compare_stats(client.stats, truth.stats, key.stats, reason)) {
    return false;
  }
  if (!compare_moves(client.moves, truth.moves, "stored ", key.moves, reason)) {
    return false;
  }
  if (!static_cast<bool>(key.status) &&
      normalize_status(client.status) != normalize_status(truth.status)) {
    reason += "stored status";
    return false;
  }
  if (!static_cast<bool>(key.species) && client.species != truth.species) {
    reason += "stored species";
    return false;
  }
  if (!key.types && client.types != truth.types) {
    reason += "stored types";
    return false;
  }
  if (!key.level && client.level != truth.level) {
    reason += "stored level";
    return false;
  }
  auto diff =
      client.hp > truth.hp ? client.hp - truth.hp : truth.hp - client.hp;
  if (diff > key.hp) {
    reason += "stored hp " + std::to_string(client.hp) + " " +
              std::to_string(truth.hp) +
              " exceeds max_diff: " + std::to_string(key.hp);
    return false;
  }
  return true;
}

inline bool compare_pokemon_sleep(const PokemonSleep &client,
                                  const PokemonSleep &truth,
                                  const PKMN::Pokemon &key,
                                  std::string &reason) {
  const bool client_fainted = (client.first.hp == 0);
  const bool truth_fainted = (truth.first.hp == 0);
  if (!key.hp && client_fainted != truth_fainted) {
    reason += "faint";
    return false;
  }
  if (!truth_fainted) {
    if (!compare_pokemon(client.first, truth.first, key, reason)) {
      return false;
    }
    if (!static_cast<bool>(key.status) &&
        (is_slept(client.first.status) || is_slept(truth.first.status))) {
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
                          const PKMN::Side &key, std::string &reason) {

  if (!compare_pokemon_sleep(client[0], truth[0], key.pokemon[0], reason)) {
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
      reason += "species : Could not match client's " +
                PKMN::species_string(ps.first.species) + " at slot " +
                std::to_string(slot);
      return false;
    }
    if (!compare_pokemon_sleep(ps, *matching, key.pokemon[0], reason)) {
      return false;
    }
  }
  return true;
}

inline bool compare_side(int player, const PKMN::Battle &client_battle,
                         const PKMN::Duration &client_duration,
                         const PKMN::Battle &truth_battle,
                         const PKMN::Duration &truth_duration,
                         const PKMN::Side &key,
                         const PKMN::Duration &duration_key,
                         std::string &reason) {

  const auto &client_side = client_battle.sides[player];
  const auto &truth_side = truth_battle.sides[player];
  const auto client_fainted = client_side.stored().hp == 0;
  const auto truth_fainted = truth_side.stored().hp == 0;
  if (client_fainted != truth_fainted) {
    reason += "stored faint mismatch";
    return false;
  }
  if (!truth_fainted) {
    if (!compare_active(player, client_battle, truth_battle, key.active,
                        reason)) {
      return false;
    }
    COMPARE_PROPERTY(duration_key, client_duration, truth_duration, confusion,
                     " duration");
    COMPARE_PROPERTY(duration_key, client_duration, truth_duration, attacking,
                     " duration");
    COMPARE_PROPERTY(duration_key, client_duration, truth_duration, disable,
                     " duration");
    COMPARE_PROPERTY(duration_key, client_duration, truth_duration, binding,
                     " duration");
  }
  const auto client_bench = get_bench(client_side, client_duration);
  const auto truth_bench = get_bench(truth_side, truth_duration);
  if (!compare_bench(client_bench, truth_bench, key, reason)) {
    return false;
  }

  COMPARE_FIELD(key, client_side, truth_side, last_used_move, "");
  COMPARE_FIELD(key, client_side, truth_side, last_selected_move, "");

  // Data::Move last_used_move;
  return true;
}

inline bool compare_battles(const PKMN::Battle &client_battle,
                            const PKMN::Durations &client_durations,
                            const PKMN::Battle &truth_battle,
                            const PKMN::Durations &truth_durations,
                            const PKMN::Battle &key,
                            const PKMN::Durations &durations_key,
                            std::string &reason) {
  for (auto i = 0; i < 2; ++i) {
    const auto &client_side = client_battle.sides[i];
    const auto &truth_side = truth_battle.sides[i];

    if (!compare_side(i, client_battle, client_durations.get(i), truth_battle,
                      truth_durations.get(i), key.sides[i],
                      durations_key.get(i), reason)) {
      return false;
    }
    COMPARE_FIELD(key, client_battle, truth_battle, last_moves[i].counterable,
                  "");
    // last_move.index
    const bool should_compare_index =
        !static_cast<bool>(key.last_moves[i].index) &&
        !static_cast<bool>(
            key.sides[i].last_used_move) && // they both match already from
                                                // compare side
        static_cast<bool>(
            client_battle.sides[i]
                .last_used_move); // is this is None then last_move.index is
                                      // *probably* 1
    if (should_compare_index) {
      const auto client_id =
          client_side.active.moves[client_battle.last_moves[i].index - 1].id;
      const auto truth_id =
          truth_side.active.moves[truth_battle.last_moves[i].index - 1].id;
      if (client_id != truth_id) {
        reason += "last_moves.index " + PKMN::move_string(client_id) + " " +
                  PKMN::move_string(truth_id);
        return false;
      }
    }
  }
  // last damage
  bool client_null_damage = (client_battle.last_damage == 0);
  bool truth_null_damage = (truth_battle.last_damage == 0);
  if (client_null_damage != truth_null_damage) {
    reason += "last_damage null";
    return false;
  }
  if (!client_null_damage) {
    auto c_dmg = client_battle.last_damage;
    auto t_dmg = truth_battle.last_damage;
    auto diff = (c_dmg > t_dmg ? c_dmg - t_dmg : t_dmg - c_dmg);
    auto damage_ratio = (float)c_dmg / (t_dmg);
    auto exceeds = (c_dmg > t_dmg);
    // seemingly, normal damage variation seems really high with low damage
    // numbers do to loss of precision
    if (((c_dmg > 15) && (damage_ratio < (.8))) ||
        (exceeds && (diff > key.last_damage))) {
      reason +=
          "last_damage " + std::to_string(c_dmg) + " " + std::to_string(t_dmg);
      return false;
    }
  }

  return true;
}

} // namespace PKMN::Client