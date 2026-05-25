#pragma once

#include <libpkmn/data.h>
#include <libpkmn/data/boosts.h>
#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/data/types.h>

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace PKMN {

namespace Init {

using PKMN::Data::Move;
using PKMN::Data::Species;
using PKMN::Data::Status;

struct Pokemon {
  Species species;
  std::array<Move, 4> moves;
  std::array<uint8_t, 4> pp = {64, 64, 64, 64};
  // actual hp value, ignored if negative
  int hp = -1;
  // percent
  uint32_t percent = 100;
  Status status = {};
  uint8_t sleeps = 0;
  uint8_t level = 100;
};

constexpr uint16_t compute_stat(uint8_t base, bool hp = false,
                                uint8_t level = 100, uint16_t ev = 255,
                                uint8_t dv = 15) {
  const uint32_t core = (2 * (base + dv)) + (ev / 4);
  const uint32_t factor = hp ? level + 10 : 5;
  return core * level / 100 + factor;
}
static_assert(compute_stat(100, false) == 298);
static_assert(compute_stat(250, true) == 703);
static_assert(compute_stat(5, false) == 108);

// The actual PKMN::Boost struct is awkward becuase it uses i4, so we define a
// helper struct
struct Boosts {
  int atk;
  int def;
  int spe;
  int spc;
};

constexpr uint16_t boost(uint16_t stat, int b) {
  const auto &pair = PKMN::Data::boosts[b + 6];
  return std::min(999, stat * pair[0] / pair[1]);
}

inline void apply_boosts(PKMN::ActivePokemon &active, const auto &boosts) {
  if constexpr (requires { boosts.atk; }) {
    active.stats.atk = boost(active.stats.atk, boosts.atk);
    active.boosts.set_atk(boosts.atk);
  } else if constexpr (requires { boosts.atk(); }) {
    active.stats.atk = boost(active.stats.atk, boosts.atk());
    active.boosts.set_atk(boosts.atk());
  }
  if constexpr (requires { boosts.def; }) {
    active.stats.def = boost(active.stats.def, boosts.def);
    active.boosts.set_def(boosts.def);
  } else if constexpr (requires { boosts.def(); }) {
    active.stats.def = boost(active.stats.def, boosts.def());
    active.boosts.set_def(boosts.def());
  }
  if constexpr (requires { boosts.spe; }) {
    active.stats.spe = boost(active.stats.spe, boosts.spe);
    active.boosts.set_spe(boosts.spe);
  } else if constexpr (requires { boosts.spe(); }) {
    active.stats.spe = boost(active.stats.spe, boosts.spe());
    active.boosts.set_spe(boosts.spe());
  }
  if constexpr (requires { boosts.spc; }) {
    active.stats.spc = boost(active.stats.spc, boosts.spc);
    active.boosts.set_spc(boosts.spc);
  } else if constexpr (requires { boosts.spc(); }) {
    active.stats.spc = boost(active.stats.spc, boosts.spc());
    active.boosts.set_spc(boosts.spc());
  }
}

constexpr PKMN::Pokemon init_pokemon(const auto &set) {
  PKMN::Pokemon pokemon{};
  // species
  pokemon.species = set.species;
  if (pokemon.species == Species::None) {
    return pokemon;
  }
  // level
  if constexpr (requires { set.level; }) {
    pokemon.level = set.level;
    assert(pokemon.level >= 1 && pokemon.level <= 100);
  } else {
    pokemon.level = 100;
  }
  // stats
  const auto base_stats = get_species_data(pokemon.species).base_stats;
  pokemon.stats.hp = compute_stat(base_stats.hp, true, pokemon.level);
  pokemon.stats.atk = compute_stat(base_stats.atk, false, pokemon.level);
  pokemon.stats.def = compute_stat(base_stats.def, false, pokemon.level);
  pokemon.stats.spe = compute_stat(base_stats.spe, false, pokemon.level);
  pokemon.stats.spc = compute_stat(base_stats.spc, false, pokemon.level);
  // moves
  for (auto m = 0; m < 4; ++m) {
    pokemon.moves[m].id = static_cast<Move>(set.moves[m]);
    if constexpr (requires { set.pp; }) {
      pokemon.moves[m].pp = std::min(set.pp[m], max_pp(pokemon.moves[m].id));
    } else {
      pokemon.moves[m].pp = max_pp(pokemon.moves[m].id);
    }
  }

  // hp
  uint16_t hp = pokemon.stats.hp;
  if constexpr (requires { set.percent; }) {
    hp = pokemon.stats.hp * set.percent / 100;
  }
  if constexpr (requires { set.hp; }) {
    if (set.hp >= 0) {
      hp = set.hp;
    }
  }
  pokemon.hp = hp;

  // status
  if constexpr (requires { set.status; }) {
    pokemon.status = static_cast<Status>(set.status);
  }
  // types
  const auto types = get_types(pokemon.species);
  pokemon.types =
      static_cast<uint8_t>(types[0]) | (static_cast<uint8_t>(types[1]) << 4);
  return pokemon;
}

constexpr PKMN::Side init_side(const auto &sets) {
  PKMN::Side side{};
  for (auto i = 0; i < sets.size(); ++i) {
    const auto pokemon = init_pokemon(sets[i]);
    if ((i == 0) || pokemon.hp) {
      side.order[i] = i + 1;
    }
    side.pokemon[i] = pokemon;
  }
  return side;
}

inline void init_sleeps(const auto &sets, PKMN::Duration &duration) {
  for (auto i = 0; i < sets.size(); ++i) {
    const auto &set = sets[i];
    if constexpr (requires { set.sleeps; }) {
      if (is_sleep(set.status) && !self(set.status)) {
        duration.set_sleep(i, set.sleeps);
      }
    }
  }
}

} // namespace Init

} // namespace PKMN