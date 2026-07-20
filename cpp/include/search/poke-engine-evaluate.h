#pragma once

#include <libpkmn/pkmn.h>
#include <util/random.h>

#include <cmath>

#include <libpkmn/pkmn.h>
#include <util/random.h>

#include <algorithm>
#include <cmath>

namespace PokeEngine {

inline constexpr float scaled_sigmoid(const float x) {
  return 1 / (1 + std::exp(-0.0125 * x));
}

using namespace PKMN;

constexpr float POKEMON_ALIVE = 30;
constexpr float POKEMON_HP = 100;

constexpr float POKEMON_ATTACK_BOOST = 30;
constexpr float POKEMON_DEFENSE_BOOST = 15.0f;
constexpr float POKEMON_SPECIAL_ATTACK_BOOST = 30;
constexpr float POKEMON_SPEED_BOOST = 30;

constexpr float POKEMON_BOOST_MULTIPLIER_6 = 3.3f;
constexpr float POKEMON_BOOST_MULTIPLIER_5 = 3.15f;
constexpr float POKEMON_BOOST_MULTIPLIER_4 = 3.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_3 = 2.5f;
constexpr float POKEMON_BOOST_MULTIPLIER_2 = 2.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_1 = 1.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_0 = 0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_1 = -1.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_2 = -2.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_3 = -2.5f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_4 = -3.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_5 = -3.15f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_6 = -3.3f;

constexpr float POKEMON_FROZEN = -40;
constexpr float POKEMON_ASLEEP = -25.0f;
constexpr float POKEMON_PARALYZED = -25.0f;
constexpr float POKEMON_TOXIC = -30;
constexpr float POKEMON_POISONED = -10;
constexpr float POKEMON_BURNED = -25.0f;

constexpr float LEECH_SEED = -30;
constexpr float SUBSTITUTE = 40;
constexpr float CONFUSION = -20;

constexpr float REFLECT = 20;
constexpr float LIGHT_SCREEN = 20;

inline float get_boost_multiplier(int8_t boost) {
  switch (boost) {
  case 6:
    return POKEMON_BOOST_MULTIPLIER_6;
  case 5:
    return POKEMON_BOOST_MULTIPLIER_5;
  case 4:
    return POKEMON_BOOST_MULTIPLIER_4;
  case 3:
    return POKEMON_BOOST_MULTIPLIER_3;
  case 2:
    return POKEMON_BOOST_MULTIPLIER_2;
  case 1:
    return POKEMON_BOOST_MULTIPLIER_1;
  case 0:
    return POKEMON_BOOST_MULTIPLIER_0;
  case -1:
    return POKEMON_BOOST_MULTIPLIER_NEG_1;
  case -2:
    return POKEMON_BOOST_MULTIPLIER_NEG_2;
  case -3:
    return POKEMON_BOOST_MULTIPLIER_NEG_3;
  case -4:
    return POKEMON_BOOST_MULTIPLIER_NEG_4;
  case -5:
    return POKEMON_BOOST_MULTIPLIER_NEG_5;
  case -6:
    return POKEMON_BOOST_MULTIPLIER_NEG_6;
  default:
    assert(false);
    return 0;
  }
}

inline float evaluate_burned(const PKMN::Pokemon &pokemon) noexcept {
  float multiplier = 0;

  for (int m = 0; m < 4; ++m) {
    auto moveid = pokemon.moves[m].id;
    const auto &move = Data::move_data(moveid);
    if (move.bp > 0 && Data::is_physical(move.type)) {
      multiplier += 1.0f;
    }
  }

  if (pokemon.stats.spc > pokemon.stats.atk) {
    multiplier *= 0.5f;
  }

  return multiplier * POKEMON_BURNED;
}

inline float evaluate_status(const PKMN::Pokemon &pokemon) noexcept {
  switch (pokemon.status) {
  case Status::Burn: {
    return evaluate_burned(pokemon);
  }
  case Status::Freeze: {
    return POKEMON_FROZEN;
  }
  case Status::Paralysis: {
    return POKEMON_PARALYZED;
  }
  case Status::Toxic: {
    return POKEMON_TOXIC;
  }
  case Status::Poison: {
    return POKEMON_POISONED;
  }
  default: {
    if (is_sleep(pokemon.status)) {
      return POKEMON_ASLEEP;
    } else {
      return 0;
    }
  }
  }
}

inline float evaluate_pokemon(const PKMN::Pokemon &pokemon) noexcept {
  float score = 0;
  if (pokemon.hp) {
    score += (POKEMON_HP * pokemon.hp) / pokemon.stats.hp;
    score += evaluate_status(pokemon);
    score = std::max(score, 0.0f);
    score += POKEMON_ALIVE;
  }
  return score;
}

inline float evaluate_active(const PKMN::ActivePokemon &active,
                             const PKMN::Pokemon &stored) noexcept {
  float score = 0;
  if (stored.hp) {
    score += evaluate_pokemon(stored);
    const auto &vol = active.volatiles;
    if (vol.leech_seed()) {
      score += LEECH_SEED;
    }
    if (vol.substitute()) {
      score += SUBSTITUTE;
    }
    if (vol.confusion()) {
      score += CONFUSION;
    }
    if (vol.reflect()) {
      score += REFLECT;
    }
    if (vol.light_screen()) {
      score += LIGHT_SCREEN;
    }
    const auto &boosts = active.boosts;
    score += POKEMON_ATTACK_BOOST * get_boost_multiplier(boosts.atk());
    score += POKEMON_DEFENSE_BOOST * get_boost_multiplier(boosts.def());
    score += POKEMON_SPECIAL_ATTACK_BOOST * get_boost_multiplier(boosts.spc());
    score += POKEMON_SPEED_BOOST * get_boost_multiplier(boosts.spe());
  }
  return score;
}

inline float evaluate_side(const PKMN::Side &side) noexcept {
  float score = 0;
  score += evaluate_active(side.active, side.stored());
  for (auto slot = 2; slot <= 6; ++slot) {
    const auto id = side.order[slot - 1];
    if (id != 0) {
      score += evaluate_pokemon(side.pokemon[id - 1]);
    }
  }
  return score;
}

inline float evaluate_battle(const PKMN::Battle &battle) noexcept {
  float score = evaluate_side(battle.sides[0]) - evaluate_side(battle.sides[1]);
  return score;
}

struct Eval {
  float root_score;

  void get_root_score(const pkmn_gen1_battle &b) noexcept {
    const auto &battle = PKMN::view(b);
    root_score = evaluate_battle(battle);
  }

  float evaluate(const pkmn_gen1_battle &b) const noexcept {
    const auto &battle = PKMN::view(b);
    const float score = evaluate_battle(battle);
    const float value = scaled_sigmoid(score - root_score);
    return value;
  }
};

} // namespace PokeEngine

namespace PokeEngine2 {

inline constexpr float scaled_sigmoid(const float x) {
  return 1 / (1 + std::exp(-0.0125 * x));
}

using namespace PKMN;

constexpr float POKEMON_ALIVE = 30;
constexpr float POKEMON_HP = 100;

constexpr float POKEMON_ATTACK_BOOST = 30;
constexpr float POKEMON_DEFENSE_BOOST = 15.0f;
constexpr float POKEMON_SPECIAL_ATTACK_BOOST = 30;
constexpr float POKEMON_SPEED_BOOST = 30;

constexpr float POKEMON_BOOST_MULTIPLIER_6 = 3.3f;
constexpr float POKEMON_BOOST_MULTIPLIER_5 = 3.15f;
constexpr float POKEMON_BOOST_MULTIPLIER_4 = 3.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_3 = 2.5f;
constexpr float POKEMON_BOOST_MULTIPLIER_2 = 2.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_1 = 1.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_0 = 0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_1 = -1.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_2 = -2.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_3 = -2.5f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_4 = -3.0f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_5 = -3.15f;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_6 = -3.3f;

constexpr float POKEMON_FROZEN = 20.0f; // positive
// constexpr float POKEMON_ASLEEP = -25.0f;
constexpr float POKEMON_PARALYZED = -25.0f;
// constexpr float POKEMON_TOXIC = -30;
constexpr float POKEMON_POISONED = -10;
constexpr float POKEMON_BURNED = -25.0f;

constexpr float LEECH_SEED = -30;
constexpr float SUBSTITUTE = 40;
constexpr float CONFUSION = -20;

constexpr float REFLECT = 20;
constexpr float LIGHT_SCREEN = 20;

inline float get_boost_multiplier(int8_t boost) {
  switch (boost) {
  case 6:
    return POKEMON_BOOST_MULTIPLIER_6;
  case 5:
    return POKEMON_BOOST_MULTIPLIER_5;
  case 4:
    return POKEMON_BOOST_MULTIPLIER_4;
  case 3:
    return POKEMON_BOOST_MULTIPLIER_3;
  case 2:
    return POKEMON_BOOST_MULTIPLIER_2;
  case 1:
    return POKEMON_BOOST_MULTIPLIER_1;
  case 0:
    return POKEMON_BOOST_MULTIPLIER_0;
  case -1:
    return POKEMON_BOOST_MULTIPLIER_NEG_1;
  case -2:
    return POKEMON_BOOST_MULTIPLIER_NEG_2;
  case -3:
    return POKEMON_BOOST_MULTIPLIER_NEG_3;
  case -4:
    return POKEMON_BOOST_MULTIPLIER_NEG_4;
  case -5:
    return POKEMON_BOOST_MULTIPLIER_NEG_5;
  case -6:
    return POKEMON_BOOST_MULTIPLIER_NEG_6;
  default:
    assert(false);
    return 0;
  }
}

inline float evaluate_burned(const PKMN::Pokemon &pokemon) noexcept {
  float multiplier = 0;

  for (int m = 0; m < 4; ++m) {
    auto moveid = pokemon.moves[m].id;
    const auto &move = Data::move_data(moveid);
    if (move.bp > 0 && Data::is_physical(move.type)) {
      multiplier += 1.0f;
    }
  }

  if (pokemon.stats.spc > pokemon.stats.atk) {
    multiplier *= 0.5f;
  }

  return multiplier * POKEMON_BURNED;
}

inline float evaluate_status(const PKMN::Pokemon &pokemon,
                             uint8_t sleep) noexcept {
  switch (pokemon.status) {
  case Status::Burn: {
    return evaluate_burned(pokemon);
  }
  // case Status::Freeze: {
  //   return POKEMON_FROZEN;
  // }
  case Status::Paralysis: {
    return POKEMON_PARALYZED;
  }
  case Status::Toxic: {
    return POKEMON_POISONED;
  }
  case Status::Poison: {
    return POKEMON_POISONED;
  }
  default: {
    if (Data::is_sleep(pokemon.status)) {
      if (Data::self(pokemon.status)) {
        auto remaining = static_cast<uint8_t>(pokemon.status) & 7;
        return -7.0f * remaining;
      } else {
        auto remaining = 7 - sleep;
        assert(remaining >= 0);
        return -7.0f * remaining;
      }
    } else {
      return 0;
    }
  }
  }
}

inline float evaluate_pokemon(const PKMN::Pokemon &pokemon,
                              uint8_t sleep) noexcept {
  float score = 0;
  if (pokemon.hp) {
    if (pokemon.status == Data::Status::Freeze) {
      score += POKEMON_FROZEN; // positive cus sleep clause
    } else {
      score += (POKEMON_HP * pokemon.hp) / pokemon.stats.hp;
      score += evaluate_status(pokemon, sleep);
      score = std::max(score, 0.0f);
      score += POKEMON_ALIVE;
    }
  }
  return score;
}

inline float evaluate_active(const PKMN::ActivePokemon &active,
                             const PKMN::Pokemon &stored,
                             uint8_t sleep) noexcept {
  float score = 0;
  if (stored.hp) {
    score += evaluate_pokemon(stored, sleep);
    const auto &vol = active.volatiles;
    if (vol.leech_seed()) {
      score += LEECH_SEED;
    }
    if (vol.substitute()) {
      score += SUBSTITUTE;
    }
    if (vol.confusion()) {
      score += CONFUSION;
    }
    if (vol.reflect()) {
      score += REFLECT;
    }
    if (vol.light_screen()) {
      score += LIGHT_SCREEN;
    }
    const auto &boosts = active.boosts;
    score += POKEMON_ATTACK_BOOST * get_boost_multiplier(boosts.atk());
    score += POKEMON_DEFENSE_BOOST * get_boost_multiplier(boosts.def());
    score += POKEMON_SPECIAL_ATTACK_BOOST * get_boost_multiplier(boosts.spc());
    score += POKEMON_SPEED_BOOST * get_boost_multiplier(boosts.spe());
  }
  return score;
}

inline float evaluate_side(const PKMN::Side &side,
                           const PKMN::Duration &duration) noexcept {
  float score = 0;
  score += evaluate_active(side.active, side.stored(), duration.sleep(0));
  for (auto slot = 2; slot <= 6; ++slot) {
    const auto id = side.order[slot - 1];
    if (id != 0) {
      score += evaluate_pokemon(side.pokemon[id - 1], duration.sleep(slot - 1));
    }
  }
  return score;
}

inline float evaluate_battle(const PKMN::Battle &battle,
                             const PKMN::Durations &durations) noexcept {
  float score = evaluate_side(battle.sides[0], durations.get(0)) -
                evaluate_side(battle.sides[1], durations.get(0));
  return score;
}

struct Eval {
  float root_score;

  void get_root_score(const pkmn_gen1_battle &b,
                      const pkmn_gen1_chance_durations &d) noexcept {
    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    root_score = evaluate_battle(battle, durations);
  }

  float evaluate(const pkmn_gen1_battle &b,
                 const pkmn_gen1_chance_durations &d) const noexcept {
    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    const float score = evaluate_battle(battle, durations);
    const float value = scaled_sigmoid(score - root_score);
    return value;
  }
};

} // namespace PokeEngine2