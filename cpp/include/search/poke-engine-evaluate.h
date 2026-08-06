#pragma once

#include <libpkmn/pkmn.h>
#include <util/random.h>

#include <cmath>

namespace PokeEngine {

constexpr float scaled_sigmoid(const float x) {
  return 1 / (1 + std::exp(-0.0125 * x));
}

using namespace PKMN;
struct Params {
  float POKEMON_ALIVE = 30;
  float POKEMON_HP = 100;

  float POKEMON_ATTACK_BOOST = 30;
  float POKEMON_DEFENSE_BOOST = 15.0f;
  float POKEMON_SPECIAL_ATTACK_BOOST = 30;
  float POKEMON_SPEED_BOOST = 30;

  float POKEMON_BOOST_MULTIPLIER_6 = 3.3f;
  float POKEMON_BOOST_MULTIPLIER_5 = 3.15f;
  float POKEMON_BOOST_MULTIPLIER_4 = 3.0f;
  float POKEMON_BOOST_MULTIPLIER_3 = 2.5f;
  float POKEMON_BOOST_MULTIPLIER_2 = 2.0f;
  float POKEMON_BOOST_MULTIPLIER_1 = 1.0f;
  float POKEMON_BOOST_MULTIPLIER_0 = 0;
  float POKEMON_BOOST_MULTIPLIER_NEG_1 = -1.0f;
  float POKEMON_BOOST_MULTIPLIER_NEG_2 = -2.0f;
  float POKEMON_BOOST_MULTIPLIER_NEG_3 = -2.5f;
  float POKEMON_BOOST_MULTIPLIER_NEG_4 = -3.0f;
  float POKEMON_BOOST_MULTIPLIER_NEG_5 = -3.15f;
  float POKEMON_BOOST_MULTIPLIER_NEG_6 = -3.3f;

  float POKEMON_FROZEN = -40;
  float POKEMON_ASLEEP = -25.0f;
  float POKEMON_PARALYZED = -25.0f;
  float POKEMON_TOXIC = -30;
  float POKEMON_POISONED = -10;
  float POKEMON_BURNED = -25.0f;

  float LEECH_SEED = -30;
  float SUBSTITUTE = 40;
  float CONFUSION = -20;

  float REFLECT = 20;
  float LIGHT_SCREEN = 20;
};

struct Eval : Params {
  float root_score;

  float get_boost_multiplier(int8_t boost) {
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

  float evaluate_burned(const PKMN::Pokemon &pokemon) const noexcept {
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

  float evaluate_status(const PKMN::Pokemon &pokemon) const noexcept {
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

  float evaluate_pokemon(const PKMN::Pokemon &pokemon) const noexcept {
    float score = 0;
    if (pokemon.hp) {
      score += (POKEMON_HP * pokemon.hp) / pokemon.stats.hp;
      score += evaluate_status(pokemon);
      score = std::max(score, 0.0f);
      score += POKEMON_ALIVE;
    }
    return score;
  }

  float evaluate_active(const PKMN::ActivePokemon &active,
                        const PKMN::Pokemon &stored) const noexcept {
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
      score +=
          POKEMON_SPECIAL_ATTACK_BOOST * get_boost_multiplier(boosts.spc());
      score += POKEMON_SPEED_BOOST * get_boost_multiplier(boosts.spe());
    }
    return score;
  }

  float evaluate_side(const PKMN::Side &side) const noexcept {
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

  float evaluate_battle(const PKMN::Battle &battle) const noexcept {
    float score =
        evaluate_side(battle.sides[0]) - evaluate_side(battle.sides[1]);
    return score;
  }

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
