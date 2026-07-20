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

// ---------------------------------------------------------------------------
// Ratio-based boost scoring.
//
// Gen 1's stat-modifier glitch means active.stats can diverge from what the
// [-6, 6] boost stage implies (e.g. compounded paralysis speed cuts). Rather
// than trusting the boost stage, we treat the *actual* stat ratio
// (active.stats.X / stored().stats.X) as ground truth and interpolate a
// score from it. The table below uses Gen 1's real stage multipliers as
// anchor points, paired with the same score values the old lookup table
// used, so "normal" boosted stats score identically to before -- this only
// changes behavior when the ratio doesn't correspond to a legal stage.
// ---------------------------------------------------------------------------

struct RatioAnchor {
  float ratio;
  float score;
};

// Gen 1 stat-stage multipliers (numerator/denominator table from the games),
// paired with the score constants the original implementation assigned to
// each stage. Sorted ascending by ratio.
inline constexpr RatioAnchor BOOST_RATIO_TABLE[] = {
    {0.25f, -3.3f},  // -6
    {0.28f, -3.15f}, // -5
    {0.33f, -3.0f},  // -4
    {0.40f, -2.5f},  // -3
    {0.50f, -2.0f},  // -2
    {0.66f, -1.0f},  // -1
    {1.00f, 0.0f},   //  0
    {1.50f, 1.0f},   // +1
    {2.00f, 2.0f},   // +2
    {2.50f, 2.5f},   // +3
    {3.00f, 3.0f},   // +4
    {3.50f, 3.15f},  // +5
    {4.00f, 3.3f},   // +6
};

constexpr int BOOST_RATIO_TABLE_SIZE =
    sizeof(BOOST_RATIO_TABLE) / sizeof(RatioAnchor);

// Maps an arbitrary stat ratio (current/stored) to a score, via piecewise
// linear interpolation between anchors. Ratios outside the table (e.g. a
// stacked-glitch 1/16 speed cut) are extrapolated linearly using the slope
// of the nearest edge segment, rather than being clamped -- a 1/16 cut
// should score noticeably worse than a legal -6, not identically.
inline float score_from_ratio(float ratio) noexcept {
  if (ratio <= BOOST_RATIO_TABLE[0].ratio) {
    const auto &a = BOOST_RATIO_TABLE[0];
    const auto &b = BOOST_RATIO_TABLE[1];
    const float slope = (b.score - a.score) / (b.ratio - a.ratio);
    return a.score + slope * (ratio - a.ratio);
  }
  if (ratio >= BOOST_RATIO_TABLE[BOOST_RATIO_TABLE_SIZE - 1].ratio) {
    const auto &a = BOOST_RATIO_TABLE[BOOST_RATIO_TABLE_SIZE - 2];
    const auto &b = BOOST_RATIO_TABLE[BOOST_RATIO_TABLE_SIZE - 1];
    const float slope = (b.score - a.score) / (b.ratio - a.ratio);
    return b.score + slope * (ratio - b.ratio);
  }
  for (int i = 0; i < BOOST_RATIO_TABLE_SIZE - 1; ++i) {
    const auto &a = BOOST_RATIO_TABLE[i];
    const auto &b = BOOST_RATIO_TABLE[i + 1];
    if (ratio >= a.ratio && ratio <= b.ratio) {
      const float t = (ratio - a.ratio) / (b.ratio - a.ratio);
      return a.score + t * (b.score - a.score);
    }
  }
  assert(false);
  return 0.0f;
}

// TODO(api): confirm active.stats.{atk,def,spc,spe} exist and are the
// post-modifier stat values (i.e. already reflect boosts, paralysis, glitch
// stacking, etc.), and that stored().stats.{atk,def,spc,spe} are the clean
// unmodified base-derived stats. If the field names differ, update here.
inline float stat_ratio_score(uint16_t active_stat,
                              uint16_t stored_stat) noexcept {
  if (stored_stat == 0)
    return 0.0f; // defensive; shouldn't happen
  const float ratio =
      static_cast<float>(active_stat) / static_cast<float>(stored_stat);
  return score_from_ratio(ratio);
}

// ---------------------------------------------------------------------------
// Status conditions
// ---------------------------------------------------------------------------

// Paralysis is split into two pieces now:
//  - the speed-cut portion is captured for free by stat_ratio_score() reading
//    active.stats.spe directly, so it naturally scales with the mon's own
//    speed and captures glitch-stacked cuts.
//  - PARALYSIS_TURN_LOSS covers the ~25% chance of a fully wasted turn,
//    independent of how much speed was actually lost.
constexpr float PARALYSIS_TURN_LOSS = -10.0f;

constexpr float POKEMON_FROZEN =
    0.0f; // handled specially, see evaluate_pokemon
constexpr float FREEZE_CLAUSE_BONUS = 10.0f; // side-level, see evaluate_side

// A freshly-inflicted (n=0) genuine sleep has E[turns] = 3.5 (matches the
// well-known Gen 1 average). We scale so that baseline reproduces roughly
// the old flat value, then decay toward 0 as waking becomes likely.
constexpr float POKEMON_ASLEEP_PER_EXPECTED_TURN = -25.0f / 3.5f;

// Rest is voluntary, bounded, and already paired with a full heal (which the
// HP term rewards separately) -- so we score it far more mildly than
// opponent-inflicted sleep. Still scales down as it nears completion.
constexpr float POKEMON_RESTING_PER_REMAINING_TURN = -4.0f;

constexpr float POKEMON_TOXIC_BASE = -12.0f;       // close to POKEMON_POISONED;
constexpr float POKEMON_TOXIC_PER_COUNTER = -1.5f; // grows with counter, but
                                                   // switching resets it to
                                                   // regular poison anyway
constexpr float POKEMON_TOXIC_CAP = -20.0f; // never worse than this pre-switch
constexpr float POKEMON_POISONED = -10.0f;
constexpr float POKEMON_BURNED = -25.0f;

constexpr float LEECH_SEED = -30.0f;
constexpr float SUBSTITUTE = 40.0f;
constexpr float CONFUSION = -20.0f;

// Reflect/Light Screen in Gen 1 are volatiles with no field/side existence
// and no turn counter -- they double the internal Def/Spc stat used in
// damage calc until the mon switches out, making them much closer to a
// semi-permanent stat boost than a modern "screen". We scale their value by
// the mon's own bulk (stat * hp%) so they're worth far more on something
// like Snorlax than on a frail mon that happens to have one up.
constexpr float REFLECT_BULK_SCALE = 0.35f;
constexpr float LIGHT_SCREEN_BULK_SCALE = 0.35f;
constexpr float BULK_REFERENCE_STAT = 100.0f; // normalizes so an "average"
                                              // defensive stat produces
                                              // roughly the old flat value

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

inline bool is_resting(const PKMN::Pokemon &pokemon) noexcept {
  return Data::self(pokemon.status);
}

inline uint8_t rest_turns_remaining(const PKMN::Pokemon &pokemon) noexcept {
  return static_cast<uint8_t>(pokemon.status) & 7;
}

inline float evaluate_sleep(const PKMN::Pokemon &pokemon,
                            const auto sleep = 0) noexcept {
  if (is_resting(pokemon)) {
    const uint8_t remaining = rest_turns_remaining(pokemon);
    return POKEMON_RESTING_PER_REMAINING_TURN * static_cast<float>(remaining);
  }
  assert(sleep > 0);
  assert(sleep < 8);
  const uint8_t n = sleep - 1; // 0..6, since 7 would mean awake
  const float expected_remaining_turns = (7.0f - static_cast<float>(n)) / 2.0f;
  return POKEMON_ASLEEP_PER_EXPECTED_TURN * expected_remaining_turns;
}

// Note: freeze is intentionally NOT handled here anymore -- it's special
// cased in evaluate_pokemon, since a frozen mon should be scored as ~dead
// weight rather than status-penalized-but-alive.
inline float evaluate_status(const PKMN::Pokemon &pokemon,
                             const auto sleep) noexcept {
  switch (pokemon.status) {
  case Status::Burn: {
    return evaluate_burned(pokemon);
  }
  case Status::Paralysis: {
    return PARALYSIS_TURN_LOSS; // speed portion handled via stat ratio
  }
  case Status::Toxic: {
    return POKEMON_POISONED;
  }
  case Status::Poison: {
    return POKEMON_POISONED;
  }
  default: {
    if (is_sleep(pokemon.status)) {
      return evaluate_sleep(pokemon, sleep);
    } else {
      return 0;
    }
  }
  }
}

inline float evaluate_pokemon(const PKMN::Pokemon &pokemon,
                              const auto sleep) noexcept {
  float score = 0;
  if (pokemon.hp) {
    if (pokemon.status == Status::Freeze) {
      // Freeze Clause + no viable Fire moves in Gen 1 means this is
      // effectively a permanent KO. Score it like a fainted mon -- the
      // side-level Freeze Clause bonus (a slot on the opposing side is
      // permanently locked out) is added once in evaluate_side.
      return POKEMON_FROZEN;
    }
    score += (POKEMON_HP * pokemon.hp) / pokemon.stats.hp;
    score += evaluate_status(pokemon, sleep);
    score = std::max(score, 0.0f);
    score += POKEMON_ALIVE;
  }
  return score;
}

inline float evaluate_active(const PKMN::ActivePokemon &active,
                             const PKMN::Pokemon &stored,
                             const auto sleep) noexcept {
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
      // Scale by the mon's own Defense and HP% -- this is a semi-permanent
      // doubled-Defense volatile, not a temporary field screen, so it's
      // worth much more on a bulky wall than a flat constant implies.
      const float def_factor =
          static_cast<float>(stored.stats.def) / BULK_REFERENCE_STAT;
      const float hp_factor =
          static_cast<float>(stored.hp) / static_cast<float>(stored.stats.hp);
      score += REFLECT_BULK_SCALE * def_factor * hp_factor * POKEMON_HP;
    }
    if (vol.light_screen()) {
      const float spc_factor =
          static_cast<float>(stored.stats.spc) / BULK_REFERENCE_STAT;
      const float hp_factor =
          static_cast<float>(stored.hp) / static_cast<float>(stored.stats.hp);
      score += LIGHT_SCREEN_BULK_SCALE * spc_factor * hp_factor * POKEMON_HP;
    }

    score += POKEMON_ATTACK_BOOST *
             stat_ratio_score(active.stats.atk, stored.stats.atk);
    score += POKEMON_DEFENSE_BOOST *
             stat_ratio_score(active.stats.def, stored.stats.def);
    score += POKEMON_SPECIAL_ATTACK_BOOST *
             stat_ratio_score(active.stats.spc, stored.stats.spc);
    score += POKEMON_SPEED_BOOST *
             stat_ratio_score(active.stats.spe, stored.stats.spe);
  }
  return score;
}

inline float evaluate_side(const PKMN::Side &side,
                           const PKMN::Duration &duration) noexcept {
  float score = 0;
  score += evaluate_active(side.active, side.stored(), duration.sleep(0));

  bool side_has_frozen = std::any_of(
      side.pokemon.begin(), side.pokemon.end(), [](const auto &pokemon) {
        return pokemon.status == Data::Status::Freeze;
      });

  for (auto slot = 2; slot <= 6; ++slot) {
    const auto id = side.order[slot - 1];
    if (id != 0) {
      const auto &pkmn = side.pokemon[id - 1];
      score += evaluate_pokemon(pkmn, duration.sleep(slot - 1));
      if (pkmn.status == Status::Freeze) {
        side_has_frozen = true;
      }
    }
  }

  if (side_has_frozen) {
    score += FREEZE_CLAUSE_BONUS;
  }

  return score;
}

inline float evaluate_battle(const PKMN::Battle &battle,
                             const PKMN::Durations &durations) noexcept {
  float score = evaluate_side(battle.sides[0], durations.get(0)) -
                evaluate_side(battle.sides[1], durations.get(1));
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