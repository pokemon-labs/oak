#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/init.h>
#include <libpkmn/strings.h>

namespace PKMN {

namespace Options {

#ifdef LOG
constexpr bool log = true;
#else
constexpr bool log = false;
#endif

#ifdef CHANCE
constexpr bool chance = true;
#else
constexpr bool chance = false;
#endif

#ifdef CALC
constexpr bool calc = true;
#else
constexpr bool calc = false;
#endif

}; // namespace Options

struct Set {
  Data::Species species;
  std::array<Data::Move, 4> moves;
  uint8_t level{100};

  constexpr bool operator==(const Set &other) const {
    auto a = moves;
    auto b = other.moves;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return (species == other.species) && (a == b) && (level == other.level);
  }
};

using Team = std::array<Set, 6>;

constexpr auto battle(const auto &p1, const auto &p2,
                      uint64_t seed = 0x123456) {
  PKMN::Battle battle{};
  battle.sides[0] = Init::init_side(p1);
  battle.sides[1] = Init::init_side(p2);
  battle.rng = seed;
  return std::bit_cast<pkmn_gen1_battle>(battle);
}

#ifdef LOG
inline pkmn_gen1_battle_options
options(const pkmn_gen1_log_options &log_options) {
  if (!log_options.buf) {
    throw std::runtime_error{
        "Trying to initialize options when the log has null buffer."};
  }
  pkmn_gen1_battle_options options{};
  pkmn_gen1_battle_options_set(&options, &log_options, nullptr, nullptr);
  return options;
}
#else
inline constexpr pkmn_gen1_battle_options options() { return {}; }
#endif

inline constexpr pkmn_gen1_chance_durations durations() { return {}; }

inline auto &durations(pkmn_gen1_battle_options &options) {
  return *pkmn_gen1_battle_options_chance_durations(&options);
}

inline const auto &durations(const pkmn_gen1_battle_options &options) {
  return *pkmn_gen1_battle_options_chance_durations(&options);
}

inline auto log_options(auto &buffer) {
  return pkmn_gen1_log_options{buffer.data(), buffer.size()};
}

inline void set(pkmn_gen1_battle_options &options) {
  return pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
}
inline void set(pkmn_gen1_battle_options &options,
                pkmn_gen1_log_options &log_options) {
  return pkmn_gen1_battle_options_set(&options, &log_options, nullptr, nullptr);
}
inline void set(pkmn_gen1_battle_options &options,
                pkmn_gen1_chance_options &chance_options) {
  return pkmn_gen1_battle_options_set(&options, nullptr, &chance_options,
                                      nullptr);
}
inline void set(pkmn_gen1_battle_options &options,
                pkmn_gen1_calc_options &calc_options) {
  return pkmn_gen1_battle_options_set(&options, nullptr, nullptr,
                                      &calc_options);
}

[[nodiscard]] inline pkmn_result update(auto &b, const auto c1, const auto c2,
                                        pkmn_gen1_battle_options &options) {
  const auto get_choice = [](const auto c, const uint8_t *side) -> pkmn_choice {
    using Choice = std::remove_cv<decltype(c)>::type;
    if constexpr (std::is_same_v<Choice, Data::Species>) {
      for (uint8_t i = 1; i < 6; ++i) {
        const auto id = side[Layout::Offsets::Side::order + i] - 1;
        if (static_cast<uint8_t>(c) ==
            side[24 * id + Layout::Offsets::Pokemon::species]) {
          return ((i + 1) << 2) | 2;
        }
      }
      throw std::runtime_error{"PKMN::update - invalid switch"};
    } else if constexpr (std::is_same_v<Choice, Data::Move>) {
      for (uint8_t i = 0; i < 4; ++i) {
        if (static_cast<uint8_t>(c) ==
            side[Layout::Offsets::Side::active +
                 Layout::Offsets::ActivePokemon::moves + 2 * i]) {
          return ((i + 1) << 2) | 1;
        }
      }
      throw std::runtime_error{"PKMN::update - invalid move"};
    } else if constexpr (std::is_integral_v<Choice>) {
      return c;
    } else {
      assert(false);
    }
  };
  auto &battle = *std::bit_cast<pkmn_gen1_battle *>(&b);
  pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  return pkmn_gen1_battle_update(
      &battle, get_choice(c1, battle.bytes),
      get_choice(c2, battle.bytes + Layout::Sizes::Side), &options);
}

inline auto choices(const pkmn_gen1_battle &battle, const pkmn_result result)
    -> std::pair<std::vector<pkmn_choice>, std::vector<pkmn_choice>> {
  std::vector<pkmn_choice> p1_choices;
  std::vector<pkmn_choice> p2_choices;
  p1_choices.resize(PKMN_GEN1_MAX_CHOICES);
  p2_choices.resize(PKMN_GEN1_MAX_CHOICES);
  const auto m =
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
  const auto n =
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
  p1_choices.resize(m);
  p2_choices.resize(n);
  return {p1_choices, p2_choices};
}

inline auto choice_labels(const pkmn_gen1_battle &battle,
                          const pkmn_result result)
    -> std::pair<std::vector<std::string>, std::vector<std::string>> {
  const auto [p1_choices, p2_choices] = choices(battle, result);
  std::vector<std::string> p1_labels{};
  std::vector<std::string> p2_labels{};
  for (auto i = 0; i < p1_choices.size(); ++i) {
    p1_labels.push_back(side_choice_string(battle.bytes, p1_choices[i]));
  }
  for (auto i = 0; i < p2_choices.size(); ++i) {
    p2_labels.push_back(
        side_choice_string(battle.bytes + Layout::Sizes::Side, p2_choices[i]));
  }
  return {p1_labels, p2_labels};
}

inline float score(const pkmn_result result) noexcept {
  switch (pkmn_result_type(result)) {
  case PKMN_RESULT_WIN: {
    return 1.0;
  }
  case PKMN_RESULT_LOSE: {
    return 0.0;
  }
  case PKMN_RESULT_TIE: {
    return 0.5;
  }
  default: {
    assert(false);
    return 0.5;
  }
  }
}

inline uint8_t score2(const pkmn_result result) noexcept {
  switch (pkmn_result_type(result)) {
  case PKMN_RESULT_WIN: {
    return 2;
  }
  case PKMN_RESULT_LOSE: {
    return 0;
  }
  case PKMN_RESULT_TIE: {
    return 1;
  }
  default: {
    assert(false);
    return 1;
  }
  }
}

inline pkmn_result_kind result_type(const pkmn_result result) noexcept {
  return pkmn_result_type(result);
}

enum class Result : std::underlying_type_t<std::byte> {
  None = 0,
  Win = 1,
  Lose = 2,
  Tie = 3,
  Error = 4,
};

enum class Choice : std::underlying_type_t<std::byte> {
  Pass = 0,
  Move = 1,
  Switch = 2,
};

enum class Player : std::underlying_type_t<std::byte> {
  P1 = 1,
  P2 = 2,
};

constexpr pkmn_result result(Result result = Result::None,
                             Choice p1 = Choice::Move,
                             Choice p2 = Choice::Move) {
  return static_cast<uint8_t>(result) | (static_cast<uint8_t>(p1) << 4) |
         (static_cast<uint8_t>(p2) << 6);
}

inline pkmn_result result(const pkmn_gen1_battle &b) {
  const auto &battle = PKMN::view(b);
  const auto &p1 = battle.sides[0];
  const auto &p2 = battle.sides[1];

  bool p1_alive = false;
  bool p2_alive = false;
  for (const auto &pokemon : p1.pokemon) {
    p1_alive |= (pokemon.hp);
  }
  for (const auto &pokemon : p2.pokemon) {
    p2_alive |= (pokemon.hp);
  }

  if (!p1_alive) {
    if (!p2_alive) {
      return result(Result::Tie, Choice::Pass, Choice::Pass);
    } else {
      return result(Result::Lose, Choice::Pass, Choice::Pass);
    }
  }
  if (!p2_alive) {
    return result(Result::Win, Choice::Pass, Choice::Pass);
  }

  if (!p1.stored().hp) {
    if (!p2.stored().hp) {
      return result(Result::None, Choice::Switch, Choice::Switch);
    } else {
      return result(Result::None, Choice::Switch, Choice::Pass);
    }
  }
  if (!p2.stored().hp) {
    return result(Result::None, Choice::Pass, Choice::Switch);
  }

  return result();
}

} // namespace PKMN
