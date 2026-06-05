#pragma once

#include <libpkmn/layout.h>
#include <libpkmn/pkmn.h>
#include <libpkmn/strings.h>
#include <util/strings.h>

#include <string>
#include <vector>

namespace Parse {

// convert vector of strings to a pokemon/durations initializer
auto parse_set(const auto &words) {

  PKMN::Init::Pokemon set{};

  set.species = PKMN::string_to_species(words[0]);

  auto n_moves = 0;
  // remaining words after only mandatory one: species
  for (auto i = 1; i < words.size(); ++i) {
    const auto &word = words[i];

    if (n_moves < 4) {
      const auto move_pp = split(word, ':');
      std::string move = move_pp[0];
      uint8_t pp = 0xFF;
      // try/catch is necessary since some sets don't have 4 moves
      // and omitting bad moves is a simple way to make search more effective
      PKMN::Data::Move m;
      bool move_parse_success = true;
      try {
        m = PKMN::string_to_move(move);
      } catch (...) {
        move_parse_success = false;
      }

      if (move_parse_success) {
        set.moves[n_moves] = m;
        if (move_pp.size() > 1) {
          pp = std::min(255UL, std::stoul(move_pp[1]));
        }
        set.pp[n_moves] = pp;
        ++n_moves;
        continue;
      }
    }

    using PKMN::Data::Status;
    auto lower = word;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](auto c) { return std::tolower(c); });

    if (lower.ends_with('%')) {
      const auto percent = std::stoul(lower.substr(0, lower.size() - 1));
      set.percent = percent;
    } else if (lower.ends_with("hp")) {
      const auto raw = std::stoul(lower.substr(0, lower.size() - 2));
      set.hp = raw;
    }

    if (lower == "par") {
      set.status = Status::Paralysis;
    } else if (lower == "frz") {
      set.status = Status::Freeze;
    } else if (lower == "psn") {
      set.status = Status::Poison;
    } else if (lower == "brn") {
      set.status = Status::Burn;
    } else if (lower.starts_with("slp")) {
      const auto sleeps_client = std::stoul(lower.substr(3));
      if (sleeps_client >= 7) {
        throw std::runtime_error(
            "parse_set(): Invalid turns slept (must be [0, 6]): " +
            std::to_string(sleeps_client));
      }
      set.status = Status::Sleep7;
      set.sleeps = sleeps_client + 1;
    } else if (lower.starts_with("rst")) {
      const auto hidden = std::stoul(lower.substr(3));
      if (hidden > 3 || hidden == 0) {
        throw std::runtime_error("parse_set(): Invalid sleep duration for "
                                 "rest (must be [1, 3]): " +
                                 std::to_string(hidden));
      }
      set.status = PKMN::Data::rest(hidden);
    }

    if (lower.starts_with("lvl")) {
      set.level = std::stoul(lower.substr(3));
    }
  }

  return set;
}

// TODO moves (presumably no way to parse active moves when they diverge from
// stored moves) struct alignas(1) ActivePokemon {
//   Stats stats;
//   PKMN::Data::Species species;
//   uint8_t types;
//   Boosts boosts;
//   Volatiles volatiles;
//   std::array<MoveSlot, 4> moves;
// };

inline auto parse_active(PKMN::Pokemon &pokemon, const auto &words)
    -> std::pair<PKMN::ActivePokemon, PKMN::Duration> {
  PKMN::Duration duration{};
  PKMN::ActivePokemon active = PKMN::switch_in(pokemon);

  PKMN::Stats stats{};
  auto &vol = active.volatiles;

  for (const auto &word : words) {

    using PKMN::Data::Status;
    auto lower = word;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](auto c) { return std::tolower(c); });

    // stats
    const auto read_stat = [](const auto &word) -> uint16_t {
      uint16_t x = std::stoul(word.substr(4));
      assert(x <= 999);
      return x;
    };
    const auto read_boost = [](const auto &word) -> int8_t {
      int8_t x = std::stoi(word.substr(3)); // +/- included
      assert(x < 7 && x > -7);
      return x;
    };

    if (lower.starts_with("atk=")) {
      stats.atk = read_stat(lower);
    } else if (lower.starts_with("atk+") || lower.starts_with("atk-")) {
      active.boosts.set_atk(read_boost(lower));
    }
    if (lower.starts_with("def=")) {
      stats.def = read_stat(lower);
    } else if (lower.starts_with("def+") || lower.starts_with("def-")) {
      active.boosts.set_def(read_boost(lower));
    }
    if (lower.starts_with("spe=")) {
      stats.spe = read_stat(lower);
    } else if (lower.starts_with("spe+") || lower.starts_with("spe-")) {
      active.boosts.set_spe(read_boost(lower));
    }
    if (lower.starts_with("spc=")) {
      stats.spc = read_stat(lower);
    } else if (lower.starts_with("spc+") || lower.starts_with("spc-")) {
      active.boosts.set_spc(read_boost(lower));
    }

    // volatliles

    if (lower == "(leech-seed)" || lower == "(leechseed)" ||
        lower == "(leech)") {
      vol.set_leech_seed(true);
    }
    if (lower == "(invuln)" || lower == "(invulnerable)" || lower == "(dig)" ||
        lower == "(fly)") {
      vol.set_invulnerable(true);
    }
    if (lower == "(lightscreen)" || lower == "(light-screen)" ||
        lower == "(ls)") {
      vol.set_light_screen(true);
    }
    if (lower == "(reflect)") {
      vol.set_reflect(true);
    }

    const auto parse_colon_split = [](auto word, const auto start) -> int {
      int x = -1;
      if (word.back() == ')') {
        word = word.substr(0, word.size() - 1);
      }
      if (word.starts_with(start)) {
        const auto s = split(word, ':');
        if (s.size() >= 2) {
          x = std::stoi(s[1]);
        }
      }
      return x;
    };

    // durations

    if (const auto conf = parse_colon_split(lower, "(conf"); conf >= 0) {
      if (conf == 0 || conf > 5) {
        throw std::runtime_error{
            "parse_active(): Confusion duration must be [1, 5] (1 means "
            "confused after moving, 5 means due to snap out.)"};
      }
      vol.set_confusion(true);
      duration.set_confusion(conf);
    }
    // if (const auto disable = parse_colon_split(lower, "(disable"); disable >=
    // 0) {
    //   vol.set_disable(true);
    //   duration.set_disable(disable);
    // }
    if (const auto thrashing = parse_colon_split(lower, "(thrash");
        thrashing >= 0) {
      vol.set_thrashing(true);
      duration.set_attacking(thrashing);
    }
    if (const auto thrashing = parse_colon_split(lower, "(petal");
        thrashing >= 0) {
      vol.set_thrashing(true);
      duration.set_attacking(thrashing);
    }
    if (const auto binding = parse_colon_split(lower, "(bind"); binding >= 0) {
    }
    if (const auto binding = parse_colon_split(lower, "(wrap"); binding >= 0) {
    }
  }

  // apply boosts first then overwrite with explicit stats to accomodate stat
  // modifcation glitch
  PKMN::Init::apply_boosts(active, active.boosts);
  if (stats.atk) {
    active.stats.atk = stats.atk;
  }
  if (stats.def) {
    active.stats.def = stats.def;
  }
  if (stats.spe) {
    active.stats.spe = stats.spe;
  }
  if (stats.spc) {
    active.stats.spc = stats.spc;
  }

  return {active, duration};
}

inline auto parse_side(const std::string &side_string)
    -> std::pair<PKMN::Side, PKMN::Duration> {
  const auto set_strings = split(side_string, ';');
  const auto n_sets = set_strings.size();
  if (n_sets == 0 || n_sets > 6) {
    throw std::runtime_error("parse_side(): " + std::to_string(n_sets) +
                             " set given. [1, 6] required.");
  }
  std::vector<PKMN::Init::Pokemon> sets;
  sets.resize(n_sets);
  std::transform(set_strings.begin(), set_strings.end(), sets.begin(),
                 [](const auto &string) {
                   const auto words = split(string, ' ');
                   return parse_set(words);
                 });
  auto side = PKMN::Init::init_side(sets);

  const auto active_words = split(set_strings[0], ' ');
  auto [active, duration] = parse_active(side.pokemon[0], active_words);
  side.active = active;
  PKMN::Init::init_sleeps(sets, duration);
  return {side, duration};
}

inline std::pair<pkmn_gen1_battle, pkmn_gen1_chance_durations>
parse_battle(const std::string &battle_string, uint64_t seed = 0x123456) {
  const auto side_strings = split(battle_string, '|');
  if (side_strings.size() != 2) {
    throw std::runtime_error(
        "parse_battle(): must have two sides, delineated by \'|\'");
  }
  const auto [p1, p1_dur] = parse_side(side_strings[0]);
  const auto [p2, p2_dur] = parse_side(side_strings[1]);
  auto battle = PKMN::Battle{};
  auto durations = PKMN::Durations{};
  battle.sides[0] = p1;
  battle.sides[1] = p2;
  battle.turn = 1;
  battle.rng = seed;
  durations.get(0) = p1_dur;
  durations.get(1) = p2_dur;
  return {std::bit_cast<pkmn_gen1_battle>(battle),
          std::bit_cast<pkmn_gen1_chance_durations>(durations)};
}

} // namespace Parse