// Teams: Showdown's export format for Gen 1 (one Pokémon per paragraph,
// "=== [gen1ou] name ===" headers between teams), and truly random teams for
// the tests and a run with no pool.
//
// Gen 1 has no items, abilities or natures, and every set is assumed to max
// its DVs and stat experience (the user's, 2026-10-31), so a set is its
// species, level and moves; libpkmn's Init computes the stats. Lines that
// don't name these (EVs, IVs, an item, a nickname's details) are ignored.
//
// Names are matched after dropping everything but letters and digits,
// lowercased ("Mr. Mime" is MrMime, "Nidoran-F" NidoranF, "Thunder Punch"
// ThunderPunch).

#pragma once

#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <libpkmn/data.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/pkmn.h>

namespace ataraxos {

using Set = PKMN::Set;
// 1-6 sets; a pool team or a game's own.
using Team = std::vector<Set>;

inline constexpr size_t SPECIES_COUNT = 152; // None + 151
inline constexpr size_t MOVE_COUNT = 166;   // None + 165 (Struggle last)

namespace detail {

inline std::string key(std::string_view s) {
  std::string out;
  for (char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c))) out += char(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

inline std::string trim(std::string s) {
  const auto a = s.find_first_not_of(" \t\r");
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(" \t\r") - a + 1);
}

inline std::optional<PKMN::Data::Species> species(std::string_view name) {
  const std::string k = key(name);
  for (size_t s = 1; s < SPECIES_COUNT; s++) {
    if (key(PKMN::Data::SPECIES_CHAR_ARRAY[s].data()) == k) return PKMN::Data::Species(s);
  }
  return std::nullopt;
}

inline std::optional<PKMN::Data::Move> move(std::string_view name) {
  const std::string k = key(name);
  for (size_t m = 1; m < MOVE_COUNT; m++) {
    if (key(PKMN::Data::MOVE_CHAR_ARRAY[m].data()) == k) return PKMN::Data::Move(m);
  }
  return std::nullopt;
}

// A paragraph's first line: "Nickname (Species) (M) @ Item", "Species @
// Item", "Species".
inline std::optional<PKMN::Data::Species> species_line(std::string line) {
  if (const auto at = line.find(" @ "); at != std::string::npos) line = line.substr(0, at);
  line = trim(line);
  for (const char* g : {" (M)", " (F)"}) {
    if (line.size() >= 4 && line.ends_with(g)) line = trim(line.substr(0, line.size() - 4));
  }
  if (line.ends_with(")")) {
    const auto open = line.rfind('(');
    if (open != std::string::npos) {
      if (auto s = species(line.substr(open + 1, line.size() - open - 2))) return s;
    }
  }
  return species(line);
}

} // namespace detail

struct NamedTeam {
  std::string name;
  Team team;
};

// Every team in Showdown's export text; `errors` counts the teams with a
// line that names an unknown species or move, or with no Pokémon or more
// than 6, which are skipped.
inline std::vector<NamedTeam> parse_teams(const std::string& text, size_t& errors) {
  std::vector<NamedTeam> out;
  NamedTeam cur;
  bool bad = false, open = false;
  std::optional<Set> set;
  const auto end_set = [&] {
    if (set) cur.team.push_back(*set);
    set.reset();
  };
  const auto end_team = [&] {
    end_set();
    if (open || !cur.team.empty()) {
      if (bad || cur.team.empty() || cur.team.size() > 6) {
        errors++;
      } else {
        out.push_back(std::move(cur));
      }
    }
    cur = {};
    bad = false;
    open = false;
  };
  std::istringstream in(text);
  std::string raw;
  while (std::getline(in, raw)) {
    const std::string line = detail::trim(raw);
    if (line.starts_with("===")) {
      end_team();
      std::string name = detail::trim(line.substr(3, line.size() >= 6 ? line.size() - 6 : 0));
      if (name.starts_with("[")) {
        const auto close = name.find(']');
        if (close != std::string::npos) name = detail::trim(name.substr(close + 1));
      }
      cur.name = name;
      open = true;
      continue;
    }
    if (line.empty()) {
      end_set();
      continue;
    }
    if (!set) {
      const auto s = detail::species_line(line);
      if (!s) {
        bad = true;
        set = Set{};
        continue;
      }
      set = Set{};
      set->species = *s;
      set->moves = {};
      continue;
    }
    if (line.starts_with("- ")) {
      std::string m = detail::trim(line.substr(2));
      // "Hidden Power [Type]" never appears in Gen 1; take the first name.
      if (const auto slash = m.find(" / "); slash != std::string::npos) m = m.substr(0, slash);
      const auto id = detail::move(m);
      size_t k = 0;
      while (k < 4 && set->moves[k] != PKMN::Data::Move::None) k++;
      if (!id || k == 4) {
        bad = true;
      } else {
        set->moves[k] = *id;
      }
      continue;
    }
    if (line.starts_with("Level:")) {
      const int level = std::atoi(line.c_str() + 6);
      if (level < 1 || level > 100) {
        bad = true;
      } else {
        set->level = uint8_t(level);
      }
    }
  }
  end_team();
  return out;
}

// Every team in `files`, pooled; `names` gets each kept team's name.
inline std::vector<Team> load_teams(const std::vector<std::string>& files, size_t& errors,
                                    std::vector<std::string>* names = nullptr) {
  std::vector<Team> teams;
  errors = 0;
  for (const std::string& file : files) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("can't read " + file);
    std::stringstream ss;
    ss << in.rdbuf();
    for (NamedTeam& t : parse_teams(ss.str(), errors)) {
      teams.push_back(std::move(t.team));
      if (names) names->push_back(std::move(t.name));
    }
  }
  return teams;
}

// One truly random team: 1-6 Pokémon, any species (repeats allowed), 1-4
// distinct moves but Struggle, level 1-100.
inline Team random_team(std::mt19937_64& rng) {
  const auto pick = [&](uint32_t n) { return uint32_t(rng() % n); };
  Team team(1 + pick(6));
  for (Set& p : team) {
    p.species = PKMN::Data::Species(1 + pick(SPECIES_COUNT - 1));
    p.moves = {};
    const uint8_t n = uint8_t(1 + pick(4));
    for (uint8_t i = 0; i < n; i++) {
      PKMN::Data::Move m;
      bool repeat;
      do {
        m = PKMN::Data::Move(1 + pick(MOVE_COUNT - 2));
        repeat = false;
        for (uint8_t j = 0; j < i; j++) repeat |= p.moves[j] == m;
      } while (repeat);
      p.moves[i] = m;
    }
    p.level = uint8_t(1 + pick(100));
  }
  return team;
}

// A battle of two teams, libpkmn's seed `seed`.
inline pkmn_gen1_battle init_battle(uint64_t seed, const Team& t1, const Team& t2) {
  PKMN::Battle b{};
  b.sides[0] = PKMN::Init::init_side(t1);
  b.sides[1] = PKMN::Init::init_side(t2);
  b.rng = seed;
  return std::bit_cast<pkmn_gen1_battle>(b);
}

} // namespace ataraxos
