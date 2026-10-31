// Checks the text parser (src/text_parser.hpp) on Showdown's own text: the
// dump tests/showdown/play.cjs writes from Showdown's simulator (Gen 1 OU,
// our teams, uniform choices). Each update's three streams (the raw log with
// its |split| triplets, and each player's stream) must parse with no
// ParseError, and the HP, max HP and status the parsed bytes last showed
// for each Pokémon must be the simulator's: exact in the raw log and for a
// player's own side, the other side as Showdown's percent/100 (view.hpp).
// A status is compared only while the Pokémon stands ("0 fnt" keeps the
// last one shown).
//
// It also counts the lines the parser read past, by tag, so what real text
// carries that libpkmn doesn't is on show.
//
//   node tests/showdown/play.cjs GAMES SEED TEAM_FILE > dump.txt
//   check_showdown dump.txt

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "format.hpp"
#include "text_parser.hpp"

using namespace ataraxos;

namespace {

struct Truth {
  uint8_t side;
  std::string nick;
  uint16_t hp, max;
  std::string status;
};

std::vector<std::string> split_tabs(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    const size_t t = s.find('\t', start);
    out.push_back(s.substr(start, t == std::string::npos ? t : t - start));
    if (t == std::string::npos) return out;
    start = t + 1;
  }
}

using Shown = std::array<std::map<std::string, std::tuple<uint16_t, uint16_t, uint8_t>>, 2>;

// What one parsed update shows: the last HP, max and status per Pokémon.
void track(const std::vector<uint8_t>& bytes, const Names& names, Shown& shown) {
  size_t i = 0;
  while (i < bytes.size() && bytes[i] != 0) {
    const ArgType t = ArgType(bytes[i]);
    const uint8_t* d = &bytes[i + 1];
    i += 1 + message_size(t, d);
    const auto nick = [&](uint8_t id) -> const std::string& {
      return names.nickname[ident_player(id)][ident_slot(id) - 1];
    };
    const auto health = [&](size_t at) {
      shown[ident_player(d[0])][nick(d[0])] = {uint16_t(d[at] | d[at + 1] << 8),
                                               uint16_t(d[at + 2] | d[at + 3] << 8), d[at + 4]};
    };
    switch (t) {
    case ArgType::Switch: health(3); break;
    case ArgType::Damage:
    case ArgType::Heal: health(1); break;
    case ArgType::Status: std::get<2>(shown[ident_player(d[0])][nick(d[0])]) = d[1]; break;
    case ArgType::CureStatus: std::get<2>(shown[ident_player(d[0])][nick(d[0])]) = 0; break;
    default: break;
    }
  }
}

struct Battle {
  std::array<Names, 3> names;  // the raw log's, p1's, p2's
  std::array<std::unique_ptr<TextParser>, 3> parsers;
  std::array<Shown, 3> shown;
};

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check_showdown DUMP\n");
    return 2;
  }
  std::ifstream in(argv[1]);
  std::map<std::string, size_t, std::less<>> skipped;
  std::map<std::string, size_t> mismatches;
  size_t battles = 0, updates = 0, errors = 0, compared = 0, printed = 0;
  std::unique_ptr<Battle> b;
  std::array<std::vector<std::string>, 3> text;
  std::vector<Truth> truth;
  bool pending = false;
  bool broken = false; // a ParseError: the rest of the battle isn't read

  const auto finish = [&] {
    if (!pending || !b || broken) {
      pending = false;
      truth.clear();
      return;
    }
    pending = false;
    updates++;
    for (int c = 0; c < 3; c++) {
      try {
        const std::vector<uint8_t> bytes = b->parsers[c]->parse(text[c]);
        track(bytes, b->names[c], b->shown[c]);
      } catch (const ParseError& e) {
        broken = true;
        if (errors++ < 10) {
          std::printf("== stream %d, battle %zu: %s\n", c, battles, e.what());
          for (const std::string& l : text[c]) std::printf("  %s\n", l.c_str());
        }
        truth.clear();
        return;
      }
    }
    for (const Truth& t : truth) {
      for (int c = 0; c < 3; c++) {
        const auto it = b->shown[c][t.side].find(t.nick);
        if (it == b->shown[c][t.side].end()) continue;
        const auto [hp, max, st] = it->second;
        const bool exact = c == 0 || t.side == c - 1;
        const uint16_t want_hp = exact ? t.hp : view_percent(t.hp, t.max);
        const uint16_t want_max = exact ? t.max : 100;
        compared++;
        std::string what;
        if (hp != want_hp) what = "hp";
        else if (hp && max != want_max) what = "max";
        else if (hp && format::status(st) != t.status) what = "status";
        if (what.empty()) continue;
        const std::string key = std::string(c == 0 ? "raw " : c == 1 ? "p1 " : "p2 ") + what;
        mismatches[key]++;
        if (printed++ < 10) {
          std::printf("== %s mismatch, battle %zu, side %d %s: shown %u/%u %s, truth %u/%u %s\n", key.c_str(),
                      battles, t.side + 1, t.nick.c_str(), hp, max, format::status(st).c_str(), want_hp,
                      want_max, t.status.c_str());
          for (const std::string& l : text[c]) std::printf("  %s\n", l.c_str());
        }
      }
    }
    truth.clear();
  };

  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with("B ")) {
      finish();
      battles++;
      b = std::make_unique<Battle>();
      broken = false;
      for (Names& n : b->names) n.player = {"Alice", "Bob 2"};
      continue;
    }
    if (line.starts_with("P ")) {
      const uint8_t s = uint8_t(line[2] - '0');
      const std::vector<std::string> nicks = split_tabs(line.substr(4));
      for (size_t k = 0; k < nicks.size() && k < 6; k++) {
        b->names[0].nickname[s][k] = nicks[k];
        b->names[1 + s].nickname[s][k] = nicks[k];
      }
      if (s == 1) {
        b->parsers[0] = std::make_unique<TextParser>(b->names[0], Stream::Omniscient);
        b->parsers[1] = std::make_unique<TextParser>(b->names[1], Stream::View, 0);
        b->parsers[2] = std::make_unique<TextParser>(b->names[2], Stream::View, 1);
        for (auto& p : b->parsers) p->skipped = &skipped;
      }
      continue;
    }
    if (line.starts_with("U ")) {
      const int c = line[2] - '0';
      if (c == 0) finish();
      text[c].clear();
      while (std::getline(in, line) && line != ".") {
        if (!line.empty()) text[c].push_back(line);
      }
      pending = true;
      continue;
    }
    if (line.starts_with("T ")) {
      const std::vector<std::string> f = split_tabs(line.substr(4));
      if (f.size() < 4) continue;
      truth.push_back({uint8_t(line[2] - '0'), f[0], uint16_t(std::stoi(f[1])), uint16_t(std::stoi(f[2])), f[3]});
      continue;
    }
    if (line.starts_with("E ")) finish();
  }
  finish();
  std::printf("lines read past:");
  for (const auto& [tag, n] : skipped) std::printf(" %s %zu,", tag.c_str(), n);
  std::printf("\n");
  size_t total = 0;
  for (const auto& [k, n] : mismatches) {
    std::printf("  %s: %zu\n", k.c_str(), n);
    total += n;
  }
  std::printf("check_showdown: %zu battles, %zu updates, %zu comparisons: %zu mismatches, %zu parse errors\n",
              battles, updates, compared, total, errors);
  return total + errors != 0;
}
