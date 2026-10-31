// Checks the text parser (src/text_parser.hpp) by round trip: each update's
// binary log -> format.hpp's text -> TextParser -> the same bytes exactly.
// Three streams per update: the omniscient log (its |split| triplets read by
// the exact line), and each player's view (view.hpp, formatted as
// Stream::View: Showdown's stream for that player). A view's parser starts
// with only its own side's nicknames and numbers the other side's as they
// switch in, as on the ladder.
//
// Random self-play on truly random teams (teams.hpp), uniform choices.
// Nicknames are the species' names, made distinct within a side (repeats get
// " 2", " 3", ...). A mismatch is counted by the first differing message's
// kind and printed with its update's text and both byte strings.
//
// The text's one loss is ignored: the sleep counter and Rest's flag ("slp"
// parses as status 1), so the omniscient bytes are compared with their
// status bytes as the view writes them.
//
//   check_text_parser [GAMES] [SEED]

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "battle_data.hpp"
#include "format.hpp"
#include "teams.hpp"
#include "text_parser.hpp"

using namespace ataraxos;

namespace {

std::map<std::string, size_t> failures;
int printed = 0;

const char* kind_name(ArgType t) {
  static const char* NAMES[ARG_TYPES] = {
      "None", "LastStill", "LastMiss", "Move", "Switch", "Cant", "Faint", "Turn", "Win", "Tie", "Damage",
      "Heal", "Status", "CureStatus", "Boost", "ClearAllBoost", "Fail", "Miss", "HitCount", "Prepare",
      "MustRecharge", "Activate", "FieldActivate", "Start", "End", "OHKO", "Crit", "SuperEffective",
      "Resisted", "Immune", "Transform", "Drag", "Item", "EndItem", "CureTeam", "SetHP", "SetBoost",
      "CopyBoost", "SideStart", "SideEnd", "SingleMove", "SingleTurn", "Weather"};
  return uint8_t(t) < ARG_TYPES ? NAMES[uint8_t(t)] : "?";
}

// The status bytes as the text shows them (sleep as 1).
void text_statuses(std::vector<uint8_t>& bytes) {
  size_t i = 0;
  while (i < bytes.size() && bytes[i] != 0) {
    const ArgType t = ArgType(bytes[i]);
    uint8_t* d = &bytes[i + 1];
    if (t == ArgType::Switch) d[7] = view_status(d[7]);
    if (t == ArgType::Damage || t == ArgType::Heal) d[5] = view_status(d[5]);
    if (t == ArgType::Status || t == ArgType::CureStatus) d[1] = view_status(d[1]);
    i += 1 + message_size(t, d);
  }
}

std::string hex(const std::vector<uint8_t>& bytes, size_t from, size_t to) {
  std::string s;
  char buf[4];
  for (size_t i = from; i < to && i < bytes.size(); i++) {
    std::snprintf(buf, sizeof buf, "%02x ", bytes[i]);
    s += buf;
  }
  return s;
}

void compare(const char* stream, const std::vector<uint8_t>& want, const std::vector<uint8_t>& got,
             const std::vector<std::string>& text) {
  if (want == got) return;
  // the first message where they differ
  size_t i = 0, j = 0;
  ArgType kind = ArgType::None;
  while (i < want.size() && j < got.size()) {
    const size_t a = want[i] ? 1 + message_size(ArgType(want[i]), &want[i + 1]) : 1;
    const size_t b = got[j] ? 1 + message_size(ArgType(got[j]), &got[j + 1]) : 1;
    if (a != b || !std::equal(want.begin() + i, want.begin() + i + a, got.begin() + j)) {
      kind = ArgType(want[i]);
      break;
    }
    i += a;
    j += b;
  }
  const std::string key = std::string(stream) + " " + kind_name(kind);
  failures[key]++;
  if (printed++ < 10) {
    std::printf("== %s mismatch\n", key.c_str());
    for (const std::string& l : text) std::printf("  %s\n", l.c_str());
    std::printf("  want %s\n  got  %s\n", hex(want, i, want.size()).c_str(), hex(got, j, got.size()).c_str());
  }
}

} // namespace

int main(int argc, char** argv) {
  const size_t games = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 2000;
  std::mt19937_64 rng(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1);
  size_t updates = 0, errors = 0;
  for (size_t g = 0; g < games; g++) {
    const Team t1 = random_team(rng), t2 = random_team(rng);
    pkmn_gen1_battle b = init_battle(rng(), t1, t2);
    const PKMN::Battle& battle = PKMN::view(b);

    Names omni;
    omni.player = {"Alice", "Bob 2"};
    for (uint8_t s = 0; s < 2; s++) {
      std::map<std::string, int> count;
      for (uint8_t k = 0; k < 6; k++) {
        const uint8_t sp = uint8_t(battle.sides[s].pokemon[k].species);
        if (!sp) continue;
        const std::string n(names::SPECIES[sp]);
        const int c = ++count[n];
        omni.nickname[s][k] = c == 1 ? n : n + " " + std::to_string(c);
      }
    }
    // each view's own: the formatter's (the other side by the view's
    // numbers, filled as it numbers them) and the parser's (ours only)
    std::array<Names, 2> fmt_names, parse_names;
    std::array<TextParser*, 2> parsers{};
    for (uint8_t p = 0; p < 2; p++) {
      fmt_names[p].player = parse_names[p].player = omni.player;
      fmt_names[p].nickname[p] = parse_names[p].nickname[p] = omni.nickname[p];
      parsers[p] = new TextParser(parse_names[p], Stream::View, p);
    }
    Names omni_parse = omni;
    TextParser omni_parser(omni_parse, Stream::Omniscient);
    const Formatter omni_fmt(omni, Stream::Omniscient);

    std::array<uint8_t, LOG_SIZE> buf{};
    pkmn_gen1_battle_options options{};
    View views[2] = {View{0}, View{1}};
    pkmn_choice c1 = 0, c2 = 0;
    for (int turn = 0; turn < 1000; turn++) {
      pkmn_gen1_log_options log{buf.data(), buf.size()};
      pkmn_gen1_battle_options_set(&options, &log, nullptr, nullptr);
      const pkmn_result r = pkmn_gen1_battle_update(&b, c1, c2, &options);
      updates++;
      size_t used = 0;
      while (buf[used] != 0) used += 1 + message_size(ArgType(buf[used]), &buf[used + 1]);

      std::vector<uint8_t> raw(buf.begin(), buf.begin() + std::ptrdiff_t(used + 1));
      try {
        const std::vector<std::string> text = omni_fmt.format(raw);
        std::vector<uint8_t> want = raw;
        text_statuses(want);
        compare("omniscient", want, omni_parser.parse(text), text);
      } catch (const ParseError& e) {
        if (errors++ < 10) std::printf("omniscient: %s\n", e.what());
      }
      for (uint8_t p = 0; p < 2; p++) {
        std::vector<uint8_t> v = raw;
        views[p].apply(v);
        const uint8_t foe = 1 - p;
        for (uint8_t slot = 1; slot <= 6; slot++) {
          if (views[p].number[slot]) fmt_names[p].nickname[foe][views[p].number[slot] - 1] = omni.nickname[foe][slot - 1];
        }
        const std::vector<std::string> text = Formatter(fmt_names[p], Stream::View).format(v);
        try {
          compare(p ? "p2 view" : "p1 view", v, parsers[p]->parse(text), text);
        } catch (const ParseError& e) {
          if (errors++ < 10) std::printf("p%d view: %s\n", p + 1, e.what());
        }
      }

      if (pkmn_result_type(r)) break;
      pkmn_choice chosen[2];
      for (uint8_t p = 0; p < 2; p++) {
        std::array<pkmn_choice, MAX_CHOICES> opts{};
        const pkmn_choice_kind kind = p ? pkmn_result_p2(r) : pkmn_result_p1(r);
        const size_t n = pkmn_gen1_battle_choices(&b, p ? PKMN_PLAYER_P2 : PKMN_PLAYER_P1, kind, opts.data(),
                                                  opts.size());
        chosen[p] = opts[rng() % n];
      }
      c1 = chosen[0];
      c2 = chosen[1];
    }
    for (TextParser* x : parsers) delete x;
  }
  size_t total = 0;
  for (const auto& [k, n] : failures) {
    std::printf("  %s: %zu\n", k.c_str(), n);
    total += n;
  }
  std::printf("check_text_parser: %zu games, %zu updates: %zu mismatches, %zu parse errors\n", games, updates, total,
              errors);
  return total + errors != 0;
}
