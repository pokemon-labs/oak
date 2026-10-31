// libpkmn's Gen 1 binary log (obs.hpp) -> Showdown's text protocol, as
// libpkmn's own driver writes it (cpp/extern/engine/src/pkg/protocol.ts,
// which libpkmn's integration test checks against Showdown's output) with
// Showdown's spellings where the two differ (data/mods/gen1: Mist is
// -activate, not -block; no trailing empty field on -activate; no [of] on
// Recoil, found by tests/cpp/check_showdown.cpp). The text
// parser (text_parser.hpp) is its inverse; tests/cpp/check_text_parser.cpp
// checks the round trip.
//
// Two streams, as birch's Formatter has:
// - Omniscient: the raw bytes (party positions, exact HP). A line with HP
//   (switch, -damage, -heal) is Showdown's |split|pN triplet: the exact
//   line, then the public one with the HP as percent/100 (view.hpp).
// - View: one player's bytes as view.hpp rewrote them (the other side
//   numbered by first appearance, its HP already percent/100), printed as
//   they are: Showdown's stream for that player.
//
// Pokémon are named by `names` (the nickname per side and number: the party
// position, or for the other side in a view, the view's number).

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "names.hpp"
#include "obs.hpp"
#include "view.hpp"

namespace ataraxos {

struct Names {
  std::array<std::string, 2> player = {"p1", "p2"};
  std::array<std::array<std::string, 6>, 2> nickname{};
};

enum class Stream : uint8_t { Omniscient, View };

namespace format {

// libpkmn's reasons (cpp/extern/engine/src/lib/common/protocol.zig).
enum Cant : uint8_t { CantSleep, CantFreeze, CantParalysis, CantBound, CantFlinch, CantDisable, CantRecharge, CantPP };
enum Damage : uint8_t { DmgNone, DmgPoison, DmgBurn, DmgConfusion, DmgLeechSeed, DmgRecoilOf };
enum Heal : uint8_t { HealNone, HealSilent, HealDrain };
enum Status : uint8_t { StatusNone, StatusSilent, StatusFrom };
enum CureStatus : uint8_t { CureMessage, CureSilent };
enum Fail : uint8_t { FailNone, FailSleep, FailPoison, FailBurn, FailFreeze, FailParalysis, FailToxic, FailSubstitute, FailWeak };
enum Activate : uint8_t { ActBide, ActConfusion, ActHaze, ActMist, ActStruggle, ActSubstitute, ActSplash };
enum Start : uint8_t {
  StartBide, StartConfusion, StartConfusionSilent, StartFocusEnergy, StartLeechSeed, StartLightScreen,
  StartMist, StartReflect, StartSubstitute, StartTypeChange, StartDisable, StartMimic
};
enum End : uint8_t {
  EndDisable = 0, EndConfusion = 1, EndBide = 2, EndSubstitute = 3,
  EndDisableSilent = 15, EndConfusionSilent, EndMistSilent, EndFocusEnergySilent, EndLeechSeedSilent,
  EndToxicSilent, EndLightScreenSilent, EndReflectSilent, EndBideSilent
};
enum Immune : uint8_t { ImmuneNone, ImmuneOHKO };

inline constexpr const char* CANT[] = {"slp", "frz", "par", "partiallytrapped", "flinch", "Disable", "recharge", "nopp"};
inline constexpr const char* DAMAGE[] = {"", "psn", "brn", "confusion", "Leech Seed", "Recoil"};
inline constexpr const char* BOOSTS[] = {"atk", "atk", "def", "spe", "spa", "spd", "accuracy", "evasion"};
inline constexpr const char* FAIL[] = {"", "slp", "psn", "brn", "frz", "par", "tox"};
inline constexpr const char* ACTIVATE[] = {"Bide", "confusion", "move: Haze", "move: Mist", "move: Struggle", "Substitute", "move: Splash"};
inline constexpr const char* START[] = {"Bide", "confusion", "confusion", "move: Focus Energy", "move: Leech Seed",
                                        "Light Screen", "Mist", "Reflect", "Substitute"};
// End by reason; the silent ones from EndDisableSilent on.
inline constexpr const char* END[] = {"Disable", "confusion", "Bide", "Substitute"};
inline constexpr const char* END_SILENT[] = {"Disable", "confusion", "Mist", "move: Focus Energy", "move: Leech Seed",
                                             "Toxic counter", "Light Screen", "Reflect", "move: Bide"};

// Showdown's status text (any sleep "slp"), "" for none.
inline std::string status(uint8_t s) {
  if (s & 0b111) return "slp";
  if (s & 0b10000000) return "tox";
  if (s & 0b01000000) return "par";
  if (s & 0b00100000) return "frz";
  if (s & 0b00010000) return "brn";
  if (s & 0b00001000) return "psn";
  return "";
}

inline std::string hp(uint16_t h, uint16_t max, uint8_t s) {
  if (h == 0) return "0 fnt";
  std::string out = std::to_string(h) + "/" + std::to_string(max);
  const std::string st = status(s);
  return st.empty() ? out : out + " " + st;
}

inline std::string move_name(uint8_t m) { return std::string(names::MOVES.at(m)); }
inline std::string species_name(uint8_t s) { return std::string(names::SPECIES.at(s)); }

} // namespace format

class Formatter {
 public:
  Formatter(const Names& names, Stream stream) : names_(names), stream_(stream) {}

  // One update's bytes (up to its null) -> its lines.
  std::vector<std::string> format(std::span<const uint8_t> bytes) const {
    std::vector<std::string> out;
    size_t last_move = SIZE_MAX;
    size_t pos = 0;
    while (pos < bytes.size()) {
      const ArgType type = ArgType(bytes[pos++]);
      if (type == ArgType::None) break;
      const uint8_t* d = bytes.data() + pos;
      pos += message_size(type, d);
      if (type == ArgType::LastStill || type == ArgType::LastMiss) {
        if (last_move != SIZE_MAX) out[last_move] += type == ArgType::LastStill ? "|[still]" : "|[miss]";
        continue;
      }
      if (type == ArgType::Move) last_move = out.size();
      line(type, d, out);
    }
    return out;
  }

  std::string ident(uint8_t id) const {
    if (id == 0) return "";
    const uint8_t p = ident_player(id);
    return "p" + std::to_string(p + 1) + "a: " + names_.nickname[p][ident_slot(id) - 1];
  }

 private:
  const Names& names_;
  Stream stream_;

  bool split(ArgType t) const {
    return stream_ == Stream::Omniscient &&
           (t == ArgType::Switch || t == ArgType::Damage || t == ArgType::Heal);
  }

  // A line with HP fields at d[at..at+4]: a triplet in the omniscient stream.
  void health_line(const std::string& head, const uint8_t* d, size_t at, const std::string& tail,
                   std::vector<std::string>& out) const {
    const uint16_t h = uint16_t(d[at] | d[at + 1] << 8), max = uint16_t(d[at + 2] | d[at + 3] << 8);
    const uint8_t s = d[at + 4];
    if (stream_ == Stream::View) {
      out.push_back(head + format::hp(h, max, s) + tail);
      return;
    }
    out.push_back("|split|p" + std::to_string(ident_player(d[0]) + 1));
    out.push_back(head + format::hp(h, max, s) + tail);
    out.push_back(head + format::hp(view_percent(h, max), max ? 100 : 0, s) + tail);
  }

  void line(ArgType type, const uint8_t* d, std::vector<std::string>& out) const {
    using namespace format;
    switch (type) {
    case ArgType::Move: {
      std::string s = "|move|" + ident(d[0]) + "|" + move_name(d[1]) + "|" + ident(d[2]);
      if (d[3] != 0) s += "|[from] " + move_name(d[4]);
      out.push_back(s);
      return;
    }
    case ArgType::Switch: {
      std::string details = species_name(d[1]);
      if (d[2] != 100) details += ", L" + std::to_string(d[2]);
      return health_line("|switch|" + ident(d[0]) + "|" + details + "|", d, 3, "", out);
    }
    case ArgType::Cant: {
      std::string s = "|cant|" + ident(d[0]) + "|" + CANT[d[1]];
      if (d[1] == CantDisable) s += "|" + move_name(d[2]);
      out.push_back(s);
      return;
    }
    case ArgType::Faint: out.push_back("|faint|" + ident(d[0])); return;
    case ArgType::Turn: out.push_back("|turn|" + std::to_string(d[0] | d[1] << 8)); return;
    case ArgType::Win: out.push_back("|win|" + names_.player[d[0] ? 1 : 0]); return;
    case ArgType::Tie: out.push_back("|tie"); return;
    case ArgType::Damage: {
      std::string tail;
      // Recoil's [of] (the other side's active) isn't in Showdown's text
      if (d[6] != DmgNone) tail = std::string("|[from] ") + DAMAGE[d[6]];
      return health_line("|-damage|" + ident(d[0]) + "|", d, 1, tail, out);
    }
    case ArgType::Heal: {
      std::string tail;
      if (d[6] == HealSilent) tail = "|[silent]";
      if (d[6] == HealDrain) tail = "|[from] drain|[of] " + ident(d[7]);
      return health_line("|-heal|" + ident(d[0]) + "|", d, 1, tail, out);
    }
    case ArgType::Status: {
      std::string s = "|-status|" + ident(d[0]) + "|" + status(d[1]);
      if (d[2] == StatusSilent) s += "|[silent]";
      if (d[2] == StatusFrom) s += "|[from] move: " + move_name(d[3]);
      out.push_back(s);
      return;
    }
    case ArgType::CureStatus:
      out.push_back("|-curestatus|" + ident(d[0]) + "|" + status(d[1]) +
                    (d[2] == CureSilent ? "|[silent]" : "|[msg]"));
      return;
    case ArgType::Boost: {
      const int n = int(d[2]) - 6;
      std::string s = std::string(n > 0 ? "|-boost|" : "|-unboost|") + ident(d[0]) + "|" + BOOSTS[d[1]] + "|" +
                      std::to_string(n < 0 ? -n : n);
      if (d[1] == 0) s += "|[from] Rage";
      out.push_back(s);
      return;
    }
    case ArgType::ClearAllBoost: out.push_back("|-clearallboost|[silent]"); return;
    case ArgType::Fail: {
      std::string s = "|-fail|" + ident(d[0]);
      if (d[1] == FailSubstitute || d[1] == FailWeak) s += "|move: Substitute";
      else if (d[1] != FailNone) s += std::string("|") + FAIL[d[1]];
      if (d[1] == FailWeak) s += "|[weak]";
      out.push_back(s);
      return;
    }
    case ArgType::Miss: out.push_back("|-miss|" + ident(d[0])); return;
    case ArgType::HitCount: out.push_back("|-hitcount|" + ident(d[0]) + "|" + std::to_string(d[1])); return;
    case ArgType::Prepare: out.push_back("|-prepare|" + ident(d[0]) + "|" + move_name(d[1])); return;
    case ArgType::MustRecharge: out.push_back("|-mustrecharge|" + ident(d[0])); return;
    case ArgType::Activate:
      if (d[1] == ActSplash) out.push_back("|-activate||move: Splash");
      else out.push_back("|-activate|" + ident(d[0]) + "|" + ACTIVATE[d[1]] + (d[1] == ActSubstitute ? "|[damage]" : ""));
      return;
    case ArgType::FieldActivate: out.push_back("|-fieldactivate|move: Pay Day"); return;
    case ArgType::Start: {
      std::string s = "|-start|" + ident(d[0]) + "|";
      if (d[1] < StartTypeChange) {
        s += START[d[1]];
        if (d[1] == StartConfusionSilent) s += "|[silent]";
      } else if (d[1] == StartTypeChange) {
        const std::string a(names::TYPES.at(d[2] & 0xF)), b(names::TYPES.at(d[2] >> 4));
        s += "typechange|" + (a == b ? a : a + "/" + b) + "|[from] move: Conversion|[of] " + ident(d[3]);
      } else {
        s += std::string(d[1] == StartDisable ? "Disable|" : "Mimic|") + move_name(d[2]);
      }
      out.push_back(s);
      return;
    }
    case ArgType::End:
      if (d[1] >= EndDisableSilent) {
        out.push_back("|-end|" + ident(d[0]) + "|" + END_SILENT[d[1] - EndDisableSilent] + "|[silent]");
      } else {
        out.push_back("|-end|" + ident(d[0]) + "|" + END[d[1]]);
      }
      return;
    case ArgType::OHKO: out.push_back("|-ohko"); return;
    case ArgType::Crit: out.push_back("|-crit|" + ident(d[0])); return;
    case ArgType::SuperEffective: out.push_back("|-supereffective|" + ident(d[0])); return;
    case ArgType::Resisted: out.push_back("|-resisted|" + ident(d[0])); return;
    case ArgType::Immune: out.push_back("|-immune|" + ident(d[0]) + (d[1] == ImmuneOHKO ? "|[ohko]" : "")); return;
    case ArgType::Transform: out.push_back("|-transform|" + ident(d[0]) + "|" + ident(d[1])); return;
    default: out.push_back("|unknown"); return;
    }
  }
};

} // namespace ataraxos
