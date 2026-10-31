// Showdown's text protocol -> libpkmn's Gen 1 binary log: the inverse of
// format.hpp's Formatter (birch's text_parser.hpp for Gen 1), so the
// bookkeeping and the net read one input whether the battle came from
// libpkmn or from Showdown.
//
// It reads either of the Formatter's streams. The omniscient one: a
// |split|pN triplet is taken by its exact (second) line. A player's stream
// (Stream::View, what Showdown sends that player): the other side's HP as
// HP/100, and it parses into that player's view bytes (view.hpp). There the
// other side's Pokémon are numbered by first appearance: a |switch| with a
// nickname not yet in `names` takes the next number, as the view numbers
// them, so on the ladder only our own nicknames (the request's) are needed.
//
// What the text doesn't carry is rebuilt where libpkmn's order fixes it
// (cpp/extern/engine/src/lib/gen1/mechanics.zig, Showdown mode):
//
// - LastStill and LastMiss from the move line's [still] and [miss]:
//   LastMiss right before the next -miss, LastStill right before the next
//   -prepare (the charging turn); one left over goes right after its line.
// - "0 fnt" has no max HP or status: the last ones shown for that Pokémon.
// - -activate||move: Splash has no ident: the last move line's user.
// - Recoil's [of], which Showdown doesn't write: the other side's active
//   Pokémon, as libpkmn's is.
//
// And what it can't, so a round trip differs there: the sleep counter and
// Rest's flag ("slp" is status 1, as the view writes it).
//
// Showdown writes some things libpkmn doesn't, and they're read past: the
// lines in SKIP (libpkmn's integration test's FILTER, and the room's), an
// [of] on any damage but Recoil's (Leech Seed's, a status's), a [from] on a
// move line that isn't another move's (lockedmove), -block for Mist's
// -activate, and a trailing empty field.
//
// Text it can't read (an unknown line, name or nickname, a malformed field)
// throws ParseError, so a ladder client can give up on one battle without
// crashing.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "format.hpp"
#include "names.hpp"
#include "obs.hpp"

namespace ataraxos {

struct ParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class TextParser {
 public:
  // `viewer` is the player whose stream a View is (unused for Omniscient).
  // `names` must give each Pokémon of a side a distinct nickname.
  TextParser(Names& names, Stream stream, uint8_t viewer = 0) : names_(names), stream_(stream), viewer_(viewer) {
    tables();
  }

  // When set, counts the lines read past (SKIP's) by tag.
  std::map<std::string, size_t, std::less<>>* skipped = nullptr;

  // One update's lines -> its bytes, ending with None.
  std::vector<uint8_t> parse(const std::vector<std::string>& lines) {
    std::vector<std::string_view> ls;
    for (size_t i = 0; i < lines.size(); i++) {
      if (lines[i].starts_with("|split|")) {
        if (i + 2 >= lines.size()) throw ParseError("a |split| with no lines after it");
        ls.push_back(lines[i + 1]);
        i += 2;
      } else {
        ls.push_back(lines[i]);
      }
    }
    out_.clear();
    move_at_ = NONE;
    still_ = miss_ = false;
    for (const std::string_view l : ls) {
      try {
        line(split(l));
      } catch (const ParseError& e) {
        throw ParseError(std::string(e.what()) + " in \"" + std::string(l) + "\"");
      }
    }
    flush();
    b(0);
    return out_;
  }

 private:
  static constexpr size_t NONE = SIZE_MAX;

  // A line's positional fields (after the tag) and its [keyword] fields.
  struct Line {
    std::string_view tag;
    std::vector<std::string_view> args;
    std::vector<std::string_view> kw;

    std::string_view operator[](size_t i) const {
      if (i >= args.size()) throw ParseError("a field missing");
      return args[i];
    }
    // The keyword `key`'s value ("" for a flag); `found` whether it's there.
    std::string_view get(std::string_view key, bool* found = nullptr) const {
      for (const std::string_view k : kw) {
        if (k.starts_with(key) && (k.size() == key.size() || k[key.size()] == ' ')) {
          if (found) *found = true;
          return k.size() == key.size() ? std::string_view() : k.substr(key.size() + 1);
        }
      }
      if (found) *found = false;
      return {};
    }
    bool has(std::string_view key) const {
      bool f;
      get(key, &f);
      return f;
    }
  };

  // Lines libpkmn has no message for (its integration test's FILTER, and
  // what a room adds).
  inline static const std::set<std::string, std::less<>> SKIP = {
      "", "t:", "gametype", "player", "teamsize", "gen", "message", "done", "error", "bigerror", "tier",
      "rule", "start", "upkeep", "-message", "-hint", "debug", "j", "J", "l", "L", "n", "N", "c", "c:",
      "raw", "html", "uhtml", "uhtmlchange", "inactive", "inactiveoff", "title", "request", "init",
      "clearpoke", "poke", "teampreview", "badge", "rated", "seed", "timestamp", "join", "leave", "chat",
      "name", "deinit", "noinit", "expire", "notify", "tempnotify", "tempnotifyoff", "unlink", "bigerror",
      "sentchoice", "callback", "zbroken"};

  Names& names_;
  Stream stream_;
  uint8_t viewer_;
  std::vector<uint8_t> out_;

  // The last move line: where its bytes end, its user, and the amendments
  // its text shows that are still to be placed.
  size_t move_at_ = NONE;
  uint8_t move_source_ = 0;
  bool still_ = false, miss_ = false;

  // Per side, the active Pokémon's ident.
  std::array<uint8_t, 2> active_{};
  // Per side and number, the last max HP and status shown.
  std::array<std::array<uint16_t, 6>, 2> max_hp_{};
  std::array<std::array<uint8_t, 6>, 2> status_{};

  std::map<std::string, uint8_t, std::less<>> species_, moves_, types_;

  // --- Output.

  void b(uint8_t x) { out_.push_back(x); }
  void u16(uint16_t v) {
    b(uint8_t(v));
    b(uint8_t(v >> 8));
  }
  void t(ArgType a) { b(uint8_t(a)); }

  // Places the amendments left on the last move line right after it.
  void flush() {
    if (move_at_ == NONE) return;
    std::vector<uint8_t> a;
    if (still_) a.push_back(uint8_t(ArgType::LastStill));
    if (miss_) a.push_back(uint8_t(ArgType::LastMiss));
    out_.insert(out_.begin() + std::ptrdiff_t(move_at_), a.begin(), a.end());
    still_ = miss_ = false;
    move_at_ = NONE;
  }

  // --- Reading.

  static void need(bool ok, const std::string& what) {
    if (!ok) throw ParseError(what);
  }

  static Line split(std::string_view s) {
    need(s.size() >= 1 && s[0] == '|', "not a protocol line");
    std::vector<std::string_view> p;
    size_t start = 1;
    while (true) {
      const size_t bar = s.find('|', start);
      p.push_back(s.substr(start, bar == std::string_view::npos ? bar : bar - start));
      if (bar == std::string_view::npos) break;
      start = bar + 1;
    }
    Line l;
    l.tag = p[0];
    for (size_t i = 1; i < p.size(); i++) {
      if (p[i].starts_with("[")) l.kw.push_back(p[i]);
      else l.args.push_back(p[i]);
    }
    // a trailing empty field (protocol.ts's -activate) is nothing
    while (!l.args.empty() && l.args.back().empty() && l.tag != "move") l.args.pop_back();
    return l;
  }

  static std::string_view after(std::string_view s, std::string_view prefix) {
    need(s.starts_with(prefix), "expected \"" + std::string(prefix) + "\"");
    return s.substr(prefix.size());
  }

  template <typename Map> static uint8_t find(const Map& m, std::string_view k) {
    const auto it = m.find(k);
    need(it != m.end(), "unknown name \"" + std::string(k) + "\"");
    return it->second;
  }

  uint8_t move_named(std::string_view s) const { return find(moves_, s); }

  static uint16_t number(std::string_view s) {
    uint32_t n = 0;
    need(!s.empty() && s.size() <= 5, "bad number \"" + std::string(s) + "\"");
    for (const char c : s) {
      need(c >= '0' && c <= '9', "bad number \"" + std::string(s) + "\"");
      n = n * 10 + uint32_t(c - '0');
    }
    need(n <= 0xFFFF, "bad number \"" + std::string(s) + "\"");
    return uint16_t(n);
  }

  // "p1a: NICK" -> libpkmn's ident byte. `fresh`: a switch may bring in the
  // other side's next Pokémon in a view.
  uint8_t ident(std::string_view s, bool fresh = false) {
    const size_t colon = s.find(": ");
    need(s.size() > 4 && s[0] == 'p' && (s[1] == '1' || s[1] == '2') && (colon == 2 || colon == 3),
         "bad ident \"" + std::string(s) + "\"");
    const uint8_t p = s[1] == '1' ? 0 : 1;
    const std::string_view nick = s.substr(colon + 2);
    auto& nicks = names_.nickname[p];
    for (uint8_t i = 0; i < 6; i++) {
      if (nicks[i] == nick) return uint8_t((p ? IDENT_P2 : 0) | (i + 1));
    }
    if (fresh && stream_ == Stream::View && p != viewer_) {
      for (uint8_t i = 0; i < 6; i++) {
        if (nicks[i].empty()) {
          nicks[i] = std::string(nick);
          return uint8_t((p ? IDENT_P2 : 0) | (i + 1));
        }
      }
    }
    need(false, "unknown nickname \"" + std::string(nick) + "\"");
    return 0;
  }
  // A move's target: "" or "null" is none.
  uint8_t target(std::string_view s) { return s.empty() || s == "null" ? 0 : ident(s); }

  static uint8_t status_bits(std::string_view s) {
    if (s == "slp") return 1;
    if (s == "tox") return 0b10001000;
    if (s == "psn") return 0b00001000;
    if (s == "brn") return 0b00010000;
    if (s == "frz") return 0b00100000;
    if (s == "par") return 0b01000000;
    need(s.empty(), "unknown status \"" + std::string(s) + "\"");
    return 0;
  }

  // "HP/MAX[ STATUS]" or "0 fnt", for the Pokémon `x`: writes HP, max, status.
  void hp(uint8_t x, std::string_view s) {
    uint16_t& max = max_hp_[ident_player(x)][ident_slot(x) - 1];
    uint8_t& st = status_[ident_player(x)][ident_slot(x) - 1];
    if (s == "0 fnt") {
      u16(0);
      u16(max);
      b(st);
      return;
    }
    const size_t slash = s.find('/'), space = s.find(' ');
    need(slash != std::string_view::npos, "bad HP \"" + std::string(s) + "\"");
    const uint16_t h = number(s.substr(0, slash));
    max = number(s.substr(slash + 1, space == std::string_view::npos ? space : space - slash - 1));
    st = space == std::string_view::npos ? 0 : status_bits(s.substr(space + 1));
    u16(h);
    u16(max);
    b(st);
  }

  template <size_t N>
  static uint8_t index(const char* const (&table)[N], std::string_view s, size_t from = 0) {
    for (size_t i = from; i < N; i++) {
      if (s == table[i]) return uint8_t(i);
    }
    need(false, "unknown form \"" + std::string(s) + "\"");
    return 0;
  }

  void tables() {
    for (size_t i = 1; i < names::SPECIES.size(); i++) species_.emplace(names::SPECIES[i], uint8_t(i));
    for (size_t i = 1; i < names::MOVES.size(); i++) moves_.emplace(names::MOVES[i], uint8_t(i));
    for (size_t i = 0; i < names::TYPES.size(); i++) types_.emplace(names::TYPES[i], uint8_t(i));
  }

  // --- Lines.

  void line(const Line& l) {
    using namespace format;
    const std::string_view tag = l.tag;
    if (SKIP.contains(tag)) {
      if (skipped) (*skipped)[std::string(tag)]++;
      return;
    }
    if (tag == "move") {
      flush();
      const uint8_t source = ident(l[0]);
      const uint8_t m = move_named(l[1]);
      const uint8_t tgt = l.args.size() > 2 ? target(l[2]) : 0;
      // [from] another move (Metronome's, Mirror Move's), not lockedmove
      bool from = false;
      const std::string_view f = l.get("[from]", &from);
      from = from && moves_.contains(f);
      t(ArgType::Move);
      b(source);
      b(m);
      b(tgt);
      b(from ? 1 : 0);
      if (from) b(move_named(f));
      move_at_ = out_.size();
      move_source_ = source;
      still_ = l.has("[still]");
      miss_ = l.has("[miss]");
      return;
    }
    if (tag == "switch" || tag == "drag") {
      need(tag == "switch", "drag in Gen 1");
      const uint8_t x = ident(l[0], true);
      // DETAILS: species[, Llevel][, M|F][, shiny]
      std::string_view d = l[1];
      const size_t comma = d.find(", ");
      const uint8_t s = find(species_, d.substr(0, comma));
      uint8_t level = 100;
      while (d.find(", ") != std::string_view::npos) {
        d = d.substr(d.find(", ") + 2);
        if (d.starts_with("L")) level = uint8_t(number(d.substr(1, d.find(", ") == std::string_view::npos
                                                                        ? std::string_view::npos
                                                                        : d.find(", ") - 1)));
      }
      active_[ident_player(x)] = x;
      t(ArgType::Switch);
      b(x);
      b(s);
      b(level);
      hp(x, l[2]);
      return;
    }
    if (tag == "cant") {
      t(ArgType::Cant);
      b(ident(l[0]));
      const uint8_t r = index(CANT, l[1]);
      b(r);
      if (r == CantDisable) b(move_named(l[2]));
      return;
    }
    if (tag == "faint") return one(ArgType::Faint, l);
    if (tag == "turn") {
      t(ArgType::Turn);
      u16(number(l[0]));
      return;
    }
    if (tag == "win") {
      t(ArgType::Win);
      b(l[0] == names_.player[1] ? 1 : 0);
      return;
    }
    if (tag == "tie") return t(ArgType::Tie);
    if (tag == "-damage") {
      const uint8_t x = ident(l[0]);
      t(ArgType::Damage);
      b(x);
      hp(x, l[1]);
      bool from = false;
      const std::string_view f = l.get("[from]", &from);
      if (!from) return b(DmgNone);
      const uint8_t r = index(DAMAGE, f, 1);
      b(r);
      if (r == DmgRecoilOf) b(l.has("[of]") ? ident(l.get("[of]")) : active_[1 - ident_player(x)]);
      return;
    }
    if (tag == "-heal") {
      const uint8_t x = ident(l[0]);
      t(ArgType::Heal);
      b(x);
      hp(x, l[1]);
      if (l.has("[silent]")) return b(HealSilent);
      bool from = false;
      const std::string_view f = l.get("[from]", &from);
      if (!from) return b(HealNone);
      need(f == "drain", "unknown heal");
      b(HealDrain);
      b(ident(l.get("[of]")));
      return;
    }
    if (tag == "-status") {
      t(ArgType::Status);
      const uint8_t x = ident(l[0]);
      b(x);
      b(status_[ident_player(x)][ident_slot(x) - 1] = status_bits(l[1]));
      bool from = false;
      const std::string_view f = l.get("[from]", &from);
      if (from) {
        b(StatusFrom);
        b(move_named(after(f, "move: ")));
      } else {
        b(l.has("[silent]") ? StatusSilent : StatusNone);
      }
      return;
    }
    if (tag == "-curestatus") {
      t(ArgType::CureStatus);
      const uint8_t x = ident(l[0]);
      b(x);
      b(status_bits(l[1]));
      status_[ident_player(x)][ident_slot(x) - 1] = 0;
      b(l.has("[silent]") ? CureSilent : CureMessage);
      return;
    }
    if (tag == "-boost" || tag == "-unboost") {
      t(ArgType::Boost);
      b(ident(l[0]));
      const uint8_t r = l.get("[from]") == "Rage" ? 0 : index(BOOSTS, l[1], 1);
      b(r);
      const int n = number(l[2]);
      b(uint8_t(6 + (tag == "-boost" ? n : -n)));
      return;
    }
    if (tag == "-clearallboost") return t(ArgType::ClearAllBoost);
    if (tag == "-fail") {
      t(ArgType::Fail);
      b(ident(l[0]));
      if (l.args.size() < 2) return b(FailNone);
      if (l[1] == "move: Substitute") return b(l.has("[weak]") ? FailWeak : FailSubstitute);
      b(index(FAIL, l[1], 1));
      return;
    }
    if (tag == "-miss") {
      if (miss_) {
        t(ArgType::LastMiss);
        miss_ = false;
      }
      return one(ArgType::Miss, l);
    }
    if (tag == "-hitcount") {
      one(ArgType::HitCount, l);
      b(uint8_t(number(l[1])));
      return;
    }
    if (tag == "-prepare") {
      if (still_) {
        t(ArgType::LastStill);
        still_ = false;
      }
      one(ArgType::Prepare, l);
      b(move_named(l[1]));
      return;
    }
    if (tag == "-mustrecharge") return one(ArgType::MustRecharge, l);
    if (tag == "-activate" || tag == "-block") {
      t(ArgType::Activate);
      if (l.args.size() >= 2 && l[1] == "move: Splash") {
        b(l[0].empty() ? move_source_ : ident(l[0]));
        b(ActSplash);
        return;
      }
      b(ident(l[0]));
      b(index(ACTIVATE, l[1]));
      return;
    }
    if (tag == "-fieldactivate") return t(ArgType::FieldActivate);
    if (tag == "-start") {
      t(ArgType::Start);
      b(ident(l[0]));
      const std::string_view e = l[1];
      if (e == "typechange") {
        b(StartTypeChange);
        const std::string_view ty = l[2];
        const size_t slash = ty.find('/');
        const uint8_t a = find(types_, ty.substr(0, slash));
        const uint8_t c = slash == std::string_view::npos ? a : find(types_, ty.substr(slash + 1));
        b(uint8_t(a | c << 4));
        b(ident(l.get("[of]")));
      } else if (e == "Disable" || e == "Mimic") {
        b(e == "Disable" ? StartDisable : StartMimic);
        b(move_named(l[2]));
      } else if (e == "confusion") {
        b(l.has("[silent]") ? StartConfusionSilent : StartConfusion);
      } else {
        b(index(START, e));
      }
      return;
    }
    if (tag == "-end") {
      t(ArgType::End);
      b(ident(l[0]));
      if (l.has("[silent]")) b(uint8_t(EndDisableSilent + index(END_SILENT, l[1])));
      else b(index(END, l[1]));
      return;
    }
    if (tag == "-ohko") return t(ArgType::OHKO);
    if (tag == "-crit") return one(ArgType::Crit, l);
    if (tag == "-supereffective") return one(ArgType::SuperEffective, l);
    if (tag == "-resisted") return one(ArgType::Resisted, l);
    if (tag == "-immune") {
      one(ArgType::Immune, l);
      b(l.has("[ohko]") ? ImmuneOHKO : ImmuneNone);
      return;
    }
    if (tag == "-transform") {
      one(ArgType::Transform, l);
      b(ident(l[1]));
      return;
    }
    need(false, "unknown line");
  }

  void one(ArgType a, const Line& l) {
    t(a);
    b(ident(l[0]));
  }
};

} // namespace ataraxos
