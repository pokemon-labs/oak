// One Showdown battle as one player sees it (birch's client.hpp for Gen 1):
// the battle room's text and the |request| JSON in, the network's input out.
// That is the same tokens, request fields, choices and bookkeeping state
// BattleData gives that player in training (battle_data.hpp), checked by
// tests/cpp/check_client.cpp.
//
// The driver (ataraxos/play/showdown.py) hands over each message of the
// battle room, its ">room" header line removed (`feed`), and each request
// (`request`). Showdown sends each update's log in one message and then the
// requests (sim/battle.ts sendUpdates; server/room-battle.ts forwards them
// in order), so a request comes after the log it answers.
//
// - The lines: those the TextParser reads, and those it reads past (its
//   SKIP); a tag on neither list is dropped and kept in `unknown` for the
//   driver to log. Each message is parsed on its own, as one update; a
//   rejoin's |init| message replays the whole log, and splits into its
//   updates at each `t:` line after `start`.
// - The names: our side's nicknames by party position, from the first
//   request with the switch-ins before it undone (a rejoin's first request
//   follows the replayed log; `name_side`). The parser numbers the other
//   side's as they switch in, as View does. Messages are held until the
//   first request, since the leads come in before it.
// - Our side: libpkmn's Init on the first request's sets (species, level,
//   moves by party position), as training builds it from the team (every
//   set maxes its DVs and stat experience). On a rejoin its HP, status and
//   PP are the team's at the start, so the state is off there.
// - The request: RequestData from the JSON, as request_data writes it from
//   the battle, and Showdown's choices for it (showdown_choices). A fainted
//   Pokémon's condition is "0 fnt", so its max HP is the last one shown.
// - The state: a Bookkeeping over each parsed update, as BattleData's with
//   `track`; it hears each request and each choice sent (`choose`), sized
//   by the `teamsize` lines; `state` is its encode_state.
//
// Text or JSON it can't read throws (ParseError, json's runtime_error); the
// driver gives up on modelling that battle and plays on from the requests.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bookkeeping.hpp"
#include "format.hpp"
#include "json.hpp"
#include "obs.hpp"
#include "request_data.hpp"
#include "state_tokens.hpp"
#include "teams.hpp"
#include "text_parser.hpp"
#include "tokens.hpp"

namespace ataraxos {

namespace client {

// Every tag the TextParser reads (the rest it reads past, or throws on).
inline constexpr std::string_view PARSED_TAGS[] = {
    "move", "switch", "drag", "cant", "faint", "turn", "win", "tie", "-damage", "-heal", "-status",
    "-curestatus", "-boost", "-unboost", "-clearallboost", "-fail", "-miss", "-hitcount", "-prepare",
    "-mustrecharge", "-activate", "-fieldactivate", "-start", "-end", "-ohko", "-crit", "-supereffective",
    "-resisted", "-immune", "-transform", "-block",
};

inline bool parsed(std::string_view tag) {
  for (const std::string_view t : PARSED_TAGS) {
    if (t == tag) return true;
  }
  return false;
}

inline std::vector<std::string_view> fields(std::string_view line) {
  std::vector<std::string_view> p;
  size_t start = 0;
  while (true) {
    const size_t bar = line.find('|', start);
    p.push_back(line.substr(start, bar == std::string_view::npos ? bar : bar - start));
    if (bar == std::string_view::npos) return p;
    start = bar + 1;
  }
}

[[noreturn]] inline void bad_request(const std::string& what) { throw ParseError("request: " + what); }

inline uint16_t number(std::string_view s) {
  if (s.empty() || s.size() > 5) bad_request("bad number \"" + std::string(s) + "\"");
  unsigned n = 0;
  for (const char c : s) {
    if (c < '0' || c > '9') bad_request("bad number \"" + std::string(s) + "\"");
    n = n * 10 + unsigned(c - '0');
  }
  if (n > 0xFFFF) bad_request("bad number \"" + std::string(s) + "\"");
  return uint16_t(n);
}

inline PKMN::Data::Move move_id(std::string_view id) {
  const auto m = ataraxos::detail::move(id);
  if (!m) bad_request("unknown move \"" + std::string(id) + "\"");
  return *m;
}

// "HP/MAX[ STATUS]" or "0 fnt" (max 0); the status as the text shows it.
inline void condition(std::string_view s, uint16_t& hp, uint16_t& max, uint8_t& status) {
  status = 0;
  if (s == "0 fnt") {
    hp = max = 0;
    return;
  }
  const size_t slash = s.find('/'), space = s.find(' ');
  if (slash == std::string_view::npos) bad_request("bad condition \"" + std::string(s) + "\"");
  hp = number(s.substr(0, slash));
  max = number(s.substr(slash + 1, space == std::string_view::npos ? space : space - slash - 1));
  if (space == std::string_view::npos) return;
  const std::string_view st = s.substr(space + 1);
  using S = PKMN::Data::Status;
  status = st == "slp"   ? 1
           : st == "tox" ? uint8_t(S::Toxic)
           : st == "psn" ? uint8_t(S::Poison)
           : st == "brn" ? uint8_t(S::Burn)
           : st == "frz" ? uint8_t(S::Freeze)
           : st == "par" ? uint8_t(S::Paralysis)
                         : (bad_request("bad status \"" + std::string(st) + "\""), 0);
}

inline const std::string& text(const json::Value& v, const char* key) {
  const json::Value& x = v[key];
  if (x.type != json::Value::String) bad_request(std::string("no string ") + key);
  return x.string;
}

inline uint16_t integer(const json::Value& v, const char* key) {
  const json::Value& x = v[key];
  if (x.type != json::Value::Number || x.number < 0 || x.number > 65535) bad_request(std::string("no number ") + key);
  return uint16_t(x.number);
}

// A boolean field, false when absent. Showdown's `disabled` may be a string
// (a source), which counts as true.
inline bool flag(const json::Value& v, const char* key) {
  const json::Value& x = v[key];
  if (x.type == json::Value::Bool) return x.boolean;
  if (x.type == json::Value::String) return !x.string.empty();
  return false;
}

inline std::string_view nickname(std::string_view ident) {
  const size_t colon = ident.find(": ");
  if (colon == std::string_view::npos) bad_request("bad ident \"" + std::string(ident) + "\"");
  return ident.substr(colon + 2);
}

} // namespace client

class Client {
 public:
  Client() = default;
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  // One message of the battle room, its ">room" line removed.
  void feed(std::string_view message) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= message.size()) {
      const size_t nl = message.find('\n', start);
      const std::string_view line = message.substr(start, nl == std::string_view::npos ? nl : nl - start);
      start = nl == std::string_view::npos ? message.size() + 1 : nl + 1;
      if (line.empty() || line[0] != '|') continue;
      const std::vector<std::string_view> p = client::fields(line);
      const std::string_view tag = p[1];
      if (tag == "t:" && started_ && !lines.empty()) {
        pending_.push_back(std::move(lines));
        lines.clear();
      }
      if (tag == "start") started_ = true;
      if (tag == "player" && p.size() > 3 && (p[2] == "p1" || p[2] == "p2")) {
        names_.player[p[2] == "p1" ? 0 : 1] = std::string(p[3]);
      }
      if (tag == "teamsize" && p.size() > 3 && (p[2] == "p1" || p[2] == "p2")) {
        teamsize_[p[2] == "p1" ? 0 : 1] = uint8_t(client::number(p[3]));
      }
      if (client::parsed(tag) || TextParser::skips(tag)) {
        if (!TextParser::skips(tag)) lines.emplace_back(line);
      } else {
        unknown.emplace_back(tag);
      }
    }
    if (!lines.empty()) pending_.push_back(std::move(lines));
    if (named_) parse_pending();
  }

  // A request's JSON. Returns whether it asks for a choice (not a wait, not
  // "null"); `req`, `fields`, `options` and `n` then describe it.
  bool request(std::string_view json_text) {
    if (json_text == "null" || json_text.empty()) return false;
    const json::Value v = json::parse(json_text);
    const json::Value& side = v["side"];
    const std::string& id = client::text(side, "id");
    if (id != "p1" && id != "p2") client::bad_request("bad side id");
    const uint8_t p = id == "p1" ? 0 : 1;
    if (!named_) {
      viewer_ = p;
      parser_.emplace(names_, Stream::View, p);
      name_side(side);
      named_ = true;
      parse_pending();
    } else if (p != viewer_) {
      client::bad_request("side changed");
    }
    rqid = v["rqid"].type == json::Value::Number ? int(client::integer(v, "rqid")) : -1;
    req = read(v);
    encode(req, fields.data());
    n = req.kind == RequestData::Wait ? 0 : showdown_choices(req, options.data());
    bk_.request(req);
    return req.kind != RequestData::Wait;
  }

  // The text to send for choice `i` of the last request.
  std::string choice(uint8_t i) const {
    if (i >= n) throw std::out_of_range("choice");
    const pkmn_choice c = options[i];
    std::string s = "/choose ";
    switch (choice_type(c)) {
    case PKMN_CHOICE_MOVE: s += "move " + std::to_string(choice_data(c)); break;
    case PKMN_CHOICE_SWITCH: s += "switch " + std::to_string(choice_data(c)); break;
    default: s += "pass"; break;
    }
    return rqid >= 0 ? s + "|" + std::to_string(rqid) : s;
  }

  // Choice `i` is sent: the bookkeeping hears it, as BattleData's choose.
  // Returns its text.
  std::string choose(uint8_t i) {
    std::string s = choice(i);
    bk_.chose(bk_.selected(req, options[i]));
    return s;
  }

  uint8_t viewer() const { return viewer_; }

  // Whether choice `i` of the last request names only what the log has
  // shown the other player (birch's, for redacted chat): a switch to a
  // Pokémon that has appeared, or a move the active Pokémon has used. A
  // one-entry request counts as shown.
  bool revealed(uint8_t i) const {
    if (i >= n) throw std::out_of_range("choice");
    const pkmn_choice c = options[i];
    if (choice_type(c) == PKMN_CHOICE_SWITCH) return appeared_[party_[choice_data(c) - 1]];
    if (choice_type(c) != PKMN_CHOICE_MOVE || req.active.locked) return true;
    const PKMN::Data::Move m = req.active.moves[choice_data(c) - 1].id;
    for (const PKMN::Data::Move u : used_[party_[0]]) {
      if (u == m) return true;
    }
    return false;
  }

  // The bookkeeping state now, STATE_FIELDS int16s.
  std::array<int16_t, STATE_FIELDS> state() const {
    std::array<int16_t, STATE_FIELDS> out{};
    encode_state(bk_, out.data());
    return out;
  }
  const Names& names() const { return names_; }
  // For the history sampler: the bookkeeping now, each update's messages and
  // the bookkeeping after it.
  const Bookkeeping& bookkeeping() const { return bk_; }
  const std::vector<std::vector<Obs>>& history() const { return history_; }
  const std::vector<Bookkeeping>& after() const { return after_; }

  // The tokens since the driver last took them: every update parsed since.
  std::vector<Token> tokens;
  // Tags neither read nor known to carry nothing, in order.
  std::vector<std::string> unknown;

  RequestData req{};
  int rqid = -1;
  std::array<int16_t, REQUEST_FIELDS> fields{};
  std::array<pkmn_choice, MAX_CHOICES> options{};
  uint8_t n = 0;

 private:
  Names names_;
  std::optional<TextParser> parser_; // from the first request, which names our side
  uint8_t viewer_ = 0;
  bool named_ = false;
  bool started_ = false; // whether `start` has been fed
  std::vector<std::vector<std::string>> pending_;
  std::array<uint8_t, 2> teamsize_{6, 6};
  Bookkeeping bk_;
  std::vector<std::vector<Obs>> history_;
  std::vector<Bookkeeping> after_;
  // Our side's max HP by party position, for "0 fnt".
  std::array<uint16_t, 6> max_hp_{};
  // The last request's Pokémon's party positions, by request position.
  std::array<uint8_t, 6> party_{};
  // By party position: switched in yet, and the moves used.
  std::array<bool, 6> appeared_{};
  std::array<std::vector<PKMN::Data::Move>, 6> used_{};

  // Our nicknames, by party position: the first request's order with our
  // switch-ins held so far undone. A request lists the side in its current
  // order, and a switch-in swaps the incoming Pokémon with the active one
  // (sim/side.ts), so on a rejoin, whose first request follows the whole
  // log, the first request's order isn't the team's. Then our side's
  // Bookkeeping, from the sets in party order.
  void name_side(const json::Value& side) {
    const json::Value& pokemon = side["pokemon"];
    if (pokemon.type != json::Value::Array || pokemon.size() == 0 || pokemon.size() > 6) {
      client::bad_request("bad side");
    }
    const size_t size = pokemon.size();
    std::array<size_t, 6> at{}; // party position -> request position
    std::array<std::string, 6>& order = names_.nickname[viewer_];
    for (size_t i = 0; i < size; i++) {
      order[i] = std::string(client::nickname(client::text(pokemon[i], "ident")));
      at[i] = i;
    }
    const std::string ours = std::string(viewer_ == 0 ? "p1a: " : "p2a: ");
    std::vector<std::string> ins;
    std::vector<uint16_t> max; // each switch-in's max HP, exact on our side
    for (const std::vector<std::string>& lines : pending_) {
      for (const std::string& l : lines) {
        const std::vector<std::string_view> p = client::fields(l);
        if ((p[1] == "switch" || p[1] == "drag") && p.size() > 4 && p[2].starts_with(ours)) {
          ins.emplace_back(p[2].substr(ours.size()));
          const std::string_view hp = p[4].substr(0, p[4].find(' '));
          const size_t slash = hp.find('/');
          max.push_back(slash == std::string_view::npos ? 0 : client::number(hp.substr(slash + 1)));
        }
      }
    }
    for (size_t k = ins.size(); k-- > 1;) {
      size_t j = 1;
      while (j < size && order[j] != ins[k - 1]) j++;
      if (order[0] != ins[k] || j == size) client::bad_request("switches don't match the side");
      std::swap(order[0], order[j]);
      std::swap(at[0], at[j]);
    }
    for (size_t k = 0; k < ins.size(); k++) {
      for (size_t i = 0; i < size; i++) {
        if (order[i] == ins[k] && max[k]) max_hp_[i] = max[k];
      }
    }
    Team team(size);
    for (size_t i = 0; i < size; i++) {
      const json::Value& x = pokemon[at[i]];
      Set& set = team[i];
      details(client::text(x, "details"), set.species, set.level);
      set.moves = {};
      const json::Value& moves = x["moves"];
      if (moves.type != json::Value::Array || moves.size() > 4) client::bad_request("bad moves");
      for (size_t k = 0; k < moves.size(); k++) {
        if (moves[k].type != json::Value::String) client::bad_request("bad moves");
        set.moves[k] = client::move_id(moves[k].string);
      }
    }
    teamsize_[viewer_] = uint8_t(size);
    bk_ = Bookkeeping(viewer_, teamsize_, PKMN::Init::init_side(team));
  }

  // DETAILS: species[, Llevel][, M|F][, shiny]
  static void details(std::string_view d, PKMN::Data::Species& species, uint8_t& level) {
    const std::vector<std::string_view> parts = [&] {
      std::vector<std::string_view> out;
      for (size_t start = 0;;) {
        const size_t c = d.find(", ", start);
        out.push_back(d.substr(start, c == std::string_view::npos ? c : c - start));
        if (c == std::string_view::npos) return out;
        start = c + 2;
      }
    }();
    const auto s = ataraxos::detail::species(parts[0]);
    if (!s) client::bad_request("unknown species \"" + std::string(d) + "\"");
    species = *s;
    level = 100;
    for (size_t k = 1; k < parts.size(); k++) {
      if (parts[k].size() > 1 && parts[k][0] == 'L') level = uint8_t(client::number(parts[k].substr(1)));
    }
  }

  void parse_pending() {
    for (const std::vector<std::string>& lines : pending_) {
      const std::vector<uint8_t> bytes = parser_->parse(lines);
      tokenize(bytes, viewer_, tokens);
      std::vector<Obs> obs = split(bytes);
      for (const Obs& o : obs) {
        if (o.type == ArgType::Switch && ident_player(o.data[0]) == viewer_) appeared_[ident_slot(o.data[0]) - 1] = true;
        if (o.type == ArgType::Move && ident_player(o.data[0]) == viewer_) {
          used_[ident_slot(o.data[0]) - 1].push_back(PKMN::Data::Move(o.data[1]));
        }
      }
      bk_.update(obs, history_);
      history_.push_back(std::move(obs));
      after_.push_back(bk_);
    }
    pending_.clear();
  }

  RequestData read(const json::Value& v) {
    RequestData r{};
    r.update = client::flag(v, "update");
    if (client::flag(v, "wait")) {
      r.kind = RequestData::Wait;
    } else if (v["forceSwitch"].type == json::Value::Array) {
      r.kind = RequestData::Switch;
    } else if (v["active"].type == json::Value::Array) {
      r.kind = RequestData::Move_;
    } else {
      client::bad_request("not a move, switch or wait request");
    }

    const json::Value& pokemon = v["side"]["pokemon"];
    if (pokemon.type != json::Value::Array || pokemon.size() == 0 || pokemon.size() > 6) {
      client::bad_request("bad side");
    }
    const std::array<std::string, 6>& ours = names_.nickname[viewer_];
    for (size_t i = 0; i < pokemon.size(); i++) {
      const json::Value& x = pokemon[i];
      RequestData::Pokemon& q = r.side[r.size++];
      const std::string_view nick = client::nickname(client::text(x, "ident"));
      size_t at = 0;
      while (at < 6 && ours[at] != nick) at++;
      if (at == 6) client::bad_request("unknown Pokémon \"" + std::string(nick) + "\"");
      party_[i] = uint8_t(at);
      details(client::text(x, "details"), q.species, q.level);
      uint16_t max = 0;
      client::condition(client::text(x, "condition"), q.hp, max, q.status);
      if (max) max_hp_[at] = max;
      q.max_hp = max_hp_[at];
      if (!q.max_hp) client::bad_request("no max HP for \"" + std::string(nick) + "\"");
      q.active = client::flag(x, "active");
      const json::Value& stats = x["stats"];
      q.stats.hp = q.max_hp;
      q.stats.atk = client::integer(stats, "atk");
      q.stats.def = client::integer(stats, "def");
      q.stats.spe = client::integer(stats, "spe");
      q.stats.spc = client::integer(stats, "spa");
      const json::Value& moves = x["moves"];
      if (moves.type != json::Value::Array || moves.size() > 4) client::bad_request("bad moves");
      for (size_t k = 0; k < moves.size(); k++) {
        if (moves[k].type != json::Value::String) client::bad_request("bad moves");
        q.moves[k] = client::move_id(moves[k].string);
      }
    }
    if (r.kind != RequestData::Move_) return r;

    const json::Value& a = v["active"][0];
    const json::Value& moves = a["moves"];
    if (moves.type != json::Value::Array || moves.size() == 0 || moves.size() > 4) {
      client::bad_request("bad active moves");
    }
    RequestData::Active& act = r.active;
    // one entry and no PP: a lock, Fight or Struggle
    if (moves.size() == 1 && moves[0]["pp"].is_null()) {
      act.locked = true;
      act.trapped = client::flag(a, "trapped");
      const std::string& id = client::text(moves[0], "id");
      if (id == "recharge") act.recharge = true;
      else if (id != "fight") act.moves[0].id = client::move_id(id);
      return r;
    }
    act.trapped = client::flag(a, "trapped");
    act.maybe_locked = client::flag(a, "maybeLocked");
    for (size_t k = 0; k < moves.size(); k++) {
      const json::Value& m = moves[k];
      RequestData::ActiveMove& out = act.moves[k];
      out.id = client::move_id(client::text(m, "id"));
      out.pp = uint8_t(client::integer(m, "pp"));
      out.maxpp = uint8_t(client::integer(m, "maxpp"));
      out.disabled = client::flag(m, "disabled");
    }
    return r;
  }
};

} // namespace ataraxos
