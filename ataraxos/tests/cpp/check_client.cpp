// Checks src/client.hpp against training's input: each player's Client,
// fed what Showdown would send it, must give the network the same tokens,
// request fields, choices and bookkeeping state BattleData (and so the env)
// gives that player (birch's check_client for Gen 1).
//
// Per update, each player's message is its view (view.hpp) printed by the
// Formatter's Stream::View with the other side numbered as the view numbers
// it, wrapped as Showdown's room sends it: the room's header lines before
// the first update, a blank line and a timestamp, `upkeep` before `turn`, a
// chat line now and then. Then each player's request, written as
// Showdown's JSON (sim/side.ts getRequestData, sim/pokemon.ts
// getMoveRequestData: the locks, Fight, Struggle and maybeLocked as
// request_data.hpp has them). Choices are uniform over Showdown's, each
// sent to the Client too; a battle stops at turn 300.
//
// The JSON writer is this file's reading of Showdown, so the reader is
// checked against it here, and against Showdown's own requests by
// `check_client showdown DUMP` (tests/showdown/play.cjs's dump: every
// request parses, and the Client's choices are the ones Showdown takes).
//
//   check_client [GAMES] [SEED]
//   check_client showdown DUMP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "battle_data.hpp"
#include "client.hpp"
#include "format.hpp"
#include "names.hpp"
#include "teams.hpp"
#include "view.hpp"

using namespace ataraxos;

namespace {

std::string quote(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

std::string status_text(uint8_t s) {
  using S = PKMN::Data::Status;
  if (s & 7) return " slp";
  if (s == uint8_t(S::Toxic)) return " tox";
  if (s == uint8_t(S::Poison)) return " psn";
  if (s == uint8_t(S::Burn)) return " brn";
  if (s == uint8_t(S::Freeze)) return " frz";
  if (s == uint8_t(S::Paralysis)) return " par";
  return "";
}

std::string move_name(PKMN::Data::Move m) { return std::string(names::MOVES[uint8_t(m)]); }
std::string move_id(PKMN::Data::Move m) { return detail::key(names::MOVES[uint8_t(m)]); }

// `r` as Showdown's JSON for `player`, whose Pokémon in slot s are named by
// `names` at party position side.order[s - 1].
std::string to_json(const RequestData& r, uint16_t rqid, const PKMN::Side& side, const Names& names,
                    uint8_t player) {
  std::string j = "{";
  if (r.kind == RequestData::Move_) {
    const RequestData::Active& a = r.active;
    j += "\"active\":[{\"moves\":[";
    if (a.locked) {
      const PKMN::Data::Move m = a.moves[0].id;
      if (a.recharge) j += "{\"move\":\"Recharge\",\"id\":\"recharge\"}";
      else if (m == PKMN::Data::Move::None) j += "{\"move\":\"Fight\",\"id\":\"fight\"}";
      else if (m == PKMN::Data::Move::Struggle)
        j += "{\"move\":\"Struggle\",\"id\":\"struggle\",\"target\":\"randomNormal\",\"disabled\":false}";
      else j += "{\"move\":" + quote(move_name(m)) + ",\"id\":" + quote(move_id(m)) + "}";
    } else {
      bool first = true;
      for (const RequestData::ActiveMove& m : a.moves) {
        if (m.id == PKMN::Data::Move::None) continue;
        if (!first) j += ",";
        first = false;
        j += "{\"move\":" + quote(move_name(m.id)) + ",\"id\":" + quote(move_id(m.id)) +
             ",\"pp\":" + std::to_string(m.pp) + ",\"maxpp\":" + std::to_string(m.maxpp) +
             ",\"target\":\"normal\",\"disabled\":" + (m.disabled ? "true" : "false") + "}";
      }
    }
    j += "]";
    if (a.trapped) j += ",\"trapped\":true";
    if (a.maybe_locked) j += ",\"maybeLocked\":true";
    j += "}],";
  } else if (r.kind == RequestData::Switch) {
    j += "\"forceSwitch\":[true],";
  } else {
    j += "\"wait\":true,";
  }
  j += "\"side\":{\"name\":" + quote(names.player[player]) + ",\"id\":\"p" + std::to_string(player + 1) +
       "\",\"pokemon\":[";
  for (uint8_t i = 0; i < r.size; i++) {
    const RequestData::Pokemon& q = r.side[i];
    if (i) j += ",";
    std::string details(names::SPECIES[uint8_t(q.species)]);
    if (q.level != 100) details += ", L" + std::to_string(q.level);
    const std::string condition =
        q.hp == 0 ? "0 fnt" : std::to_string(q.hp) + "/" + std::to_string(q.max_hp) + status_text(q.status);
    j += "{\"ident\":" + quote("p" + std::to_string(player + 1) + ": " + names.nickname[player][side.order[i] - 1]) +
         ",\"details\":" + quote(details) + ",\"condition\":" + quote(condition) +
         ",\"active\":" + (q.active ? "true" : "false") + ",\"stats\":{\"atk\":" + std::to_string(q.stats.atk) +
         ",\"def\":" + std::to_string(q.stats.def) + ",\"spa\":" + std::to_string(q.stats.spc) +
         ",\"spd\":" + std::to_string(q.stats.spc) + ",\"spe\":" + std::to_string(q.stats.spe) + "},\"moves\":[";
    bool first = true;
    for (uint8_t k = 0; k < 4; k++) {
      if (q.moves[k] == PKMN::Data::Move::None) continue;
      if (!first) j += ",";
      first = false;
      j += quote(move_id(q.moves[k]));
    }
    j += "],\"baseAbility\":\"noability\",\"item\":\"\",\"pokeball\":\"pokeball\"}";
  }
  j += "]}";
  if (r.kind != RequestData::Wait) j += ",\"noCancel\":true";
  j += ",\"rqid\":" + std::to_string(rqid) + "}";
  return j;
}

struct Counts {
  uint64_t games = 0, decisions = 0, forced = 0, tokens = 0, capped = 0;
  uint64_t fight = 0, semi = 0, hidden = 0;
  std::map<std::string, uint64_t> failures;
  std::map<std::string, uint64_t> unknown;
  int printed = 0;
};

struct Game {
  BattleData d;
  Names names;
  View views[2] = {View{0}, View{1}};
  Names viewed[2];
  Client clients[2];
  uint16_t rqid = 0;
  bool started = false;
  bool failed = false;
  std::mt19937_64* rng;
  Counts* n;
  uint64_t game;

  void fail(const std::string& what, const std::string& detail = "") {
    failed = true;
    n->failures[what]++;
    if (n->printed < 12 && n->failures[what] <= 2) {
      n->printed++;
      std::printf("FAIL %s (game %llu, turn %u)\n%s", what.c_str(), (unsigned long long)game, d.turn(),
                  detail.c_str());
    }
  }

  int team_size(uint8_t p) const {
    int k = 0;
    for (const PKMN::Pokemon& x : d.state().sides[p].pokemon) k += x.species != PKMN::Data::Species::None;
    return k;
  }

  // Each player's message for the update in d.buffer, then its request.
  void send_update() {
    size_t used = 0;
    while (d.buffer[used] != 0) used += 1 + message_size(ArgType(d.buffer[used]), &d.buffer[used + 1]);
    for (uint8_t p = 0; p < 2; p++) {
      std::vector<uint8_t> copy(d.buffer.begin(), d.buffer.begin() + std::ptrdiff_t(used + 1));
      views[p].apply(copy);
      const uint8_t foe = 1 - p;
      viewed[p].player = names.player;
      viewed[p].nickname[p] = names.nickname[p];
      for (uint8_t slot = 1; slot <= 6; slot++) {
        if (views[p].number[slot]) viewed[p].nickname[foe][views[p].number[slot] - 1] = names.nickname[foe][slot - 1];
      }
      const Formatter formatter(viewed[p], Stream::View);
      std::string msg;
      if (!started) {
        msg += "|init|battle\n|title|" + names.player[0] + " vs. " + names.player[1] + "\n|j|☆" + names.player[p] +
               "\n|t:|1700000000\n|gametype|singles\n|player|p1|" + names.player[0] + "|1|\n|player|p2|" +
               names.player[1] + "|2|\n|teamsize|p1|" + std::to_string(team_size(0)) + "\n|teamsize|p2|" +
               std::to_string(team_size(1)) +
               "\n|gen|1\n|tier|[Gen 1] OU\n|rule|Sleep Clause Mod: Limit one foe put to sleep\n|\n|t:|1700000000\n|start\n";
      } else {
        msg += "|\n|t:|1700000000\n";
      }
      if ((*rng)() % 16 == 0) msg += "|c|☆" + names.player[foe] + "|gl hf\n";
      for (const std::string& l : formatter.format(copy)) {
        if (l.starts_with("|turn|")) msg += "|upkeep\n";
        msg += l + "\n";
      }
      if ((*rng)() % 64 == 0) msg += "|inactive|Battle timer is ON\n";
      deliver(p, msg);
    }
    started = true;
    if (d.over()) return;
    rqid++;
    for (uint8_t p = 0; p < 2; p++) ask(p);
  }

  void deliver(uint8_t p, const std::string& msg) {
    if (failed) return;
    try {
      clients[p].feed(msg);
    } catch (const std::exception& e) {
      fail(std::string("feed: ") + e.what());
    }
  }

  void ask(uint8_t p) {
    if (failed) return;
    const std::string json = to_json(d.request[p], rqid, d.state().sides[p], names, p);
    try {
      clients[p].request(json);
    } catch (const std::exception& e) {
      fail(std::string("request: ") + e.what(), "  " + json + "\n");
    }
  }

  // Compares p's Client with BattleData at a decision; `take`: the env's
  // network sees it (two or more choices), so the state and tokens are
  // compared and taken.
  void compare(uint8_t p, bool take) {
    if (failed) return;
    Client& c = clients[p];
    std::array<int16_t, REQUEST_FIELDS> fields;
    encode(d.request[p], fields.data());
    if (fields != c.fields) {
      std::string detail = "  " + to_json(d.request[p], rqid, d.state().sides[p], names, p) + "\n  fields:";
      for (size_t f = 0; f < fields.size(); f++) {
        if (fields[f] != c.fields[f]) {
          detail += " [" + std::to_string(f) + "] " + std::to_string(fields[f]) + " vs " + std::to_string(c.fields[f]);
        }
      }
      return fail("request fields", detail + "\n");
    }
    const uint8_t k = d.n_option[p];
    if (k != c.n || !std::equal(d.option[p].begin(), d.option[p].begin() + k, c.options.begin())) return fail("choices");
    for (uint8_t x = 0; x < k; x++) {
      const pkmn_choice ch = d.option[p][x];
      const std::string want = "/choose " +
                               (choice_type(ch) == PKMN_CHOICE_MOVE ? "move " + std::to_string(choice_data(ch))
                                                                     : "switch " + std::to_string(choice_data(ch))) +
                               "|" + std::to_string(rqid);
      if (c.choice(x) != want) return fail("choice text", "  " + c.choice(x) + " vs " + want + "\n");
    }
    if (!take) return;
    std::array<int16_t, STATE_FIELDS> state;
    encode_state(d.bk[p], state.data());
    const std::array<int16_t, STATE_FIELDS> got = c.state();
    if (state != got) {
      std::string detail = "  fields:";
      for (size_t f = 0; f < state.size(); f++) {
        if (state[f] != got[f]) {
          detail += " [" + std::to_string(f) + "] " + std::to_string(state[f]) + " vs " + std::to_string(got[f]);
        }
      }
      return fail("state", detail + "\n");
    }
    const std::vector<Token>& want = d.chunk[p];
    n->tokens += want.size();
    if (want.size() != c.tokens.size() ||
        std::memcmp(want.data(), c.tokens.data(), want.size() * sizeof(Token)) != 0) {
      size_t at = 0;
      while (at < want.size() && at < c.tokens.size() && std::memcmp(&want[at], &c.tokens[at], sizeof(Token)) == 0) {
        at++;
      }
      std::string detail = "  " + std::to_string(want.size()) + " vs " + std::to_string(c.tokens.size()) +
                           " tokens, first difference at " + std::to_string(at) + "\n";
      const auto row = [](const Token& t) {
        std::string r;
        for (int16_t x : t.c) r += " " + std::to_string(x);
        return r;
      };
      for (size_t q = at; q < at + 3; q++) {
        if (q < want.size()) detail += "  want" + row(want[q]) + "\n";
        if (q < c.tokens.size()) detail += "  got " + row(c.tokens[q]) + "\n";
      }
      std::string what = "tokens";
      if (at < want.size()) what += " at kind " + std::to_string(want[at].c[KIND]);
      return fail(what, detail);
    }
    d.chunk[p].clear();
    c.tokens.clear();
  }

  void play(const pkmn_gen1_battle& battle) {
    d.track = true;
    d.reset(battle);
    send_update();
    while (!d.over() && !failed) {
      if (d.turn() > 300) {
        n->capped++;
        break;
      }
      bool acted = false;
      for (uint8_t p = 0; p < 2; p++) {
        if (failed || d.over() || !d.needs_choice(p)) continue;
        acted = true;
        const uint8_t k = d.n_option[p];
        const RequestData::Active& a = d.request[p].active;
        if (d.request[p].kind == RequestData::Move_) {
          n->fight += a.locked && !a.trapped && a.moves[0].id == PKMN::Data::Move::None;
          n->semi += a.locked && !a.trapped && a.moves[0].id != PKMN::Data::Move::None &&
                     a.moves[0].id != PKMN::Data::Move::Struggle;
          n->hidden += a.maybe_locked;
        }
        compare(p, k > 1);
        if (k > 1) n->decisions++;
        else n->forced++;
        const uint8_t x = std::uniform_int_distribution<int>(0, k - 1)(*rng);
        if (!failed) clients[p].choose(x);
        switch (d.choose(p, d.option[p][x])) {
        case BattleData::Held: break;
        case BattleData::Advanced:
        case BattleData::Ended: send_update(); break;
        }
        break;
      }
      if (!acted) break;
    }
    for (const Client& c : clients) {
      for (const std::string& t : c.unknown) n->unknown[t]++;
    }
    n->games++;
  }
};

// --- check_client showdown DUMP: Showdown's own requests.

std::string choice_key(std::string s) {
  // "/choose move 2|7" -> "move 2"
  if (s.starts_with("/choose ")) s = s.substr(8);
  return s.substr(0, s.find('|'));
}

int showdown(const char* path) {
  std::ifstream in(path);
  if (!in) {
    std::printf("can't read %s\n", path);
    return 2;
  }
  std::string line;
  std::unique_ptr<Client> clients[2];
  bool dead[2] = {false, false};
  std::vector<std::string> text;
  int channel = -1;
  uint64_t battles = 0, requests = 0, choices = 0, failed = 0, printed = 0;
  std::map<std::string, uint64_t> failures, unknown, shapes;
  const auto fail = [&](int side, const std::string& what, const std::string& detail) {
    dead[side] = true;
    failures[what]++;
    failed++;
    if (printed++ < 10) std::printf("FAIL p%d %s (battle %llu)\n  %s\n", side + 1, what.c_str(), (unsigned long long)battles, detail.c_str());
  };
  while (std::getline(in, line)) {
    if (channel >= 0) {
      if (line == ".") {
        if (channel > 0 && !dead[channel - 1]) {
          std::string msg;
          for (const std::string& l : text) msg += l + "\n";
          try {
            clients[channel - 1]->feed(msg);
          } catch (const std::exception& e) {
            fail(channel - 1, "feed", e.what());
          }
        }
        text.clear();
        channel = -1;
      } else {
        text.push_back(line);
      }
      continue;
    }
    if (line.starts_with("B ")) {
      for (int s = 0; s < 2; s++) {
        if (clients[s]) {
          for (const std::string& t : clients[s]->unknown) unknown[t]++;
        }
        clients[s] = std::make_unique<Client>();
        dead[s] = false;
      }
      battles++;
    } else if (line.starts_with("U ")) {
      channel = std::atoi(line.c_str() + 2);
    } else if (line.starts_with("R ")) {
      // R SIDE JSON: a request
      const int s = line[2] - '0';
      if (dead[s]) continue;
      requests++;
      try {
        clients[s]->request(line.substr(4));
        const RequestData& q = clients[s]->req;
        if (q.kind == RequestData::Move_) {
          const RequestData::Active& a = q.active;
          shapes[a.trapped ? "hard lock" : !a.locked ? (a.maybe_locked ? "maybeLocked" : "move list")
                 : a.moves[0].id == PKMN::Data::Move::None ? "Fight"
                 : a.moves[0].id == PKMN::Data::Move::Struggle ? "Struggle" : "semi-lock"]++;
        }
      } catch (const std::exception& e) {
        fail(s, "request", std::string(e.what()) + "\n  " + line.substr(4));
      }
    } else if (line.starts_with("C ")) {
      // C SIDE CHOICE\tCHOICE...: the choices Showdown takes; X SIDE I: the one made
      const int s = line[2] - '0';
      if (dead[s]) continue;
      std::set<std::string> want, got;
      size_t start = 4;
      while (start <= line.size()) {
        size_t tab = line.find('\t', start);
        if (tab == std::string::npos) tab = line.size();
        if (tab > start) want.insert(line.substr(start, tab - start));
        start = tab + 1;
      }
      Client& c = *clients[s];
      for (uint8_t i = 0; i < c.n; i++) got.insert(choice_key(c.choice(i)));
      choices++;
      if (want != got) {
        std::string w, g;
        for (const std::string& x : want) w += " [" + x + "]";
        for (const std::string& x : got) g += " [" + x + "]";
        fail(s, "choices", "Showdown's" + w + "\n  the client's" + g);
      }
    } else if (line.starts_with("X ")) {
      const int s = line[2] - '0';
      if (dead[s]) continue;
      const std::string made = line.substr(4);
      Client& c = *clients[s];
      for (uint8_t i = 0; i < c.n; i++) {
        if (choice_key(c.choice(i)) == made) c.choose(i);
      }
    }
  }
  for (int s = 0; s < 2; s++) {
    if (clients[s]) {
      for (const std::string& t : clients[s]->unknown) unknown[t]++;
    }
  }
  std::printf("%llu battles, %llu requests, %llu choice sets compared\n", (unsigned long long)battles,
              (unsigned long long)requests, (unsigned long long)choices);
  for (const auto& [shape, count] : shapes) std::printf("  move request: %-14s %llu\n", shape.c_str(), (unsigned long long)count);
  for (const auto& [tag, count] : unknown) std::printf("  unknown tag %-20s %llu\n", tag.c_str(), (unsigned long long)count);
  for (const auto& [what, count] : failures) std::printf("  %-30s %llu\n", what.c_str(), (unsigned long long)count);
  std::printf("%llu failures\n", (unsigned long long)failed);
  return failed ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc > 2 && std::string(argv[1]) == "showdown") return showdown(argv[2]);
  const uint64_t games = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000;
  const uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;

  std::mt19937_64 rng(seed);
  Counts n;
  for (uint64_t g = 0; g < games; g++) {
    const Team t1 = random_team(rng), t2 = random_team(rng);
    auto game = std::make_unique<Game>();
    game->rng = &rng;
    game->n = &n;
    game->game = g;
    game->names.player = {"Alice Smith", "bob"};
    const Team* teams[2] = {&t1, &t2};
    for (size_t s = 0; s < 2; s++) {
      std::map<std::string, int> seen;
      for (size_t i = 0; i < teams[s]->size(); i++) {
        std::string nick(names::SPECIES[uint8_t((*teams[s])[i].species)]);
        const int k = ++seen[nick];
        if (k > 1) nick += " " + std::to_string(k);
        game->names.nickname[s][i] = nick;
      }
    }
    game->play(init_battle(rng(), t1, t2));
  }
  std::printf("%llu games (%llu capped at turn 300), %llu decisions, %llu forced, %llu tokens compared; "
              "%llu Fight, %llu semi-locks, %llu maybeLocked\n",
              (unsigned long long)n.games, (unsigned long long)n.capped, (unsigned long long)n.decisions,
              (unsigned long long)n.forced, (unsigned long long)n.tokens, (unsigned long long)n.fight,
              (unsigned long long)n.semi, (unsigned long long)n.hidden);
  for (const auto& [tag, count] : n.unknown) std::printf("  unknown tag %-20s %llu\n", tag.c_str(), (unsigned long long)count);
  uint64_t total = 0;
  for (const auto& [what, count] : n.failures) {
    std::printf("  %-50s %llu\n", what.c_str(), (unsigned long long)count);
    total += count;
  }
  std::printf("%llu games failed\n", (unsigned long long)total);
  return total ? 1 : 0;
}
