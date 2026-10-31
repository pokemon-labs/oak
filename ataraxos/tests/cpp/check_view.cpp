// Checks the view, the tokens and the requests (src/view.hpp, tokens.hpp,
// request_data.hpp) on random self-play: truly random teams (teams.hpp),
// uniform choices over libpkmn's, every update through BattleData.
//
// - each update's omniscient bytes walk to their null by message_size, and
//   split's Obs give them back exactly;
// - a view keeps every message's size and changes only the other side's HP
//   (to percent/100, ceil, 99 when not full), the status bytes (sleep as 1)
//   and the other side's slots; the other side's numbers go 1, 2, ... in
//   order of first sight, each party slot keeping its number;
// - tokenize gives one token per message, KIND nonzero, idents in range;
// - every request with two or more choices has them all legal for its
//   kind, and a locked request has exactly one move and no switch.
//
//   check_view [GAMES] [SEED]

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "battle_data.hpp"
#include "teams.hpp"

using namespace ataraxos;

namespace {

size_t failures = 0;

void fail(const char* what, size_t game) {
  if (failures++ < 20) std::printf("game %zu: %s\n", game, what);
}

uint16_t u16(const uint8_t* p) { return uint16_t(p[0] | p[1] << 8); }

// The (hp, max, status) offsets after the ArgType, or -1.
int health_at(ArgType t) {
  switch (t) {
  case ArgType::Switch: return 3;
  case ArgType::Damage:
  case ArgType::Heal:
  case ArgType::SetHP: return 1;
  default: return -1;
  }
}

} // namespace

int main(int argc, char** argv) {
  const size_t games = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 2000;
  std::mt19937_64 rng(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1);
  size_t updates = 0, messages = 0;
  for (size_t g = 0; g < games; g++) {
    const Team t1 = random_team(rng), t2 = random_team(rng);
    pkmn_gen1_battle b = init_battle(rng(), t1, t2);
    std::array<uint8_t, LOG_SIZE> buf{};
    pkmn_gen1_battle_options options{};
    View views[2] = {View{0}, View{1}};
    std::array<std::array<uint8_t, 7>, 2> seen_number{};
    pkmn_choice c1 = 0, c2 = 0;
    for (int turn = 0; turn < 1000; turn++) {
      pkmn_gen1_log_options log{buf.data(), buf.size()};
      pkmn_gen1_battle_options_set(&options, &log, nullptr, nullptr);
      const pkmn_result r = pkmn_gen1_battle_update(&b, c1, c2, &options);
      updates++;
      size_t used = 0;
      while (buf[used] != 0) {
        used += 1 + message_size(ArgType(buf[used]), &buf[used + 1]);
        messages++;
        if (used >= buf.size()) {
          fail("a message runs past the buffer", g);
          break;
        }
      }
      const std::vector<Obs> obs = split({buf.data(), used + 1});
      {
        std::vector<uint8_t> back;
        for (const Obs& o : obs) {
          back.push_back(uint8_t(o.type));
          back.insert(back.end(), o.data, o.data + o.size());
        }
        if (back.size() != used || !std::equal(back.begin(), back.end(), buf.begin())) {
          fail("split doesn't give the bytes back", g);
        }
      }
      for (uint8_t p = 0; p < 2; p++) {
        std::array<uint8_t, LOG_SIZE> v = buf;
        const size_t n = views[p].apply({v.data(), used + 1});
        if (n != used) fail("a view changed the length", g);
        size_t i = 0;
        while (i < used) {
          const ArgType t = ArgType(buf[i]);
          const size_t size = message_size(t, &buf[i + 1]);
          if (ArgType(v[i]) != t || message_size(t, &v[i + 1]) != size) fail("a view changed a message's kind or size", g);
          const uint8_t* o = &buf[i + 1];
          const uint8_t* w = &v[i + 1];
          const int h = health_at(t);
          if (h >= 0) {
            const bool ours = ident_player(o[0]) == p;
            const uint16_t hp = u16(o + h), max = u16(o + h + 2);
            const uint16_t vhp = u16(w + h), vmax = u16(w + h + 2);
            if (ours ? (vhp != hp || vmax != max) : (vmax != (max ? 100 : 0) || vhp != view_percent(hp, max))) {
              fail("HP not as Showdown shows it", g);
            }
            if (vhp > vmax) fail("HP over max in a view", g);
            if (w[h + 4] != view_status(o[h + 4])) fail("status byte not as the text shows it", g);
          }
          if (t == ArgType::Switch) {
            const uint8_t id = o[0], vid = w[0];
            if (ident_player(id) == p) {
              if (vid != id) fail("our own ident renumbered", g);
            } else {
              uint8_t& want = seen_number[p][ident_slot(id)];
              if (!want) {
                uint8_t next = 1;
                for (uint8_t x : seen_number[p]) next += x != 0;
                want = next;
              }
              if (ident_slot(vid) != want || ident_player(vid) != ident_player(id)) {
                fail("the foe not numbered by first sight", g);
              }
            }
          }
          i += 1 + size;
        }
        std::vector<Token> tokens;
        tokenize({v.data(), n}, p, tokens);
        if (tokens.size() != obs.size()) fail("not one token per message", g);
        for (const Token& tk : tokens) {
          if (tk.c[KIND] <= 0 || tk.c[KIND] >= int16_t(ARG_TYPES)) fail("a token's KIND out of range", g);
          for (Column col : {ACTOR, OTHER}) {
            if (tk.c[col] < 0 || tk.c[col] > 16) fail("an ident out of range", g);
          }
        }
      }
      if (pkmn_result_type(r)) break;
      pkmn_choice chosen[2];
      for (uint8_t p = 0; p < 2; p++) {
        std::array<pkmn_choice, MAX_CHOICES> opts{};
        const pkmn_choice_kind kind = p ? pkmn_result_p2(r) : pkmn_result_p1(r);
        const size_t n = pkmn_gen1_battle_choices(&b, p ? PKMN_PLAYER_P2 : PKMN_PLAYER_P1, kind, opts.data(),
                                                  opts.size());
        const RequestData q = request_data(PKMN::view(b), p, kind, {opts.data(), n});
        uint8_t moves = 0, switches = 0;
        for (size_t k = 0; k < n; k++) {
          const uint8_t ty = choice_type(opts[k]), d = choice_data(opts[k]);
          moves += ty == PKMN_CHOICE_MOVE;
          switches += ty == PKMN_CHOICE_SWITCH;
          if (n >= 2) {
            if (q.kind == RequestData::Move_ && ty == PKMN_CHOICE_PASS) fail("a pass in a move request", g);
            if (q.kind == RequestData::Switch && ty != PKMN_CHOICE_SWITCH) fail("a non-switch in a switch request", g);
            if (ty == PKMN_CHOICE_SWITCH && (d < 2 || d > q.size)) fail("a switch past the side", g);
            if (ty == PKMN_CHOICE_MOVE && d > 4) fail("a move past slot 4", g);
          }
        }
        if (q.active.locked && (moves != 1 || switches != 0)) fail("a locked request with other choices", g);
        if (q.kind == RequestData::Wait && (n != 1 || opts[0] != 0)) fail("a wait that isn't a pass", g);
        chosen[p] = opts[rng() % n];
      }
      c1 = chosen[0];
      c2 = chosen[1];
    }
  }
  std::printf("check_view: %zu games, %zu updates, %zu messages: %zu failures\n", games, updates, messages, failures);
  return failures != 0;
}
