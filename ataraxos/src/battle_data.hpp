// One battle as a Showdown server runs it: each player chooses on its own
// (`choose`, Showdown's side.choose), and the update runs once both have a
// choice. The request and its choices are Showdown's (request_data.hpp's
// `options`); each choice is played as libpkmn's `engine_choice`. Gen 1 has
// no hidden-illegal choice, so every one is accepted.
// Showdown's request shows a bind's turns, which `binds` tracks
// (request_data.hpp).
//
// Each player reads its own view of the log (view.hpp, Showdown's stream
// for that player): each update's bytes are rewritten per player and
// tokenized into `chunk[p]`, the tokens since p last chose. With `track`
// set (off by default), each player's view also updates its Bookkeeping
// (bookkeeping.hpp), for the state input (state_tokens.hpp).

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <libpkmn/data.h>

#include "bookkeeping.hpp"
#include "obs.hpp"
#include "request_data.hpp"
#include "tokens.hpp"
#include "view.hpp"

namespace ataraxos {

struct BattleData {
  enum Chosen : uint8_t { Held, Advanced, Ended };

  pkmn_gen1_battle battle;
  std::array<uint8_t, LOG_SIZE> buffer;
  pkmn_gen1_battle_options options;
  View view[2];
  pkmn_result result;
  RequestData request[2];
  // Showdown's choices for each player's request, and libpkmn's.
  std::array<pkmn_choice, MAX_CHOICES> option[2];
  uint8_t n_option[2];
  std::array<pkmn_choice, MAX_CHOICES> legal[2];
  uint8_t n_legal[2];
  Binds binds;
  pkmn_choice pending[2];
  bool chosen[2];
  // The server's: a request handed out and not yet answered.
  bool asked[2];
  std::vector<Token> chunk[2];
  bool track = false;
  // With `trace`, the game's omniscient log bytes, update after update
  // (trace_ends[i] is update i's end), for printing one game.
  bool trace = false;
  std::vector<uint8_t> trace_log;
  std::vector<size_t> trace_ends;
  Bookkeeping bk[2];
  std::vector<std::vector<Obs>> history[2];

  BattleData() = default;
  // `options` points at `buffer`.
  BattleData(const BattleData&) = delete;
  BattleData& operator=(const BattleData&) = delete;

  const PKMN::Battle& state() const { return PKMN::view(battle); }
  uint16_t turn() const { return state().turn; }

  // Starts `b`: the leads come in and the first requests go out.
  void reset(const pkmn_gen1_battle& b) {
    battle = b;
    for (auto& c : chunk) c.clear();
    trace_log.clear();
    trace_ends.clear();
    view[0] = View{0};
    view[1] = View{1};
    if (track) {
      std::array<uint8_t, 2> teamsize{};
      for (size_t s = 0; s < 2; s++) {
        for (const PKMN::Pokemon& p : state().sides[s].pokemon) teamsize[s] += p.species != PKMN::Data::Species::None;
      }
      for (uint8_t p = 0; p < 2; p++) {
        bk[p] = Bookkeeping(p, teamsize, state().sides[p]);
        history[p].clear();
      }
    }
    asked[0] = asked[1] = false;
    binds = Binds{};
    advance(0, 0);
  }

  bool over() const { return pkmn_result_type(result) != PKMN_RESULT_NONE; }

  // Whether `p` has a request to answer: not Wait, and no choice held.
  bool needs_choice(uint8_t p) const { return !over() && !chosen[p] && request[p].kind != RequestData::Wait; }

  // Showdown's side.choose: `c` must be one of option[p].
  Chosen choose(uint8_t p, pkmn_choice c) {
    assert(needs_choice(p));
    if (track) bk[p].chose(bk[p].selected(request[p], c));
    pending[p] = engine_choice(c, {legal[p].data(), n_legal[p]});
    chosen[p] = true;
    const uint8_t q = 1 - p;
    if (request[q].kind == RequestData::Wait) {
      pending[q] = 0;
      chosen[q] = true;
    }
    if (!chosen[q]) return Held;
    advance(pending[0], pending[1]);
    return over() ? Ended : Advanced;
  }

 private:
  void advance(pkmn_choice c1, pkmn_choice c2) {
    pkmn_gen1_log_options log{buffer.data(), buffer.size()};
    pkmn_gen1_battle_options_set(&options, &log, nullptr, nullptr);
    const PKMN::Battle before = state();
    result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    binds.update(before, state());
    assert(pkmn_result_type(result) != PKMN_RESULT_ERROR);
    size_t used = 0;
    while (buffer[used] != 0) used += 1 + message_size(ArgType(buffer[used]), &buffer[used + 1]);
    if (trace) {
      trace_log.insert(trace_log.end(), buffer.data(), buffer.data() + used);
      trace_ends.push_back(trace_log.size());
    }
    for (uint8_t p = 0; p < 2; p++) {
      std::array<uint8_t, LOG_SIZE> copy;
      std::memcpy(copy.data(), buffer.data(), used);
      copy[used] = 0;
      const size_t n = view[p].apply({copy.data(), used + 1});
      tokenize({copy.data(), n}, p, chunk[p]);
      if (track) {
        std::vector<Obs> obs = split({copy.data(), n});
        bk[p].update(obs, history[p]);
        history[p].push_back(std::move(obs));
      }
      if (over()) {
        request[p] = RequestData{};
        request[p].kind = RequestData::Wait;
        n_legal[p] = n_option[p] = 0;
        continue;
      }
      const pkmn_choice_kind kind = p == 0 ? pkmn_result_p1(result) : pkmn_result_p2(result);
      n_legal[p] = uint8_t(pkmn_gen1_battle_choices(&battle, p == 0 ? PKMN_PLAYER_P1 : PKMN_PLAYER_P2, kind,
                                                    legal[p].data(), legal[p].size()));
      request[p] = request_data(state(), p, kind, binds.of(state(), p));
      n_option[p] = showdown_choices(request[p], option[p].data());
      if (track) bk[p].request(request[p]);
    }
    chosen[0] = chosen[1] = false;
  }
};

} // namespace ataraxos
