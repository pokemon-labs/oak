// pi4's input (2026-11-24, docs/pi4.md): rby4's state (state_tokens.hpp,
// features.hpp) filled with perfect information, plus the last-move state
// (encode_last), from each player's side.
//
// The perfect-info Bookkeeping for `viewer` is the true battle (sets, PP,
// exact HP, stats, boosts, volatiles, last_damage) with libpkmn's chance
// Durations (counting up: no hidden counters), as the viewer's own
// Bookkeeping lays it out: our side by party position, the foe's numbered
// by the viewer's first appearance (view.hpp), the unseen ones after in
// party order, every one seen. Never a hidden value: each side's
// last_selected_move, last_used_move, counterable flag and `unshown` are
// tracked from the log by the *other* player's Bookkeeping (a click the log
// never showed stays unknown, flagged), so pi4 and its Showdown-view student
// read them from the same code.
//
// PI4_FIELDS per viewer: STATE_FIELDS, then LAST_FIELDS. Needs BattleData's
// `track` (the env's `state`).

#pragma once

#include <array>
#include <cstdint>

#include "battle_data.hpp"
#include "state_tokens.hpp"

namespace ataraxos {

inline constexpr size_t PI4_FIELDS = STATE_FIELDS + LAST_FIELDS;

// The foe's party slot -> the viewer's number (1-6): the seen ones as the
// view numbered them, then the unseen in party order. 0 stays 0.
inline std::array<uint8_t, 7> pi4_numbers(const View& view) {
  std::array<uint8_t, 7> num{};
  uint8_t next = view.seen;
  for (uint8_t slot = 1; slot <= 6; slot++) num[slot] = view.number[slot] ? view.number[slot] : ++next;
  return num;
}

inline Bookkeeping pi4_bookkeeping(const BattleData& d, uint8_t viewer) {
  const uint8_t foe = 1 - viewer;
  const PKMN::Battle& truth = d.state();
  Bookkeeping pk;
  pk.player = viewer;
  pk.teamsize = d.bk[viewer].teamsize;
  pk.battle = truth;
  pk.durations = PKMN::view(*pkmn_gen1_battle_options_chance_durations(
      const_cast<pkmn_gen1_battle_options*>(&d.options)));
  const std::array<uint8_t, 7> num = pi4_numbers(d.view[viewer]);
  const PKMN::Side& ts = truth.sides[foe];
  PKMN::Side& fs = pk.battle.sides[foe];
  for (uint8_t slot = 1; slot <= 6; slot++) fs.pokemon[num[slot] - 1] = ts.pokemon[slot - 1];
  for (size_t i = 0; i < 6; i++) fs.order[i] = num[ts.order[i]];
  // a Transform's target names a party slot
  for (uint8_t s = 0; s < 2; s++) {
    PKMN::Volatiles& v = pk.battle.sides[s].active.volatiles;
    const uint8_t t = v.transform_species();
    if (v.transform() && t && ((t >> 3) & 1) == foe) bk::set_transform_species(v, uint8_t((t & 8) | num[t & 7]));
  }
  for (uint8_t s = 0; s < 2; s++) {
    const Bookkeeping& other = d.bk[1 - s]; // s is its foe
    pk.battle.sides[s].last_selected_move = other.battle.sides[s].last_selected_move;
    pk.battle.sides[s].last_used_move = other.battle.sides[s].last_used_move;
    pk.battle.last_moves[s] = other.battle.last_moves[s];
    pk.unshown[s] = other.unshown[s];
  }
  return pk;
}

// `out` gets 2 * PI4_FIELDS int16s: viewer 0's, then viewer 1's.
inline void encode_pi4(const BattleData& d, int16_t* out) {
  for (uint8_t v = 0; v < 2; v++) {
    const Bookkeeping pk = pi4_bookkeeping(d, v);
    int16_t* o = out + v * PI4_FIELDS;
    encode_state(pk, o);
    encode_last(pk, o + STATE_FIELDS);
  }
}

} // namespace ataraxos
