// The bookkeeping state as network input: what one player's Bookkeeping
// (bookkeeping.hpp) holds at a decision, flattened to STATE_FIELDS int16s,
// as encode() flattens the request. Sides are relative to the viewer: "ours"
// is the viewer's, "foe" the other. birch's layout for Gen 1 (the user's,
// 2026-10-31): no field token, the turn clamped to 1000, both sides' 6
// Pokémon with PP (the request has no bench PP).
//
// Layout:
//   turn (clamped to MAX_TURN).
//   2 actives, ours first (ACTIVE_FIELDS each): species (0 none), types
//     (2, Type + 1, 0 none), boosts (6: atk, def, spe, spc, acc, eva), stats (4: atk, def,
//     spe, spc, as modified), the moves (4) and their PP (4), as the active
//     has them (Mimic's, Transform's), the disabled move, the Durations
//     (confusion, disable, attacking, binding), the substitute's HP, the
//     toxic counter, Bide's damage, then the flags (ACTIVE_FLAGS bits in
//     one int16): bide, thrashing, charging, binding, invulnerable,
//     confusion, mist, focus_energy, substitute, recharging, rage,
//     leech_seed, toxic, light_screen, reflect, transform.
//   12 Pokémon, ours then the foe's, by party position (ours) or by first
//     appearance (the foe's) (MON_FIELDS each): exists (within the team
//     size), seen (0 for the foe's not yet seen: species None and the rest
//     0), active, species, level, hp, max_hp (the foe's HP is the middle of
//     its percent), status (the raw byte), sleep turns, the moves (4) and
//     their PP (4); the foe's unrevealed moves are 0 with PP 0.
//   the matchup features (features.hpp, FEATURE_FIELDS; cs3's, 2026-11-02:
//     the nets before it read only the first BASE_STATE_FIELDS).
// The hidden counters in libpkmn's Volatiles (confusion_left, attacks,
// disable_left) aren't the viewer's to know: the Durations count up instead.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "bookkeeping.hpp"
#include "features.hpp"

namespace ataraxos {

inline constexpr uint16_t MAX_TURN = 1000;
inline constexpr size_t ACTIVE_FLAGS = 16;
inline constexpr size_t ACTIVE_FIELDS = 1 + 2 + 6 + 4 + 8 + 1 + 4 + 3 + 1;
inline constexpr size_t MON_FIELDS = 3 + 5 + 1 + 8;
inline constexpr size_t BASE_STATE_FIELDS = 1 + 2 * ACTIVE_FIELDS + 12 * MON_FIELDS;
inline constexpr size_t STATE_FIELDS = BASE_STATE_FIELDS + FEATURE_FIELDS;

namespace detail {

inline int16_t* active(const PKMN::Side& side, const PKMN::Duration& d, int16_t* o) {
  const PKMN::ActivePokemon& a = side.active;
  const PKMN::Volatiles& v = a.volatiles;
  const bool none = a.species == D::Species::None;
  *o++ = int16_t(a.species);
  *o++ = none ? 0 : (a.types & 0xF) + 1;
  *o++ = none ? 0 : (a.types >> 4) + 1;
  const PKMN::Boosts& b = a.boosts;
  for (int8_t x : {b.atk(), b.def(), b.spe(), b.spc(), b.acc(), b.eva()}) *o++ = x;
  for (uint16_t x : {a.stats.atk, a.stats.def, a.stats.spe, a.stats.spc}) *o++ = int16_t(x);
  for (const PKMN::MoveSlot& m : a.moves) *o++ = int16_t(m.id);
  for (const PKMN::MoveSlot& m : a.moves) *o++ = m.pp;
  const uint8_t dm = v.disable_move();
  *o++ = dm ? int16_t(a.moves[dm - 1].id) : 0;
  for (uint8_t x : {d.confusion(), d.disable(), d.attacking(), d.binding()}) *o++ = x;
  *o++ = v.substitute_hp();
  *o++ = v.toxic_counter();
  *o++ = int16_t(std::min<uint16_t>(v.state(), INT16_MAX));
  const bool flags[ACTIVE_FLAGS] = {v.bide(),         v.thrashing(),  v.charging(),   v.binding(),
                                    v.invulnerable(), v.confusion(),  v.mist(),       v.focus_energy(),
                                    v.substitute(),   v.recharging(), v.rage(),       v.leech_seed(),
                                    v.toxic(),        v.light_screen(), v.reflect(),  v.transform()};
  uint16_t bits = 0;
  for (size_t i = 0; i < ACTIVE_FLAGS; i++) bits |= uint16_t(flags[i]) << i;
  *o++ = int16_t(bits);
  return o;
}

inline int16_t* mons(const PKMN::Side& side, const PKMN::Duration& d, uint8_t teamsize, int16_t* o) {
  for (uint8_t k = 1; k <= 6; k++) {
    const PKMN::Pokemon& p = side.pokemon[k - 1];
    const bool seen = p.species != D::Species::None;
    uint8_t pos = 6;
    for (uint8_t j = 0; j < 6; j++) {
      if (side.order[j] == k) pos = j;
    }
    *o++ = k <= teamsize;
    *o++ = seen;
    *o++ = seen && pos == 0;
    if (!seen) {
      for (size_t j = 3; j < MON_FIELDS; j++) *o++ = 0;
      continue;
    }
    *o++ = int16_t(p.species);
    *o++ = p.level;
    *o++ = int16_t(p.hp);
    *o++ = int16_t(p.stats.hp);
    *o++ = int16_t(uint8_t(p.status));
    *o++ = pos < 6 ? d.sleep(pos) : 0;
    for (const PKMN::MoveSlot& m : p.moves) *o++ = int16_t(m.id);
    for (const PKMN::MoveSlot& m : p.moves) *o++ = m.pp;
  }
  return o;
}

} // namespace detail

// `out` gets STATE_FIELDS int16s, from bk.player's view.
inline void encode_state(const Bookkeeping& bk, int16_t* out) {
  const PKMN::Battle& b = bk.battle;
  const uint8_t us = bk.player, foe = 1 - us;
  int16_t* o = out;
  *o++ = int16_t(std::min(b.turn, MAX_TURN));
  o = detail::active(b.sides[us], bk.durations.get(us), o);
  o = detail::active(b.sides[foe], bk.durations.get(foe), o);
  o = detail::mons(b.sides[us], bk.durations.get(us), bk.teamsize[us], o);
  o = detail::mons(b.sides[foe], bk.durations.get(foe), bk.teamsize[foe], o);
  encode_features(bk, o);
}

} // namespace ataraxos
