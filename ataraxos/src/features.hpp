// Matchup features (cs3, 2026-11-02, the user's ask): numbers the net would
// otherwise rebuild from stats x type chart x the damage formula, computed
// from one player's Bookkeeping with bk::calc_damage (libpkmn's formula, as
// the bookkeeping uses it). The foe's stats are exact (every set maxes DVs
// and stat experience), its HP the middle of its percent; its unrevealed
// moves are unknown and give nothing. Appended to the state
// (state_tokens.hpp), FEATURE_FIELDS int16s.
//
// Layout:
//   our active's 4 moves against the foe's active, in the active's move
//     order (the net matches them to the request's moves by id);
//   the foe's active's 4 moves (unrevealed: 0) against our active;
//     each MOVE_FEATS: min and max damage and max on a crit (per mille of
//     the target's HP now, at most 2000), the chance it KOs and the chance
//     it KOs on a crit (of 256, over the damage rolls), the crit chance (of
//     256: Gen 1's base-speed rate with Focus Energy's glitch), the hit
//     chance (of 256 after accuracy and evasion boosts; Swift 256; 0 into
//     Fly / Dig), the type effectiveness (x4: 0 immune, 4 neutral, 16 x4),
//     and who moves first (1 the attacker, -1 the defender, 0 a tie: Quick
//     Attack and Counter's priority, then speed as modified, the defender's
//     move taken as priority 0);
//   our 6 Pokémon by party position (SWITCH_FEATS each, 0 when fainted or
//     with no foe active): the most the foe active's known moves deal it as
//     it would come in (per mille of its HP, max roll, no crit) and the
//     best KO chance among them, the most its moves deal the foe's active
//     and the best KO chance, and who's faster (1 it, -1 the foe, 0 a tie).
//     Our active is taken as it is, boosts and all.
// Multi-hit moves count 2 hits (Double Kick, Bonemerang, Twineedle) or 3
// (2-5); Psywave 1 to 1.5x the level; an OHKO move KOs when it can hit (not
// against a faster foe or an immune type); Counter and Bide give no damage.

#pragma once

#include <algorithm>
#include <cstdint>

#include "bookkeeping.hpp"

namespace ataraxos {

inline constexpr size_t MOVE_FEATS = 9;
inline constexpr size_t SWITCH_FEATS = 5;
inline constexpr size_t FEATURE_FIELDS = 8 * MOVE_FEATS + 6 * SWITCH_FEATS;

namespace features {

struct Hit {
  uint32_t min = 0, max = 0, crit_max = 0; // damage
  uint16_t ko = 0, crit_ko = 0;            // of 256
};

inline int16_t per_mille(uint32_t x, uint16_t hp) { return int16_t(std::min<uint32_t>(2000, x * 1000 / hp)); }

inline int sign(int x) { return (x > 0) - (x < 0); }

inline uint8_t hits(D::Effect e) {
  return e == D::Effect::DoubleHit || e == D::Effect::Twineedle ? 2 : e == D::Effect::MultiHit ? 3 : 1;
}

inline bool fixed(const D::MoveData& md) {
  return md.effect == D::Effect::SpecialDamage || md.effect == D::Effect::SuperFang;
}

// The type effectiveness against the defender's active, x4 (4 neutral).
inline uint8_t effectiveness(D::Type type, const PKMN::Side& def) {
  const uint8_t t1 = def.active.types & 15, t2 = (def.active.types >> 4) & 15;
  const uint8_t e1 = uint8_t(D::get_effectiveness(type, D::Type(t1)));
  return uint8_t(e1 * (t1 != t2 ? uint8_t(D::get_effectiveness(type, D::Type(t2))) : 2));
}

// Whether `att` moves before `def` with `m` (def at priority 0).
inline int first(const PKMN::Side& att, const PKMN::Side& def, bk::Move m) {
  const int p = m == bk::Move::QuickAttack ? 1 : m == bk::Move::Counter ? -1 : 0;
  if (p) return p;
  return sign(int(att.active.stats.spe) - int(def.active.stats.spe));
}

// The KO chance of 256 over the damage rolls, x the damage at roll 255.
inline uint16_t ko_chance(uint32_t x, uint8_t h, uint16_t hp) {
  if (x <= 1) return h * x >= hp ? 256 : 0;
  uint16_t n = 0;
  for (uint32_t r = 217; r <= 255; r++) n += h * (x * r / 255) >= hp;
  return uint16_t(n * 256 / 39);
}

inline Hit hit(const PKMN::Side& att, const PKMN::Side& def, bk::Move m) {
  Hit out;
  const uint16_t hp = def.stored().hp;
  if (m == bk::Move::None || m == bk::Move::Counter || m == bk::Move::Bide || hp == 0) return out;
  const D::MoveData md = bk::data(m);
  if (md.effect == D::Effect::OHKO) {
    if (effectiveness(md.type, def) && att.active.stats.spe >= def.active.stats.spe) {
      out.min = out.max = out.crit_max = hp;
      out.ko = out.crit_ko = 256;
    }
    return out;
  }
  if (m == bk::Move::Psywave) {
    const uint32_t top = att.stored().level * 3 / 2;
    out.min = 1;
    out.max = out.crit_max = top;
    out.ko = out.crit_ko = uint16_t(top >= hp ? (top - hp + 1) * 256 / top : 0);
    return out;
  }
  if (md.bp == 0 && !fixed(md)) return out;
  const uint8_t h = hits(md.effect);
  const uint32_t x = bk::calc_damage(att, def, m, false, false, true, 255);
  const uint32_t c = bk::calc_damage(att, def, m, true, false, true, 255);
  out.min = h * bk::calc_damage(att, def, m, false, false, true, 217);
  out.max = h * x;
  out.crit_max = h * c;
  if (fixed(md)) {
    out.ko = out.crit_ko = out.max >= hp ? 256 : 0;
  } else {
    out.ko = ko_chance(x, h, hp);
    out.crit_ko = ko_chance(c, h, hp);
  }
  return out;
}

// Of 256: Gen 1's crit rate (mechanics.zig checkCriticalHit); 0 for a move
// that can't crit.
inline uint16_t crit_chance(const PKMN::Side& att, bk::Move m) {
  const D::MoveData md = bk::data(m);
  if (m == bk::Move::None || md.bp == 0 || fixed(md) || md.effect == D::Effect::OHKO) return 0;
  uint16_t rate = D::get_species_data(att.stored().species).base_stats.spe / 2;
  rate = att.active.volatiles.focus_energy() ? rate / 2 : std::min<uint16_t>(rate * 2, 255);
  return md.effect == D::Effect::HighCritical ? std::min<uint16_t>(rate * 4, 255) : rate / 2;
}

// Of 256 (mechanics.zig moveHit, Showdown's order).
inline uint16_t hit_chance(const PKMN::Side& att, const PKMN::Side& def, bk::Move m) {
  if (m == bk::Move::None) return 0;
  const D::MoveData md = bk::data(m);
  if (md.effect == D::Effect::Swift) return 256;
  if (def.active.volatiles.invulnerable()) return 0;
  uint32_t a = md.accuracy;
  const auto& b = D::boosts[att.active.boosts.acc() + 6];
  a = a * b[0] / b[1];
  const auto& e = D::boosts[-def.active.boosts.eva() + 6];
  a = a * e[0] / e[1];
  return uint16_t(std::clamp<uint32_t>(a, 1, 255));
}

inline int16_t* move(const PKMN::Side& att, const PKMN::Side& def, bk::Move m, int16_t* o) {
  if (m == bk::Move::None || att.active.species == D::Species::None || def.active.species == D::Species::None ||
      def.stored().hp == 0) {
    std::fill(o, o + MOVE_FEATS, int16_t(0));
    return o + MOVE_FEATS;
  }
  const uint16_t hp = def.stored().hp;
  const Hit h = hit(att, def, m);
  *o++ = per_mille(h.min, hp);
  *o++ = per_mille(h.max, hp);
  *o++ = per_mille(h.crit_max, hp);
  *o++ = int16_t(h.ko);
  *o++ = int16_t(h.crit_ko);
  *o++ = int16_t(crit_chance(att, m));
  *o++ = int16_t(hit_chance(att, def, m));
  *o++ = effectiveness(bk::data(m).type, def);
  *o++ = int16_t(first(att, def, m));
  return o;
}

// Party slot k (1-6) of `us` as it would come in against `foe`'s active.
inline int16_t* incoming(const PKMN::Side& us, const PKMN::Side& foe, uint8_t k, int16_t* o) {
  const PKMN::Pokemon& p = us.pokemon[k - 1];
  int16_t* end = o + SWITCH_FEATS;
  std::fill(o, end, int16_t(0));
  if (p.species == D::Species::None || p.hp == 0 || foe.active.species == D::Species::None ||
      foe.stored().hp == 0)
    return end;
  PKMN::Side s = us;
  if (s.order[0] != k) {
    for (uint8_t j = 1; j < 6; j++) {
      if (s.order[j] == k) std::swap(s.order[0], s.order[j]);
    }
    s.active = PKMN::switch_in(p);
    bk::status_modify(uint8_t(p.status), s.active.stats);
  }
  uint32_t taken = 0, dealt = 0;
  uint16_t taken_ko = 0, dealt_ko = 0;
  for (const PKMN::MoveSlot& m : foe.active.moves) {
    const Hit h = hit(foe, s, m.id);
    taken = std::max(taken, h.max);
    taken_ko = std::max(taken_ko, h.ko);
  }
  for (const PKMN::MoveSlot& m : s.active.moves) {
    const Hit h = hit(s, foe, m.id);
    dealt = std::max(dealt, h.max);
    dealt_ko = std::max(dealt_ko, h.ko);
  }
  o[0] = per_mille(taken, p.hp);
  o[1] = int16_t(taken_ko);
  o[2] = per_mille(dealt, foe.stored().hp);
  o[3] = int16_t(dealt_ko);
  o[4] = int16_t(sign(int(s.active.stats.spe) - int(foe.active.stats.spe)));
  return end;
}

} // namespace features

// `out` gets FEATURE_FIELDS int16s, from bk.player's view.
inline void encode_features(const Bookkeeping& bk, int16_t* out) {
  const PKMN::Side& us = bk.battle.sides[bk.player];
  const PKMN::Side& foe = bk.battle.sides[1 - bk.player];
  int16_t* o = out;
  for (const PKMN::MoveSlot& m : us.active.moves) o = features::move(us, foe, m.id, o);
  for (const PKMN::MoveSlot& m : foe.active.moves) o = features::move(foe, us, m.id, o);
  for (uint8_t k = 1; k <= 6; k++) o = features::incoming(us, foe, k, o);
  assert(o == out + FEATURE_FIELDS);
}

} // namespace ataraxos
