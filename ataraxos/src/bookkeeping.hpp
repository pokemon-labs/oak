// Bookkeeping: the part of the state that one player's view of the log
// determines, kept in a libpkmn Battle + Durations: safari's PSBattle and
// Mechanics (~/safari/src/battle.py, mechanics.py), which read Showdown's
// text, ported to libpkmn's binary messages (obs.hpp) as view.hpp shows them.
// Checked by tests/cpp/check_bookkeeping.cpp against the true battle with
// oak's compare_battles (cpp/include/libpkmn/client.h).
//
// Every set is assumed to max its DVs and stat experience (the user's,
// 2026-10-31), so a Pokémon's stats follow from its species and level, and
// the other side's HP from its percent to within 1%.
//
// Sides are absolute (sides[0] P1). Our own side starts as our team (the
// request gives it), Pokémon by party position; the other side's Pokémon
// are numbered by first appearance (view.hpp), the k-th in pokemon[k - 1],
// its moves in the order they appear, an unseen one None. `order` follows
// the switches as libpkmn's does.
//
// What the text of a message doesn't carry is read off its neighbours, as
// safari does: a Move's [still] and [miss] are the LastStill and LastMiss
// after it; Rage takes hold unless an Immune or a Substitute's End follows;
// a Substitute's HP cost is the Damage after its Start.
//
// Gen 1's stat modification glitch: a boost or drop recomputes the stat from
// the unmodified one, and then the paralysis and burn drops are applied
// again (to the foe on a boost, to the target on a drop: `boost`, `unboost`).
// A failed boost restores the stats it changed (`stored`).

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

#include <libpkmn/data.h>
#include <libpkmn/init.h>

#include "obs.hpp"
#include "request_data.hpp"
#include "view.hpp"

namespace ataraxos {

namespace bk {

namespace D = PKMN::Data;
using D::Move;

// libpkmn's protocol enums (cpp/extern/engine/src/lib/common/protocol.zig).
enum DamageReason : uint8_t { DmgNone, DmgPoison, DmgBurn, DmgConfusion, DmgLeechSeed, DmgRecoil };
enum CantReason : uint8_t { CantSleep, CantFreeze, CantParalysis, CantBound, CantFlinch, CantDisable, CantRecharge, CantPP };
enum BoostReason : uint8_t { BoostRage, BoostAtk, BoostDef, BoostSpe, BoostSpa, BoostSpd, BoostAcc, BoostEva };
enum ActivateReason : uint8_t { ActBide, ActConfusion, ActHaze, ActMist, ActStruggle, ActSubstitute, ActSplash };
enum StartReason : uint8_t {
  StartBide, StartConfusion, StartConfusionSilent, StartFocusEnergy, StartLeechSeed, StartLightScreen,
  StartMist, StartReflect, StartSubstitute, StartTypeChange, StartDisable, StartMimic
};
enum EndReason : uint8_t {
  EndDisable = 0, EndConfusion = 1, EndBide = 2, EndSubstitute = 3,
  EndDisableSilent = 15, EndConfusionSilent, EndMistSilent, EndFocusEnergySilent, EndLeechSeedSilent,
  EndToxicSilent, EndLightScreenSilent, EndReflectSilent, EndBideSilent, EndLeechSeedFrom
};
inline constexpr uint8_t STATUS_FROM = 2;

inline constexpr uint8_t SLP = 0b00000001, PSN = 0b00001000, BRN = 0b00010000, PAR = 0b01000000;
inline constexpr uint8_t TOX = 0b10001000, REST = 0b10000010;

inline bool is_sleep(uint8_t s) { return s & 7; }

inline D::MoveData data(Move m) {
  if (m == Move::None) return D::MoveData{};
  return D::move_data(m);
}

inline bool one_of(Move m, std::initializer_list<Move> ms) { return std::find(ms.begin(), ms.end(), m) != ms.end(); }
inline bool charging_move(Move m) { return one_of(m, {Move::SkullBash, Move::SolarBeam, Move::SkyAttack, Move::RazorWind}); }
inline bool invulnerable_move(Move m) { return one_of(m, {Move::Fly, Move::Dig}); }
inline bool thrashing_move(Move m) { return one_of(m, {Move::Thrash, Move::PetalDance}); }
inline bool binding_move(Move m) { return one_of(m, {Move::Bind, Move::Clamp, Move::FireSpin, Move::Wrap}); }
inline bool drain(Move m) {
  const D::Effect e = data(m).effect;
  return e == D::Effect::DrainHP || e == D::Effect::DreamEater;
}

// Volatiles' fields with no setter in oak's header.
inline void set_substitute_hp(PKMN::Volatiles& v, uint8_t x) {
  v.bits = (v.bits & ~(uint64_t{0xFF} << 40)) | (uint64_t{x} << 40);
}
inline void set_transform_species(PKMN::Volatiles& v, uint8_t x) {
  v.bits = (v.bits & ~(uint64_t{0xF} << 48)) | (uint64_t(x & 0xF) << 48);
}

inline uint16_t boosted(uint16_t stat, int8_t b) {
  const auto& r = D::boosts[b + 6];
  return uint16_t(stat * r[0] / r[1]);
}

// Mechanics.status_modify
inline void status_modify(uint8_t status, PKMN::Stats& stats) {
  if (status == PAR) {
    stats.spe = std::max<uint16_t>(1, stats.spe / 4);
  } else if (status == BRN) {
    stats.atk = std::max<uint16_t>(1, stats.atk / 2);
  }
}

} // namespace bk

namespace D = PKMN::Data;

struct Bookkeeping {
  uint8_t player = 0; // whose view
  std::array<uint8_t, 2> teamsize{};
  PKMN::Battle battle{};
  PKMN::Durations durations{};
  // Each active's stats at the start of the message being read (safari's
  // SideAux), restored when a boost fails.
  std::array<PKMN::Stats, 2> stored{};

  Bookkeeping() = default;
  // `own` is our side as the battle starts (our team, before the leads come
  // in); `teamsize` each side's Pokémon.
  Bookkeeping(uint8_t p, std::array<uint8_t, 2> sizes, const PKMN::Side& own) : player(p), teamsize(sizes) {
    battle.sides[p] = own;
  }

  // Our own choice, before the update it makes (safari's test sets these
  // from the choice): the move we selected.
  void chose(pkmn_choice c) {
    if (choice_type(c) != PKMN_CHOICE_MOVE) return;
    PKMN::Side& s = battle.sides[player];
    const PKMN::Volatiles& v = s.active.volatiles;
    const bool forced = v.charging() || v.recharging() || v.rage() || v.thrashing();
    const uint8_t d = choice_data(c);
    if (d > 0 && !forced) {
      s.last_selected_move = s.active.moves[d - 1].id;
      battle.last_moves[player].index = d;
    } else if (d == 0) {
      s.last_selected_move = bk::Move::Struggle;
    }
  }

  // Our own active's moves and PP as the request shows them (a Transform's,
  // a Mimic's), after the update.
  void request(const RequestData& q) {
    if (q.kind != RequestData::Move_ || q.active.locked) return;
    PKMN::ActivePokemon& a = battle.sides[player].active;
    for (uint8_t i = 0; i < 4; i++) {
      a.moves[i].id = q.active.moves[i].id;
      a.moves[i].pp = q.active.moves[i].pp;
    }
  }

  // One update's messages; `history` the earlier updates' (unused: every
  // rule here reads its own update).
  void update(std::span<const Obs> obs, std::span<const std::vector<Obs>> = {}) {
    msgs = obs;
    store_stats();
    for (i = 0; i < obs.size(); i++) read(obs[i]);
  }

 private:
  std::span<const Obs> msgs;
  size_t i = 0;
  // The move each side's log line shows while its binding move goes on.
  std::array<bk::Move, 2> bind_shown{};
  // The Metronome or Mirror Move that called each side's current lock, or
  // None.
  std::array<bk::Move, 2> via{};

  uint8_t side_of(uint8_t id) const { return ident_player(id); }
  PKMN::Side& side(uint8_t s) { return battle.sides[s]; }
  PKMN::Duration& dur(uint8_t s) { return durations.get(s); }

  const Obs* prev() const { return i > 0 ? &msgs[i - 1] : nullptr; }
  // The next message that isn't a LastStill or LastMiss.
  const Obs* next() const {
    for (size_t j = i + 1; j < msgs.size(); j++) {
      if (msgs[j].type != ArgType::LastStill && msgs[j].type != ArgType::LastMiss) return &msgs[j];
    }
    return nullptr;
  }
  // Whether the Move at i has a LastStill / LastMiss before the next Move.
  bool amended(ArgType t) const {
    for (size_t j = i + 1; j < msgs.size() && msgs[j].type != ArgType::Move; j++) {
      if (msgs[j].type == t) return true;
    }
    return false;
  }

  void store_stats() {
    for (uint8_t s = 0; s < 2; s++) stored[s] = side(s).active.stats;
  }
  void unstore_stats() {
    for (uint8_t s = 0; s < 2; s++) {
      PKMN::Stats& st = side(s).active.stats;
      const uint16_t hp = st.hp;
      st = stored[s];
      st.hp = hp;
    }
  }

  // The stored Pokémon behind the active's stats: its own, or what it
  // transformed into (Mechanics.unmodified_stats).
  const PKMN::Stats& unmodified(uint8_t s) {
    const uint8_t t = side(s).active.volatiles.transform_species();
    if (t == 0) return side(s).stored().stats;
    return side((t >> 3) & 1).pokemon[(t & 7) - 1].stats;
  }

  // HP from a message's HP fields: exact for our side, from the percent for
  // the other (kept when the estimate still shows that percent).
  uint16_t hp_of(uint8_t s, const PKMN::Pokemon& p, uint16_t hp, uint16_t max) const {
    if (s == player || max == 0) return hp;
    if (hp == 0) return 0;
    if (view_percent(p.hp, p.stats.hp) == hp) return p.hp;
    return uint16_t(uint32_t(p.stats.hp) * hp / max);
  }

  // Mechanics.interrupt
  void interrupt(uint8_t s) {
    PKMN::Volatiles& v = side(s).active.volatiles;
    if (!v.rage()) v.set_state(0);
    v.set_bide(false);
    v.set_thrashing(false);
    v.set_charging(false); // invulnerable stays: Gen 1's Fly/Dig glitch
    v.set_binding(false);
    dur(s).set_attacking(0);
    dur(s).set_binding(0);
  }

  enum class Why : uint8_t { None, Sleep, Freeze, Bound, Flinch, Recharge, Disable, Confusion, Paralysis, Bide };

  // Mechanics.before_move
  void before_move(uint8_t s, Why why = Why::None) {
    PKMN::Side& sd = side(s);
    PKMN::Volatiles& v = sd.active.volatiles;
    PKMN::Duration& d = dur(s);
    if (bk::is_sleep(uint8_t(sd.stored().status))) {
      sd.last_used_move = bk::Move::None;
      return;
    }
    if (why == Why::Freeze) {
      sd.last_used_move = bk::Move::None;
      return;
    }
    if (why == Why::Bound || why == Why::Flinch) return;
    if (why == Why::Recharge) {
      v.set_recharging(false);
      return;
    }
    if (v.disable_move() != 0) d.set_disable(d.disable() + 1);
    if (why == Why::Disable) {
      v.set_charging(false);
      return;
    }
    if (why == Why::Confusion || why == Why::Paralysis) {
      interrupt(s);
      return;
    }
    if (v.bide()) v.set_state(uint16_t(v.state() + battle.last_damage));
    if (why == Why::Bide) return;
    if (v.thrashing()) d.set_attacking(d.attacking() + 1);
    if (v.binding()) d.set_binding(d.binding() + 1);
  }

  // Mechanics.faint
  void faint(uint8_t s) {
    const uint8_t o = 1 - s;
    side(s).stored().hp = 0;
    dur(s).set_binding(0);
    dur(o).set_binding(0);
    side(s).last_used_move = bk::Move::None;
    side(o).last_used_move = bk::Move::None;
    PKMN::Volatiles& ov = side(o).active.volatiles;
    if (ov.bide()) ov.set_state(0);
  }

  void cure_status(uint8_t s) {
    side(s).stored().status = D::Status::None;
    dur(s).set_sleep(0, 0);
  }

  // Mechanics.clear_boosts, clear_volatiles (Haze)
  void clear_boosts(uint8_t s) {
    PKMN::ActivePokemon& a = side(s).active;
    const PKMN::Stats& st = side(s).stored().stats;
    a.stats.atk = st.atk;
    a.stats.def = st.def;
    a.stats.spe = st.spe;
    a.stats.spc = st.spc;
    a.boosts = {};
  }
  void clear_volatiles(uint8_t s) {
    PKMN::Volatiles& v = side(s).active.volatiles;
    v.set_disable_move(0);
    v.set_confusion(false);
    v.set_mist(false);
    v.set_focus_energy(false);
    v.set_leech_seed(false);
    v.set_light_screen(false);
    v.set_reflect(false);
    v.set_toxic(false);
    v.set_toxic_counter(0);
    dur(s).set_disable(0);
    dur(s).set_confusion(0);
  }

  // Mechanics.calc_damage: `move` None is the confusion self-hit.
  uint16_t calc_damage(uint8_t a, uint8_t d, bk::Move move, bool crit, bool adjust = true, uint8_t roll = 217) {
    PKMN::Side& att = side(a);
    PKMN::Side& def = side(d);
    bool cfz = false;
    if (move == bk::Move::None) {
      cfz = true;
      move = bk::Move::Pound;
    }
    if (move == bk::Move::Counter) return uint16_t(2 * battle.last_damage);
    if (move == bk::Move::Bide) {
      const uint16_t x = uint16_t(att.active.volatiles.state() * 2);
      att.active.volatiles.set_state(0);
      return x;
    }
    const D::MoveData md = bk::data(move);
    if (md.effect == D::Effect::SpecialDamage || md.effect == D::Effect::SuperFang) {
      if (move == bk::Move::SuperFang) return std::max<uint16_t>(def.stored().hp / 2, 1);
      if (move == bk::Move::SeismicToss || move == bk::Move::NightShade) return att.stored().level;
      if (move == bk::Move::SonicBoom) return 20;
      if (move == bk::Move::DragonRage) return 40;
      return 1; // Psywave
    }
    const uint8_t type = uint8_t(md.type);
    const bool special = type >= 8;
    uint32_t attack = crit ? (special ? att.stored().stats.spc : att.stored().stats.atk)
                           : (special ? att.active.stats.spc : att.active.stats.atk);
    uint32_t defense = crit ? (special ? def.stored().stats.spc : def.stored().stats.def)
                            : (special ? def.active.stats.spc * (def.active.volatiles.light_screen() ? 2 : 1)
                                       : def.active.stats.def * (!cfz && def.active.volatiles.reflect() ? 2 : 1));
    if (attack > 255 || defense > 255) {
      attack = std::max<uint32_t>((attack / 4) & 255, 1);
      defense = std::max<uint32_t>((defense / 4) & 255, 1);
    }
    const uint32_t lvl = att.stored().level * (crit ? 2 : 1);
    if (md.effect == D::Effect::Explode) defense = std::max<uint32_t>(defense / 2, 1);
    if (defense == 0) return 0;
    uint32_t x = lvl * 2 / 5 + 2;
    x *= md.bp;
    x *= attack;
    x /= defense;
    x /= 50;
    x = std::min<uint32_t>(997, x);
    x += 2;
    if (adjust) {
      const uint8_t at = att.active.types;
      if (type == (at & 15) || type == (at >> 4)) x += x / 2;
      const uint8_t t1 = def.active.types & 15, t2 = (def.active.types >> 4) & 15;
      const uint32_t e1 = uint32_t(D::get_effectiveness(md.type, D::Type(t1))) * 5;
      const uint32_t e2 = uint32_t(D::get_effectiveness(md.type, D::Type(t2))) * 5;
      if (e1 != 10) x = x * e1 / 10;
      if (t1 != t2 && e2 != 10) x = x * e2 / 10;
      if (x == 0) return 0;
    }
    if (x <= 1) return uint16_t(x);
    return uint16_t(x * roll / 255);
  }

  // Whether the attacker against side `target` crit in this update.
  bool crit_against(uint8_t target) const {
    for (const Obs& o : msgs) {
      if (o.type == ArgType::Crit && ident_player(o.u8(0)) == target) return true;
    }
    return false;
  }

  uint16_t sub_damage(uint8_t attacker) {
    const uint8_t target = 1 - attacker;
    const uint16_t x = calc_damage(attacker, target, side(attacker).last_used_move, crit_against(target));
    battle.last_damage = x;
    return x;
  }

  void read(const Obs& o) {
    switch (o.type) {
    case ArgType::Move: move(o); break;
    case ArgType::Switch: switch_in(o); break;
    case ArgType::Cant: cant(o); break;
    case ArgType::Faint: faint(side_of(o.u8(0))); break;
    case ArgType::Turn: battle.turn = o.u16(0); break;
    case ArgType::Damage: damage(o, true); break;
    case ArgType::Heal: damage(o, false); break;
    case ArgType::Status: status(o); break;
    case ArgType::CureStatus: curestatus(o); break;
    case ArgType::Boost: boost(o); break;
    case ArgType::ClearAllBoost:
      for (uint8_t s = 0; s < 2; s++) {
        clear_boosts(s);
        clear_volatiles(s);
      }
      break;
    case ArgType::Fail:
      if (prev() && prev()->type == ArgType::Boost) unstore_stats();
      break;
    case ArgType::Miss: {
      // a miss against Fly or Dig leaves the last damage as it was
      if (side(1 - side_of(o.u8(0))).active.volatiles.invulnerable()) break;
      battle.last_damage = 0;
      const bk::Move last = side(side_of(o.u8(0))).last_used_move;
      if (last == bk::Move::JumpKick || last == bk::Move::HighJumpKick) battle.last_damage = 1;
      break;
    }
    case ArgType::Prepare: {
      PKMN::Volatiles& v = side(side_of(o.u8(0))).active.volatiles;
      const bk::Move m = bk::Move(o.u8(1));
      v.set_charging(true);
      if (bk::invulnerable_move(m)) v.set_invulnerable(true);
      break;
    }
    case ArgType::MustRecharge: side(side_of(o.u8(0))).active.volatiles.set_recharging(true); break;
    case ArgType::Activate: activate(o); break;
    case ArgType::Start: start(o); break;
    case ArgType::End: end(o); break;
    case ArgType::Transform: transform(o); break;
    default: break;
    }
  }

  void switch_in(const Obs& o) {
    const uint8_t s = side_of(o.u8(0)), opp = 1 - s;
    const uint8_t k = ident_slot(o.u8(0)); // party position (ours) or number (theirs)
    PKMN::Side& sd = side(s);
    PKMN::Side& od = side(opp);
    size_t pos = 6;
    for (size_t j = 0; j < 6; j++) {
      if (sd.order[j] == k) {
        pos = j;
        break;
      }
    }
    if (pos == 6) {
      // first sight of the other side's k-th Pokémon
      for (size_t j = 0; j < 6; j++) {
        if (sd.order[j] == 0) {
          pos = j;
          break;
        }
      }
      assert(pos < 6);
      sd.order[pos] = k;
      PKMN::Set set{};
      set.species = D::Species(o.u8(1));
      set.level = o.u8(2);
      set.moves = {};
      sd.pokemon[k - 1] = PKMN::Init::init_pokemon(set);
      PKMN::Pokemon& p = sd.pokemon[k - 1];
      for (auto& m : p.moves) m = {};
      p.hp = hp_of(s, p, o.u16(3), o.u16(5));
      p.status = D::Status(o.u8(7));
    }
    std::swap(sd.order[0], sd.order[pos]);
    battle.last_moves[s].index = 1;
    sd.last_used_move = bk::Move::None;
    od.last_used_move = bk::Move::None;
    via[s] = bk::Move::None;

    PKMN::Pokemon& p = sd.stored();
    p.hp = hp_of(s, p, o.u16(3), o.u16(5));
    sd.active = PKMN::switch_in(p);
    sd.active.species = p.species;
    bk::status_modify(uint8_t(p.status), sd.active.stats);
    store_stats();
    PKMN::Duration& d = dur(s);
    const uint8_t old = d.sleep(0);
    d.set_sleep(0, d.sleep(pos));
    d.set_sleep(pos, old);
    d.set_confusion(0);
    d.set_disable(0);
    d.set_attacking(0);
    d.set_binding(0);
    if (uint8_t(p.status) == bk::TOX) p.status = D::Status(bk::PSN);
    od.active.volatiles.set_binding(false);
  }

  void damage(const Obs& o, bool is_damage) {
    const uint8_t s = side_of(o.u8(0)), opp = 1 - s;
    PKMN::Side& sd = side(s);
    PKMN::Side& od = side(opp);
    PKMN::Volatiles& v = sd.active.volatiles;
    const uint8_t reason = o.u8(6);
    const uint16_t hp = hp_of(s, sd.stored(), o.u16(1), o.u16(3));
    const bool confusion = is_damage && reason == bk::DmgConfusion;
    bool from_sub = false, bind_into_ghost = false;
    if (const Obs* p = prev()) {
      if (p->type == ArgType::Start && p->u8(1) == bk::StartSubstitute) from_sub = true;
      if (p->type == ArgType::Move) {
        const bk::Move m = bk::Move(p->u8(1));
        const D::MoveData md = bk::data(m);
        const uint8_t t1 = sd.active.types & 15, t2 = sd.active.types >> 4;
        const int eff = int(D::get_effectiveness(md.type, D::Type(t1))) * int(D::get_effectiveness(md.type, D::Type(t2)));
        if (eff == 0 && bk::binding_move(m)) bind_into_ghost = true;
      }
    }
    const bool counts = (reason == bk::DmgNone || confusion) && !from_sub;
    if (is_damage && counts && !bind_into_ghost) {
      if (confusion && hp == 0) {
        const uint16_t def = od.active.stats.def;
        od.active.stats.def = sd.active.stats.def;
        battle.last_damage = calc_damage(s, opp, bk::Move::None, false, false, 255);
        od.active.stats.def = def;
      } else {
        const uint16_t dmg = sd.stored().hp > hp ? uint16_t(sd.stored().hp - hp) : 0;
        battle.last_damage = std::max<uint16_t>(1, dmg);
        if (!confusion && bk::drain(od.last_used_move)) {
          battle.last_damage = std::max<uint16_t>(1, battle.last_damage / 2);
        }
      }
    }
    sd.stored().hp = hp;
    if (!is_damage) return;
    switch (reason) {
    case bk::DmgConfusion:
      before_move(s, Why::Confusion);
      store_stats();
      break;
    case bk::DmgBurn:
    case bk::DmgPoison:
    case bk::DmgLeechSeed:
      if (v.toxic()) v.set_toxic_counter(v.toxic_counter() + 1);
      break;
    default: break;
    }
    if (is_damage) store_stats();
  }

  void move(const Obs& o) {
    const uint8_t s = side_of(o.u8(0)), opp = 1 - s;
    PKMN::Side& sd = side(s);
    PKMN::Side& od = side(opp);
    PKMN::Volatiles& v = sd.active.volatiles;
    PKMN::Duration& d = dur(s);
    before_move(s);
    store_stats();
    const bk::Move m = bk::Move(o.u8(1));
    const bool from = o.u8(3) != 0;
    const bk::Move from_move = from ? bk::Move(o.u8(4)) : bk::Move::None;
    const bool missed = amended(ArgType::LastMiss);
    const bool still = amended(ArgType::LastStill);
    const bool from_metronome = from_move == bk::Move::Metronome;
    const bool from_mirror = from_move == bk::Move::MirrorMove;
    const bool rage = m == bk::Move::Rage;
    // Fly and Dig charge as the charging moves do
    const bool charging = bk::charging_move(m) || bk::invulnerable_move(m);
    const bool thrashing = bk::thrashing_move(m);
    // A binding move's later turns show the move that started it: the
    // binding move, or the Metronome or Mirror Move that called it.
    const bool binding = bk::binding_move(m) || (v.binding() && m == bind_shown[s]);
    if (binding) {
      od.active.volatiles.set_recharging(false);
    } else {
      v.set_binding(false);
      d.set_binding(0);
    }
    const bool deduct = !((rage && v.rage()) || (charging && !v.charging()) || (thrashing && v.thrashing()) ||
                          (binding && v.binding()));
    // Metronome or Mirror Move calling a charging move: no PP on the turn it
    // charges.
    bool spend = deduct;
    if (m == bk::Move::Metronome || m == bk::Move::MirrorMove) {
      if (const Obs* n = next(); n && n->type == ArgType::Move && n->u8(3) && bk::Move(n->u8(4)) == m) {
        const bk::Move called = bk::Move(n->u8(1));
        if ((bk::charging_move(called) || bk::invulnerable_move(called)) && !v.charging()) spend = false;
      }
    }
    // A Mimic'd move: the active's slot where the stored Pokémon has Mimic.
    int mimic = -1;
    for (int j = 0; j < 4; j++) {
      if (sd.active.moves[j].id != bk::Move::Mimic && sd.stored().moves[j].id == bk::Move::Mimic) {
        mimic = j;
        break;
      }
    }
    const bool from_mimic = mimic >= 0 && sd.active.moves[mimic].id == m;
    if (from_mimic) {
      for (PKMN::MoveSlot* ms : {&sd.active.moves[mimic], &sd.stored().moves[mimic]}) {
        ms->pp = uint8_t(std::max(0, ms->pp - int(spend)));
      }
    }
    // A lock's later turn (Rage, Thrash, a charge's second turn, a bind)
    // that a Metronome or Mirror Move started: the move isn't the Pokémon's,
    // and PP comes off the caller.
    const bool cont = (rage && v.rage()) || (thrashing && v.thrashing()) || (charging && v.charging()) ||
                      (binding && v.binding());
    const bool called = cont && via[s] != bk::Move::None;
    const bk::Move own = called ? via[s] : m;
    if (own != bk::Move::None && own != bk::Move::Struggle && !from_metronome && !from_mimic && !from_mirror) {
      if (!v.transform()) {
        if (!called) {
          reveal(sd.stored().moves, own);
          reveal(sd.active.moves, own);
        }
        for (PKMN::MoveSlot& ms : sd.stored().moves) {
          if (ms.id == own) ms.pp = uint8_t(std::max(0, ms.pp - int(spend)));
        }
      }
      for (PKMN::MoveSlot& ms : sd.active.moves) {
        if (ms.id == own) ms.pp = uint8_t(std::max(0, ms.pp - int(spend)));
      }
    }
    if (!cont) via[s] = from ? from_move : bk::Move::None;
    if ((rage || deduct || v.thrashing()) && !(charging && !v.charging())) {
      sd.last_used_move = m;
      sd.last_selected_move = m;
    }
    // The other side's selected slot: the move it chose, not one Metronome
    // or Mirror Move called (ours comes from our choice, `chose`).
    if (s != player && !from) {
      battle.last_moves[s].index = 1;
      for (uint8_t j = 0; j < 4; j++) {
        if (sd.active.moves[j].id == m) {
          battle.last_moves[s].index = uint8_t(j + 1);
          break;
        }
      }
    }
    if ((rage || deduct || v.thrashing() || charging) && !v.charging()) sd.last_selected_move = m;
    // A Mirror Move with nothing to copy fails, its selection None.
    if (m == bk::Move::MirrorMove) {
      if (const Obs* n = next(); n && n->type == ArgType::Fail && ident_player(n->u8(0)) == s) {
        sd.last_selected_move = bk::Move::None;
      }
    }
    if (rage || deduct || v.thrashing() || charging) {
      const bk::Move last = sd.last_selected_move;
      const D::MoveData md = bk::data(last);
      battle.last_moves[s].counterable =
          last != bk::Move::Counter && md.bp > 0 && (md.type == D::Type::Normal || md.type == D::Type::Fighting);
    }
    if (thrashing && !v.thrashing()) {
      v.set_thrashing(true);
      d.set_attacking(1);
    }
    if (charging && v.charging()) {
      v.set_charging(false);
      v.set_invulnerable(false);
    }
    if (missed) {
      v.set_binding(false);
      d.set_binding(0);
    } else {
      if (binding) {
        if (v.binding()) {
          d.set_binding(std::min(4, d.binding() + 1));
        } else {
          v.set_binding(true);
          d.set_binding(1);
          bind_shown[s] = from ? from_move : m;
        }
      }
      if (m == bk::Move::Bide) {
        if (v.bide()) {
          d.set_attacking(d.attacking() + 1);
        } else {
          v.set_bide(true);
        }
      }
      if (rage && !v.rage()) {
        v.set_rage(true);
        if (const Obs* n = next()) {
          const bool immune = n->type == ArgType::Immune;
          const bool broke = n->type == ArgType::End && n->u8(1) == bk::EndSubstitute;
          if (immune || broke) v.set_rage(false);
        }
      }
    }
    const D::Effect effect = bk::data(m).effect;
    const bool on_begin = uint8_t(effect) > 0 && uint8_t(effect) <= uint8_t(D::Effect::Transform);
    if (m == bk::Move::Roar || m == bk::Move::Whirlwind) battle.last_damage = 0;
    if (!still && !on_begin && m != bk::Move::Counter) battle.last_damage = 0;
  }

  // Adds `m` to the first empty slot unless it's there (oak's
  // complete_pokemon_moves), with its max PP.
  static void reveal(std::array<PKMN::MoveSlot, 4>& moves, bk::Move m) {
    for (const PKMN::MoveSlot& ms : moves) {
      if (ms.id == m) return;
    }
    for (PKMN::MoveSlot& ms : moves) {
      if (ms.id == bk::Move::None) {
        ms.id = m;
        ms.pp = D::max_pp(m);
        return;
      }
    }
  }

  void boost(const Obs& o) {
    const uint8_t s = side_of(o.u8(0)), opp = 1 - s;
    const uint8_t reason = o.u8(1);
    const int amount = int(o.u8(2)) - 6;
    if (reason == bk::BoostSpd) return;
    PKMN::ActivePokemon& a = side(s).active;
    if (amount > 0) {
      const PKMN::Stats& base = unmodified(s);
      const auto up = [&](int8_t cur, auto set, uint16_t from, uint16_t& to) {
        const int8_t b = int8_t(std::min(6, cur + amount));
        set(b);
        to = std::min<uint16_t>(999, bk::boosted(from, b));
      };
      switch (reason) {
      case bk::BoostRage:
      case bk::BoostAtk:
        up(a.boosts.atk(), [&](int8_t b) { a.boosts.set_atk(b); }, base.atk, a.stats.atk);
        if (reason == bk::BoostRage) return;
        break;
      case bk::BoostDef: up(a.boosts.def(), [&](int8_t b) { a.boosts.set_def(b); }, base.def, a.stats.def); break;
      case bk::BoostSpe: up(a.boosts.spe(), [&](int8_t b) { a.boosts.set_spe(b); }, base.spe, a.stats.spe); break;
      case bk::BoostSpa: up(a.boosts.spc(), [&](int8_t b) { a.boosts.set_spc(b); }, base.spc, a.stats.spc); break;
      case bk::BoostEva: a.boosts.set_eva(int8_t(std::min(6, a.boosts.eva() + amount))); break;
      case bk::BoostAcc: a.boosts.set_acc(int8_t(std::min(6, a.boosts.acc() + amount))); break;
      default: break;
      }
      // the glitch: the foe's paralysis or burn drop again
      bk::status_modify(uint8_t(side(opp).stored().status), side(opp).active.stats);
      return;
    }
    // A drop. Showdown sends no -fail when Rage's boost meets a drop at 999.
    if (reason == bk::BoostAtk) {
      if (const Obs* p = prev()) {
        if (p->type == ArgType::Boost && p->u8(1) == bk::BoostRage && ident_player(p->u8(0)) == s &&
            a.stats.atk == 999) {
          a.boosts.set_atk(int8_t(a.boosts.atk() - 1));
          unstore_stats();
          return;
        }
      }
    }
    const int by = -amount;
    const PKMN::Stats& base = unmodified(s);
    const auto down = [&](int8_t cur, auto set, uint16_t from, uint16_t& to) {
      const int8_t b = int8_t(std::max(-6, cur - by));
      set(b);
      to = std::max<uint16_t>(1, bk::boosted(from, b));
    };
    switch (reason) {
    case bk::BoostAtk: down(a.boosts.atk(), [&](int8_t b) { a.boosts.set_atk(b); }, base.atk, a.stats.atk); break;
    case bk::BoostDef: down(a.boosts.def(), [&](int8_t b) { a.boosts.set_def(b); }, base.def, a.stats.def); break;
    case bk::BoostSpe: down(a.boosts.spe(), [&](int8_t b) { a.boosts.set_spe(b); }, base.spe, a.stats.spe); break;
    case bk::BoostSpa: down(a.boosts.spc(), [&](int8_t b) { a.boosts.set_spc(b); }, base.spc, a.stats.spc); break;
    case bk::BoostAcc: a.boosts.set_acc(int8_t(std::max(-6, a.boosts.acc() - by))); break;
    case bk::BoostEva: a.boosts.set_eva(int8_t(std::max(-6, a.boosts.eva() - by))); break;
    default: break;
    }
    bk::status_modify(uint8_t(side(s).stored().status), a.stats);
  }

  void status(const Obs& o) {
    const uint8_t s = side_of(o.u8(0));
    PKMN::Side& sd = side(s);
    const uint8_t byte = o.u8(1);
    const bool rest = o.u8(2) == bk::STATUS_FROM && bk::Move(o.u8(3)) == bk::Move::Rest;
    if (rest) {
      sd.stored().status = D::Status(bk::REST);
      dur(s).set_sleep(0, 1);
    } else if (bk::is_sleep(byte)) {
      sd.stored().status = D::Status(bk::SLP);
      dur(s).set_sleep(0, 1);
      sd.active.volatiles.set_recharging(false);
    } else if (byte == bk::TOX) {
      sd.active.volatiles.set_toxic(true);
      sd.active.volatiles.set_toxic_counter(0);
      sd.stored().status = D::Status(bk::TOX);
    } else {
      sd.stored().status = D::Status(byte);
    }
    bk::status_modify(uint8_t(sd.stored().status), sd.active.stats);
    store_stats();
  }

  void curestatus(const Obs& o) {
    const uint8_t s = side_of(o.u8(0));
    const bool from_haze = prev() && prev()->type == ArgType::ClearAllBoost;
    if (bk::is_sleep(uint8_t(side(s).stored().status)) && !from_haze) {
      before_move(s);
      store_stats();
    }
    cure_status(s);
  }

  void cant(const Obs& o) {
    const uint8_t s = side_of(o.u8(0));
    PKMN::Side& sd = side(s);
    PKMN::Duration& d = dur(s);
    Why why = Why::None;
    switch (o.u8(1)) {
    case bk::CantRecharge: why = Why::Recharge; break;
    case bk::CantParalysis: why = Why::Paralysis; break;
    case bk::CantFreeze: why = Why::Freeze; break;
    case bk::CantSleep: {
      d.set_sleep(0, uint8_t(d.sleep(0) + 1));
      const uint8_t st = uint8_t(sd.stored().status);
      if (st & 0x80) sd.stored().status = D::Status(uint8_t(st - 1));
      why = Why::Sleep;
      break;
    }
    case bk::CantBound: why = Why::Bound; break;
    case bk::CantFlinch:
      sd.active.volatiles.set_recharging(false);
      why = Why::Flinch;
      break;
    case bk::CantDisable: why = Why::Disable; break;
    default: break;
    }
    before_move(s, why);
    store_stats();
  }

  void activate(const Obs& o) {
    const uint8_t s = side_of(o.u8(0)), opp = 1 - s;
    PKMN::Volatiles& v = side(s).active.volatiles;
    switch (o.u8(1)) {
    case bk::ActSubstitute: {
      // a hit the substitute took without breaking
      const uint16_t x = sub_damage(opp);
      const uint8_t hp = v.substitute_hp();
      bk::set_substitute_hp(v, uint8_t(hp - std::min<uint16_t>(hp ? hp - 1 : 0, x)));
      if (bk::drain(side(opp).last_used_move)) battle.last_damage = std::max<uint16_t>(1, battle.last_damage / 2);
      break;
    }
    case bk::ActConfusion: dur(s).set_confusion(uint8_t(dur(s).confusion() + 1)); break;
    case bk::ActBide:
      before_move(s);
      store_stats();
      dur(s).set_attacking(uint8_t(dur(s).attacking() + 1));
      break;
    case bk::ActMist: v.set_mist(true); break;
    default: break;
    }
  }

  void start(const Obs& o) {
    const uint8_t s = side_of(o.u8(0));
    PKMN::Side& sd = side(s);
    PKMN::ActivePokemon& a = sd.active;
    PKMN::Volatiles& v = a.volatiles;
    PKMN::Duration& d = dur(s);
    switch (o.u8(1)) {
    case bk::StartSubstitute:
      v.set_substitute(true);
      bk::set_substitute_hp(v, uint8_t(sd.stored().stats.hp / 4 + 1));
      break;
    case bk::StartReflect: v.set_reflect(true); break;
    case bk::StartLightScreen: v.set_light_screen(true); break;
    case bk::StartBide:
      v.set_bide(true);
      v.set_state(0);
      v.set_attacks(1);
      d.set_attacking(1);
      break;
    case bk::StartDisable: {
      const bk::Move m = bk::Move(o.u8(2));
      v.set_disable_left(1);
      d.set_disable(1);
      uint8_t slot = 0;
      for (uint8_t j = 0; j < 4 && !slot; j++) {
        if (a.moves[j].id == m) slot = uint8_t(j + 1);
      }
      if (!slot) {
        // a move not seen yet
        for (uint8_t j = 0; j < 4 && !slot; j++) {
          if (a.moves[j].id == bk::Move::None) slot = uint8_t(j + 1);
        }
      }
      v.set_disable_move(slot);
      break;
    }
    case bk::StartFocusEnergy: v.set_focus_energy(true); break;
    case bk::StartMimic: {
      const bk::Move m = bk::Move(o.u8(2));
      for (PKMN::MoveSlot& ms : a.moves) {
        if (ms.id == bk::Move::Mimic) {
          ms.id = m;
          break;
        }
      }
      break;
    }
    case bk::StartLeechSeed: v.set_leech_seed(true); break;
    case bk::StartConfusionSilent:
      v.set_thrashing(false);
      d.set_attacking(0);
      [[fallthrough]];
    case bk::StartConfusion:
      v.set_confusion(true);
      v.set_confusion_left(1);
      d.set_confusion(1);
      break;
    case bk::StartMist: v.set_mist(true); break;
    case bk::StartTypeChange: a.types = o.u8(2); break;
    default: break;
    }
  }

  void end(const Obs& o) {
    const uint8_t s = side_of(o.u8(0)), opp = 1 - s;
    PKMN::Volatiles& v = side(s).active.volatiles;
    PKMN::Duration& d = dur(s);
    switch (o.u8(1)) {
    case bk::EndSubstitute:
      sub_damage(opp);
      v.set_substitute(false);
      bk::set_substitute_hp(v, 0);
      break;
    case bk::EndDisable:
    case bk::EndDisableSilent:
      v.set_disable_left(0);
      v.set_disable_move(0);
      d.set_disable(0);
      break;
    case bk::EndBide:
    case bk::EndBideSilent:
      before_move(s, Why::Bide);
      store_stats();
      v.set_bide(false);
      v.set_attacks(0);
      d.set_attacking(0);
      break;
    case bk::EndConfusion:
    case bk::EndConfusionSilent:
      v.set_confusion(false);
      v.set_confusion_left(0);
      d.set_confusion(0);
      break;
    case bk::EndMistSilent: v.set_mist(false); break;
    case bk::EndReflectSilent: v.set_reflect(false); break;
    case bk::EndLightScreenSilent: v.set_light_screen(false); break;
    case bk::EndFocusEnergySilent: v.set_focus_energy(false); break;
    case bk::EndLeechSeedSilent:
    case bk::EndLeechSeedFrom: v.set_leech_seed(false); break;
    case bk::EndToxicSilent:
      v.set_toxic(false);
      v.set_toxic_counter(0);
      break;
    default: break;
    }
  }

  void transform(const Obs& o) {
    const uint8_t s = side_of(o.u8(0)), opp = 1 - s;
    PKMN::ActivePokemon& a = side(s).active;
    PKMN::ActivePokemon& b = side(opp).active;
    PKMN::Volatiles& v = a.volatiles;
    v.set_transform(true);
    const uint8_t t = b.volatiles.transform() ? b.volatiles.transform_species()
                                              : uint8_t((opp << 3) | side(opp).order[0]);
    bk::set_transform_species(v, t);
    a.stats.atk = b.stats.atk;
    a.stats.def = b.stats.def;
    a.stats.spe = b.stats.spe;
    a.stats.spc = b.stats.spc;
    a.species = b.species;
    a.types = b.types;
    a.boosts = b.boosts;
    // The other side's moves as far as they're known; our own request fills
    // ours (`request`).
    for (uint8_t j = 0; j < 4; j++) {
      a.moves[j].id = b.moves[j].id;
      a.moves[j].pp = b.moves[j].id != bk::Move::None ? 5 : 0;
    }
  }
};

} // namespace ataraxos
