// Showdown's Gen 1 |request| JSON, typed: enums and numbers where the JSON
// has names. The ladder fills it from the JSON (client.hpp), and training
// from the true battle (`request_data`), so the network reads one struct
// either way. Training follows Showdown's request (the user's call,
// 2026-10-31): the choices offered are Showdown's (`options`), and
// BattleData maps each to libpkmn's.
//
// Showdown's fields (sim/side.ts getRequestData, sim/pokemon.ts
// getMoveRequestData, getSwitchRequestData, getMoves; data/mods/gen1):
//
// - `side.pokemon` in Showdown's order, which puts the active Pokémon first
//   and swaps on every switch: libpkmn's slot order (Side::get). `stats` are
//   the stored stats (spa and spd both Special); `moves` the active's
//   current ones (Mimic's and Transform's) for the active, else the stored.
// - `active[0]` for a move request, one of:
//   - a hard lock (Thrash, Petal Dance, Rage, a charging move; Recharge):
//     one entry, the move (Showdown's `id`, "recharge"), and `trapped`;
//   - a semi-lock (Bide; a binding move's second turn, while the bind isn't
//     `maybeLocked`): one entry, no PP, switches allowed;
//   - "Fight" (asleep, frozen, or bound by the foe's first bind turn): one
//     entry, switches allowed; any move does the same, as the Pokémon
//     can't move;
//   - Struggle, when no move is left: one entry, switches allowed;
//   - else each move slot's PP, max PP and `disabled` (Disable's slot, or
//     no PP), and `maybeLocked`: a bind's later turns, and the turn after
//     one ends on its own unless it lasted 5, where Showdown hides whether
//     it goes on (data/mods/gen1/conditions.ts partialtrappinglock,
//     partiallytrapped, fakepartiallytrapped). Then any move is the bind
//     if it goes on.
// - `forceSwitch` for a replacement, `wait` while the foe replaces.
//
// libpkmn's state doesn't hold the bind's turns, so Binds tracks them over
// the updates and hands each request a Bind.
#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <span>

#include <libpkmn/data.h>

#include "view.hpp"

namespace ataraxos {

// PKMN_GEN1_MAX_CHOICES (an extern, so not a constant): 4 moves, 5 switches.
inline constexpr size_t MAX_CHOICES = 9;

struct RequestData {
  enum Kind : uint8_t { Move_, Switch, Wait };

  struct ActiveMove {
    PKMN::Data::Move id;
    uint8_t pp;
    uint8_t maxpp;
    bool disabled;
  };

  struct Active {
    ActiveMove moves[4];
    // One entry (moves[0]): the locked move, Bide or the binding move,
    // Struggle, or (id None) Recharge or Fight.
    bool locked;
    bool recharge;
    bool trapped; // a hard lock: no switch
    bool maybe_locked;
  };

  struct Pokemon {
    PKMN::Data::Species species;
    uint8_t level;
    uint16_t hp;
    uint16_t max_hp;
    uint8_t status; // as the text shows it (view.hpp's view_status)
    bool active;
    PKMN::Stats stats; // stored; `hp` is max_hp
    PKMN::Data::Move moves[4];
  };

  Kind kind;
  bool update; // Showdown's re-request after an error: never here
  Active active; // Move_ only
  Pokemon side[6];
  uint8_t size; // side's Pokémon
};

// A choice byte (libpkmn's: type in bits 0-1, data above).
inline constexpr uint8_t choice_type(pkmn_choice c) { return c & 3; }
inline constexpr uint8_t choice_data(pkmn_choice c) { return c >> 2; }

// Showdown's partial trapping as `player`'s request shows it.
struct Bind {
  bool maybe_locked;
  bool fight; // bound by the foe's first bind turn
};

// Per player, the current bind's length (0 none; libpkmn's `attacks` is
// what's left of it, set to the length less one when it starts, and a
// switch-in ends the foe's bind, so a bind on a new foe is a new one) and
// whether one just ended on its own (Showdown's fakepartiallytrapped, for
// the next turn's request).
struct Binds {
  uint8_t length[2] = {0, 0};
  bool ended[2] = {false, false};

  // After an update from `before` to `now`.
  void update(const PKMN::Battle& before, const PKMN::Battle& now) {
    const bool turn = now.turn != before.turn; // not a replacement's update
    for (uint8_t p = 0; p < 2; p++) {
      const PKMN::Side &s0 = before.sides[p], &s1 = now.sides[p];
      const PKMN::Side &f0 = before.sides[1 - p], &f1 = now.sides[1 - p];
      const bool was = s0.active.volatiles.binding(), is = s1.active.volatiles.binding();
      const bool same = s0.order[0] == s1.order[0] && f0.order[0] == f1.order[0];
      if (turn) ended[p] = false;
      if (is && (!was || f0.order[0] != f1.order[0])) {
        length[p] = uint8_t(s1.active.volatiles.attacks() + 1);
      } else if (!is) {
        if (was && turn && same && s1.active.volatiles.attacks() == 0 && s1.get(1).hp && f1.get(1).hp) {
          ended[p] = length[p] != 5;
        }
        length[p] = 0;
      }
    }
  }

  // `p`'s request's Bind: a bind's later turns and the turn after it ends
  // on its own hide whether p (binding) is locked, and so p's foe (bound)
  // too unless asleep, frozen or in Bide (their DisableMove undoes it); a
  // bind's first turn shows the bound one "Fight".
  Bind of(const PKMN::Battle& battle, uint8_t p) const {
    const PKMN::Side& side = battle.sides[p];
    const PKMN::Side& foe = battle.sides[1 - p];
    const uint8_t status = uint8_t(side.get(1).status);
    const auto turns = [&](const PKMN::Side& s, uint8_t q) {
      return s.active.volatiles.binding() ? length[q] - s.active.volatiles.attacks() : 0;
    };
    const bool undone = PKMN::Data::is_sleep(status) || status == uint8_t(PKMN::Data::Status::Freeze) ||
                        side.active.volatiles.bide();
    Bind b{};
    b.maybe_locked = turns(side, p) >= 2 || ended[p] || ((turns(foe, 1 - p) >= 2 || ended[1 - p]) && !undone);
    b.fight = turns(foe, 1 - p) == 1;
    return b;
  }
};

inline bool hard_locked(const PKMN::Volatiles& v) { return v.recharging() || v.rage() || v.thrashing() || v.charging(); }

// The request Showdown would send `player` (0 P1, 1 P2), whose request type
// is `type` (pkmn_result_p1/p2).
inline RequestData request_data(const PKMN::Battle& battle, uint8_t player, pkmn_choice_kind type, Bind bind) {
  RequestData r{};
  r.kind = type == PKMN_CHOICE_MOVE     ? RequestData::Move_
           : type == PKMN_CHOICE_SWITCH ? RequestData::Switch
                                        : RequestData::Wait;
  const PKMN::Side& side = battle.sides[player];
  for (uint8_t s = 1; s <= 6; s++) {
    if (side.order[s - 1] == 0) continue;
    const PKMN::Pokemon& p = side.get(s);
    if (p.species == PKMN::Data::Species::None) continue;
    RequestData::Pokemon& q = r.side[r.size++];
    q.species = p.species;
    q.level = p.level;
    q.hp = p.hp;
    q.max_hp = p.stats.hp;
    q.status = view_status(uint8_t(p.status));
    q.active = s == 1;
    q.stats = p.stats;
    for (uint8_t i = 0; i < 4; i++) q.moves[i] = s == 1 ? side.active.moves[i].id : p.moves[i].id;
  }
  if (r.kind != RequestData::Move_) return r;

  RequestData::Active& act = r.active;
  const PKMN::ActivePokemon& a = side.active;
  const PKMN::Volatiles& v = a.volatiles;
  const uint8_t status = uint8_t(side.get(1).status);
  if (hard_locked(v)) {
    act.locked = act.trapped = true;
    act.recharge = v.recharging();
    if (!act.recharge) act.moves[0].id = side.last_selected_move;
    return r;
  }
  if (v.bide() || (v.binding() && !bind.maybe_locked)) {
    act.locked = true;
    act.moves[0].id = v.bide() ? PKMN::Data::Move::Bide : side.last_selected_move;
    return r;
  }
  if (PKMN::Data::is_sleep(status) || status == uint8_t(PKMN::Data::Status::Freeze) || bind.fight) {
    act.locked = true;
    return r;
  }
  bool any = false;
  for (uint8_t i = 0; i < 4; i++) {
    const PKMN::MoveSlot& m = a.moves[i];
    RequestData::ActiveMove& q = act.moves[i];
    q.id = m.id;
    if (m.id == PKMN::Data::Move::None) continue;
    q.pp = m.pp;
    q.maxpp = v.transform() ? 5 : PKMN::Data::max_pp(m.id);
    q.disabled = m.pp == 0 || (v.disable_left() && v.disable_move() == i + 1);
    any |= !q.disabled;
  }
  if (!any) {
    act = RequestData::Active{};
    act.locked = true;
    act.moves[0].id = PKMN::Data::Move::Struggle;
    return r;
  }
  act.maybe_locked = bind.maybe_locked;
  return r;
}

// Showdown's choices for `r`, as choice bytes: "switch s" as switch(s) for
// each Pokémon left on the bench (unless trapped), then "move i" as move(i)
// for each entry not disabled (move(1) for a one-entry request); a pass
// when there's nothing to choose. Returns how many.
inline uint8_t showdown_choices(const RequestData& r, pkmn_choice* out) {
  uint8_t n = 0;
  const auto choice = [](uint8_t type, uint8_t data) { return pkmn_choice(type | data << 2); };
  if (r.kind == RequestData::Wait) {
    out[n++] = 0;
    return n;
  }
  if (!(r.kind == RequestData::Move_ && r.active.trapped)) {
    for (uint8_t s = 2; s <= r.size; s++) {
      if (r.side[s - 1].hp != 0) out[n++] = choice(PKMN_CHOICE_SWITCH, s);
    }
  }
  if (r.kind == RequestData::Switch) {
    if (n == 0) out[n++] = 0;
    return n;
  }
  if (r.active.locked) {
    out[n++] = choice(PKMN_CHOICE_MOVE, 1);
    return n;
  }
  for (uint8_t i = 0; i < 4; i++) {
    const RequestData::ActiveMove& m = r.active.moves[i];
    if (m.id != PKMN::Data::Move::None && !m.disabled) out[n++] = choice(PKMN_CHOICE_MOVE, i + 1);
  }
  return n;
}

// libpkmn's choice for Showdown's `option`, given libpkmn's `legal` ones:
// the same choice when it's legal, else (a move: Fight, a one-entry lock
// libpkmn names by its slot or as Struggle, a hidden bind going on) the
// first legal move, which does the same.
inline pkmn_choice engine_choice(pkmn_choice option, std::span<const pkmn_choice> legal) {
  for (pkmn_choice c : legal) {
    if (c == option) return c;
  }
  assert(choice_type(option) == PKMN_CHOICE_MOVE);
  for (pkmn_choice c : legal) {
    if (choice_type(c) == PKMN_CHOICE_MOVE) return c;
  }
  assert(false && "no legal move for a move option");
  return legal[0];
}

} // namespace ataraxos
