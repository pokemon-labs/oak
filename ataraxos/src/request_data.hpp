// Showdown's Gen 1 |request| JSON, typed: enums and numbers where the JSON
// has names. The ladder fills it from the JSON (client.hpp), and training
// from the true battle (`request_data`), so the network reads one struct
// either way.
//
// Showdown's fields (sim/side.ts getRequestData, sim/pokemon.ts
// getMoveRequestData, getSwitchRequestData, getMoves):
//
// - `side.pokemon` in Showdown's order, which puts the active Pokémon first
//   and swaps on every switch: libpkmn's slot order (Side::get). `stats` are
//   the stored stats (spa and spd both Special); `moves` the active's
//   current ones (Mimic's and Transform's) for the active, else the stored.
// - `active[0]` for a move request: each move slot's PP, max PP and
//   whether it's disabled, or one entry when the move is forced (a lock:
//   Thrash, Petal Dance, Bide, Rage, a charging or binding move; Recharge).
//   Gen 1 has no hidden-illegal choice: Disable is announced, and partial
//   trapping only stops the move during the turn. `maybeLocked` (the foe's
//   partial trap may hold us) isn't filled yet.
// - `forceSwitch` for a replacement, `wait` while the foe replaces.
//
// In training the choices are libpkmn's (pkmn_gen1_battle_choices in
// Showdown mode), and `locked`, `recharge` and `trapped` are read off them.

#pragma once

#include <array>
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
    // One forced move: the locked move, or Recharge (moves[0].id None).
    bool locked;
    bool recharge;
    bool trapped; // no switch though a Pokémon is left to switch to
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
  bool update; // Showdown's re-request after an error: never in Gen 1
  Active active; // Move_ only
  Pokemon side[6];
  uint8_t size; // side's Pokémon
};

// A choice byte (libpkmn's: type in bits 0-1, data above).
inline constexpr uint8_t choice_type(pkmn_choice c) { return c & 3; }
inline constexpr uint8_t choice_data(pkmn_choice c) { return c >> 2; }

// The request Showdown would send `player` (0 P1, 1 P2), whose request type
// is `type` (pkmn_result_p1/p2) and choices `choices`.
inline RequestData request_data(const PKMN::Battle& battle, uint8_t player, pkmn_choice_kind type,
                                std::span<const pkmn_choice> choices) {
  RequestData r{};
  r.kind = type == PKMN_CHOICE_MOVE     ? RequestData::Move_
           : type == PKMN_CHOICE_SWITCH ? RequestData::Switch
                                        : RequestData::Wait;
  const PKMN::Side& side = battle.sides[player];
  bool can_switch = false;
  for (uint8_t s = 1; s <= 6; s++) {
    if (side.order[s - 1] == 0) continue;
    const PKMN::Pokemon& p = side.get(s);
    if (p.species == PKMN::Data::Species::None) continue;
    can_switch |= s > 1 && p.hp != 0;
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
  uint8_t moves = 0, switches = 0;
  for (pkmn_choice c : choices) {
    moves += choice_type(c) == PKMN_CHOICE_MOVE;
    switches += choice_type(c) == PKMN_CHOICE_SWITCH;
  }
  act.trapped = can_switch && switches == 0;
  act.recharge = v.recharging();
  const bool forced = v.bide() || v.thrashing() || v.rage() || v.charging() || v.binding() || v.recharging();
  if (forced && moves == 1 && switches == 0) {
    act.locked = true;
    for (pkmn_choice c : choices) {
      const uint8_t i = choice_data(c);
      if (!act.recharge && i >= 1) act.moves[0].id = a.moves[i - 1].id;
    }
    return r;
  }
  for (uint8_t i = 0; i < 4; i++) {
    const PKMN::MoveSlot& m = a.moves[i];
    RequestData::ActiveMove& q = act.moves[i];
    q.id = m.id;
    if (m.id == PKMN::Data::Move::None) continue;
    q.pp = m.pp;
    q.maxpp = v.transform() ? 5 : PKMN::Data::max_pp(m.id);
    q.disabled = v.disable_left() && v.disable_move() == i + 1;
  }
  return r;
}

} // namespace ataraxos
