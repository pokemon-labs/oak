// The network's input: one Token per message of a player's view of
// libpkmn's Gen 1 binary log (view.hpp), and the request flattened to fixed
// int16 fields.
//
// A Token is a fixed row of int16 columns. The categorical ones index an
// embedding table each (0 is none); NUM, HP, MAX_HP and STATUS are numbers
// (STATUS the raw status byte). Pokémon are relative to the viewer: an ident
// column is 1 + (side << 3 | slot), side 0 the viewer's and 1 the foe's,
// slot as the view numbers it (ours by party position, the foe's by
// appearance), 0 for a side itself (Win).
//
// The layouts are obs.hpp's message_size; tokenize asserts it read exactly
// that.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "obs.hpp"
#include "request_data.hpp"
#include "view.hpp"

namespace ataraxos {

enum Column : uint8_t {
  KIND,    // ArgType
  SUB,     // the message's reason byte (Boost's stat, Start's effect, ...)
  ACTOR,   // ident
  OTHER,   // target, [of], source
  MOVE,
  MOVE2,   // Move's [from] move
  SPECIES,
  NUM,     // turn, boost stage + 6, hit count, level, Conversion's types
  HP,
  MAX_HP,
  STATUS,
  COLUMNS,
};

struct Token {
  int16_t c[COLUMNS];
};
static_assert(sizeof(Token) == 2 * COLUMNS);

namespace detail {

inline int16_t ident(uint8_t bits, uint8_t viewer) {
  if (bits == 0) return 0;
  const uint8_t side = ident_player(bits) == viewer ? 0 : 1;
  return int16_t(1 + ((side << 3) | ident_slot(bits)));
}

} // namespace detail

// Appends one Token per message of `bytes` (one update's view, up to its
// null) to `out`, as `viewer` (0 P1, 1 P2) sees them.
inline void tokenize(std::span<const uint8_t> bytes, uint8_t viewer, std::vector<Token>& out) {
  size_t i = 0;
  while (i < bytes.size()) {
    const ArgType type = ArgType(bytes[i]);
    if (type == ArgType::None) break;
    const uint8_t* d = &bytes[i + 1];
    const size_t size = message_size(type, d);
    Token t{};
    int16_t* c = t.c;
    c[KIND] = int16_t(type);
    const auto id = [&](Column col, size_t at) { c[col] = detail::ident(d[at], viewer); };
    const auto health = [&](size_t at) {
      c[HP] = int16_t(d[at] | d[at + 1] << 8);
      c[MAX_HP] = int16_t(d[at + 2] | d[at + 3] << 8);
      c[STATUS] = d[at + 4];
    };
    switch (type) {
    case ArgType::LastStill:
    case ArgType::LastMiss:
    case ArgType::Tie:
    case ArgType::ClearAllBoost:
    case ArgType::FieldActivate:
    case ArgType::OHKO: break;
    case ArgType::Faint:
    case ArgType::Miss:
    case ArgType::MustRecharge:
    case ArgType::Crit:
    case ArgType::SuperEffective:
    case ArgType::Resisted:
    case ArgType::CureTeam: id(ACTOR, 0); break;
    case ArgType::Win:
      // The winner's player number (1 or 2).
      c[ACTOR] = int16_t(1 + ((d[0] - 1 == viewer ? 0 : 1) << 3));
      break;
    case ArgType::Turn: c[NUM] = int16_t(d[0] | d[1] << 8); break;
    case ArgType::Move:
      id(ACTOR, 0);
      c[MOVE] = d[1];
      id(OTHER, 2);
      c[SUB] = d[3];
      if (d[3]) c[MOVE2] = d[4];
      break;
    case ArgType::Switch:
      id(ACTOR, 0);
      c[SPECIES] = d[1];
      c[NUM] = d[2]; // level
      health(3);
      break;
    case ArgType::Cant:
      id(ACTOR, 0);
      c[SUB] = d[1];
      if (d[1] == reason::CANT_DISABLE) c[MOVE] = d[2];
      break;
    case ArgType::Damage:
    case ArgType::Heal:
    case ArgType::SetHP:
      id(ACTOR, 0);
      health(1);
      c[SUB] = d[6];
      if (size == 8) id(OTHER, 7);
      break;
    case ArgType::Status:
      id(ACTOR, 0);
      c[STATUS] = d[1];
      c[SUB] = d[2];
      if (d[2] == reason::STATUS_FROM) c[MOVE] = d[3];
      break;
    case ArgType::CureStatus:
      id(ACTOR, 0);
      c[STATUS] = d[1];
      c[SUB] = d[2];
      break;
    case ArgType::Boost:
      id(ACTOR, 0);
      c[SUB] = d[1];
      c[NUM] = d[2];
      break;
    case ArgType::Fail:
    case ArgType::Activate:
    case ArgType::End:
    case ArgType::Immune:
      id(ACTOR, 0);
      c[SUB] = d[1];
      break;
    case ArgType::HitCount:
    case ArgType::SetBoost:
      id(ACTOR, 0);
      c[NUM] = d[1];
      break;
    case ArgType::Prepare:
    case ArgType::SingleMove:
    case ArgType::SingleTurn:
      id(ACTOR, 0);
      c[MOVE] = d[1];
      break;
    case ArgType::Start:
      id(ACTOR, 0);
      c[SUB] = d[1];
      if (d[1] == reason::START_TYPECHANGE) {
        c[NUM] = d[2];
        id(OTHER, 3);
      } else if (d[1] == reason::START_DISABLE || d[1] == reason::START_MIMIC) {
        c[MOVE] = d[2];
      }
      break;
    case ArgType::Transform:
    case ArgType::CopyBoost:
      id(ACTOR, 0);
      id(OTHER, 1);
      break;
    default: assert(false && "not a Gen 1 message");
    }
    out.push_back(t);
    i += 1 + size;
  }
}

// The request as fixed int16 fields, in this order:
//   header (7): kind, update, locked, recharge, trapped, maybe_locked, size
//   active moves (4 x 4): id, pp, maxpp, disabled
//   side (6 x 14): species, level, hp, max_hp, status, active, atk, def,
//     spe, spc, 4 moves
inline constexpr size_t REQUEST_HEADER = 7;
inline constexpr size_t MOVE_FIELDS = 4;
inline constexpr size_t POKEMON_FIELDS = 14;
inline constexpr size_t REQUEST_FIELDS = REQUEST_HEADER + 4 * MOVE_FIELDS + 6 * POKEMON_FIELDS;

inline void encode(const RequestData& q, int16_t* out) {
  int16_t* o = out;
  *o++ = q.kind;
  *o++ = q.update;
  *o++ = q.active.locked;
  *o++ = q.active.recharge;
  *o++ = q.active.trapped;
  *o++ = q.active.maybe_locked;
  *o++ = q.size;
  for (const RequestData::ActiveMove& m : q.active.moves) {
    *o++ = int16_t(m.id);
    *o++ = m.pp;
    *o++ = m.maxpp;
    *o++ = m.disabled;
  }
  for (uint8_t s = 0; s < 6; s++) {
    const RequestData::Pokemon& p = q.side[s];
    *o++ = int16_t(p.species);
    *o++ = p.level;
    *o++ = int16_t(p.hp);
    *o++ = int16_t(p.max_hp);
    *o++ = p.status;
    *o++ = p.active;
    *o++ = int16_t(p.stats.atk);
    *o++ = int16_t(p.stats.def);
    *o++ = int16_t(p.stats.spe);
    *o++ = int16_t(p.stats.spc);
    for (PKMN::Data::Move m : p.moves) *o++ = int16_t(m);
  }
  assert(size_t(o - out) == REQUEST_FIELDS);
}

} // namespace ataraxos
