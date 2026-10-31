// One player's view of libpkmn's omniscient Gen 1 log: what Showdown's
// stream for that player shows, rewritten in place (the bytes keep their
// sizes). libpkmn writes only the omniscient stream (PROTOCOL.md: "there is
// no |split| message"), so this is the driver code it asks for.
//
// - The other side's HP as Showdown's HP Percentage Mod shows it
//   (sim/pokemon.ts getHealth): ceil(100 * hp / max), 99 when not full,
//   over 100.
// - Every status byte as the text shows it: a sleep is "slp", its counter
//   and Rest's flag gone (Status 1). The request shows our own the same way
//   (request_data.hpp).
// - The other side's Pokémon numbered by first appearance (1-6), as a
//   player knows them, where libpkmn numbers them by party position, which
//   the player doesn't know. Our own keep their party position, which the
//   request gives.
//
// One View per player per battle: the numbering is its state.

#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>

#include "obs.hpp"

namespace ataraxos {

// libpkmn's PokemonIdent byte: player bit 3 (P2), party slot 1-6 below.
inline constexpr uint8_t IDENT_P2 = 0b1000;
inline constexpr uint8_t ident_player(uint8_t id) { return (id & IDENT_P2) ? 1 : 0; }
inline constexpr uint8_t ident_slot(uint8_t id) { return id & 0b111; }

// Showdown's status text: any sleep is "slp".
inline constexpr uint8_t view_status(uint8_t s) { return (s & 0b111) ? 1 : s; }

// Showdown's shared HP: percent, rounded up, 99 when not full.
inline constexpr uint16_t view_percent(uint16_t hp, uint16_t max) {
  if (max == 0 || hp == 0) return 0;
  uint32_t p = (100u * hp + max - 1) / max;
  if (p == 100 && hp < max) p = 99;
  return uint16_t(p);
}

struct View {
  uint8_t player = 0; // 0 P1, 1 P2: whose view
  std::array<uint8_t, 7> number{}; // the other side's party slot -> appearance (0 unseen)
  uint8_t seen = 0;

  View() = default;
  explicit View(uint8_t p) : player(p) {}

  // The other side's number for its party slot `slot`, assigned on first
  // sight.
  uint8_t number_of(uint8_t slot) {
    assert(slot >= 1 && slot <= 6);
    if (!number[slot]) number[slot] = ++seen;
    return number[slot];
  }

  // Rewrites one update's bytes (up to its null) in place; returns the
  // bytes used, the null not counted.
  size_t apply(std::span<uint8_t> bytes) {
    size_t pos = 0;
    while (pos < bytes.size()) {
      const ArgType type = ArgType(bytes[pos]);
      if (type == ArgType::None) break;
      uint8_t* d = &bytes[pos + 1];
      const size_t n = message_size(type, d);
      rewrite(type, d);
      pos += 1 + n;
    }
    return pos;
  }

 private:
  void ident(uint8_t& id) {
    if (id == 0 || ident_player(id) == player) return;
    id = uint8_t((id & ~0b111) | number_of(ident_slot(id)));
  }

  // hp at d[0..1], max at d[2..3] (little-endian), status at d[4]; `who`
  // the ident before renumbering.
  void health(uint8_t who, uint8_t* d) {
    d[4] = view_status(d[4]);
    if (ident_player(who) == player) return;
    const uint16_t hp = uint16_t(d[0] | d[1] << 8), max = uint16_t(d[2] | d[3] << 8);
    const uint16_t p = view_percent(hp, max), m = max ? 100 : 0;
    d[0] = uint8_t(p), d[1] = uint8_t(p >> 8);
    d[2] = uint8_t(m), d[3] = uint8_t(m >> 8);
  }

  void rewrite(ArgType type, uint8_t* d) {
    switch (type) {
    case ArgType::Move:
      ident(d[0]);
      ident(d[2]);
      break;
    case ArgType::Switch: {
      const uint8_t who = d[0];
      health(who, d + 3);
      ident(d[0]);
      break;
    }
    case ArgType::Damage:
    case ArgType::Heal: {
      const uint8_t who = d[0];
      health(who, d + 1);
      ident(d[0]);
      if (message_size(type, d) == 8) ident(d[7]);
      break;
    }
    case ArgType::SetHP: {
      const uint8_t who = d[0];
      health(who, d + 1);
      ident(d[0]);
      break;
    }
    case ArgType::Status:
    case ArgType::CureStatus:
      ident(d[0]);
      d[1] = view_status(d[1]);
      break;
    case ArgType::Start:
      ident(d[0]);
      if (d[1] == reason::START_TYPECHANGE) ident(d[3]);
      break;
    case ArgType::Transform:
    case ArgType::CopyBoost:
      ident(d[0]);
      ident(d[1]);
      break;
    case ArgType::Cant:
    case ArgType::Faint:
    case ArgType::Boost:
    case ArgType::Fail:
    case ArgType::Miss:
    case ArgType::HitCount:
    case ArgType::Prepare:
    case ArgType::MustRecharge:
    case ArgType::Activate:
    case ArgType::End:
    case ArgType::Crit:
    case ArgType::SuperEffective:
    case ArgType::Resisted:
    case ArgType::Immune:
    case ArgType::CureTeam:
    case ArgType::SetBoost:
    case ArgType::SingleMove:
    case ArgType::SingleTurn: ident(d[0]); break;
    default: break;
    }
  }
};

} // namespace ataraxos
