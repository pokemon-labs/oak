// Obs: one message of libpkmn's Gen 1 binary log (cpp/extern/engine's
// docs/PROTOCOL.md, as -Dshowdown -Dlog writes it): its ArgType and its
// fields, zero-padded to the longest message. An update is a
// std::vector<Obs>, its messages in order with no terminating null; a log is
// a sequence of updates.
//
// Obs don't know whose view the bytes are. The views differ only in the
// other side's HP (hp/max becomes percent/100), the status byte (sleep
// counters dropped) and the other side's numbering (view.hpp), so the same
// rules read every view. The fixed size is what lets a rule look back
// through the history, which the variable-size bytes can't be walked in.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace ataraxos {

// libpkmn's ArgType (cpp/include/libpkmn/log.h). Drag through Weather are
// Gen 2's and never appear.
enum class ArgType : uint8_t {
  None = 0x00,
  LastStill = 0x01,
  LastMiss = 0x02,
  Move = 0x03,
  Switch = 0x04,
  Cant = 0x05,
  Faint = 0x06,
  Turn = 0x07,
  Win = 0x08,
  Tie = 0x09,
  Damage = 0x0A,
  Heal = 0x0B,
  Status = 0x0C,
  CureStatus = 0x0D,
  Boost = 0x0E,
  ClearAllBoost = 0x0F,
  Fail = 0x10,
  Miss = 0x11,
  HitCount = 0x12,
  Prepare = 0x13,
  MustRecharge = 0x14,
  Activate = 0x15,
  FieldActivate = 0x16,
  Start = 0x17,
  End = 0x18,
  OHKO = 0x19,
  Crit = 0x1A,
  SuperEffective = 0x1B,
  Resisted = 0x1C,
  Immune = 0x1D,
  Transform = 0x1E,
  Drag = 0x1F,
  Item = 0x20,
  EndItem = 0x21,
  CureTeam = 0x22,
  SetHP = 0x23,
  SetBoost = 0x24,
  CopyBoost = 0x25,
  SideStart = 0x26,
  SideEnd = 0x27,
  SingleMove = 0x28,
  SingleTurn = 0x29,
  Weather = 0x2A,
};
inline constexpr size_t ARG_TYPES = 0x2B;

// The reasons that add bytes (PROTOCOL.md's tables).
namespace reason {
inline constexpr uint8_t MOVE_FROM = 0x01;   // Move: [from] a move (0x02 too)
inline constexpr uint8_t CANT_DISABLE = 0x05;
inline constexpr uint8_t DAMAGE_RECOIL = 0x05; // [of]
inline constexpr uint8_t HEAL_DRAIN = 0x02;    // [of]
inline constexpr uint8_t STATUS_FROM = 0x02;   // [from] a move
inline constexpr uint8_t START_TYPECHANGE = 0x09; // types, [of]
inline constexpr uint8_t START_DISABLE = 0x0A;    // the move
inline constexpr uint8_t START_MIMIC = 0x0B;      // the move
inline constexpr uint8_t SIDE_SPIKES = 0x03;      // SideEnd's [of]
} // namespace reason

// The bytes after the ArgType, given those bytes (a reason can add some).
// Checked against the engine on random games: the sizes walk every update
// exactly to its null.
constexpr size_t message_size(ArgType type, const uint8_t* d) {
  switch (type) {
  case ArgType::None:
  case ArgType::LastStill:
  case ArgType::LastMiss:
  case ArgType::Tie:
  case ArgType::ClearAllBoost:
  case ArgType::FieldActivate:
  case ArgType::OHKO: return 0;
  case ArgType::Faint:
  case ArgType::Win:
  case ArgType::Miss:
  case ArgType::MustRecharge:
  case ArgType::Crit:
  case ArgType::SuperEffective:
  case ArgType::Resisted:
  case ArgType::CureTeam: return 1;
  case ArgType::Turn:
  case ArgType::Fail:
  case ArgType::HitCount:
  case ArgType::Prepare:
  case ArgType::Activate:
  case ArgType::End:
  case ArgType::Immune:
  case ArgType::Transform:
  case ArgType::SetBoost:
  case ArgType::CopyBoost:
  case ArgType::SideStart:
  case ArgType::SingleMove:
  case ArgType::SingleTurn:
  case ArgType::Weather: return 2;
  case ArgType::CureStatus:
  case ArgType::Boost:
  case ArgType::Item:
  case ArgType::EndItem: return 3;
  case ArgType::Move: return 4 + (d[3] != 0);
  case ArgType::Switch: return 8;
  case ArgType::Cant: return 2 + (d[1] == reason::CANT_DISABLE);
  case ArgType::Damage: return 7 + (d[6] == reason::DAMAGE_RECOIL);
  case ArgType::Heal: return 7 + (d[6] == reason::HEAL_DRAIN);
  case ArgType::Status: return 3 + (d[2] == reason::STATUS_FROM);
  case ArgType::Start:
    return 2 + (d[1] == reason::START_TYPECHANGE ? 2
                : d[1] == reason::START_DISABLE || d[1] == reason::START_MIMIC ? 1
                                                                                : 0);
  case ArgType::SideEnd: return 2 + (d[1] == reason::SIDE_SPIKES);
  case ArgType::SetHP: return 7;
  case ArgType::Drag: return 9;
  }
  assert(false && "not a Gen 1 message");
  return 0;
}

// The longest Gen 1 message after its ArgType: Switch (Drag is Gen 2's).
inline constexpr size_t MAX_MESSAGE = 9;
// libpkmn's Gen 1 log bound (PROTOCOL.md, "Size": 180), rounded up.
inline constexpr size_t LOG_SIZE = 256;

struct Obs {
  static constexpr size_t MAX_SIZE = MAX_MESSAGE;

  ArgType type = ArgType::None;
  uint8_t data[MAX_SIZE] = {};

  constexpr size_t size() const { return message_size(type, data); }

  constexpr uint8_t u8(size_t at) const {
    assert(at < MAX_SIZE);
    return data[at];
  }
  constexpr uint16_t u16(size_t at) const {
    assert(at + 2 <= MAX_SIZE);
    return uint16_t(data[at] | data[at + 1] << 8);
  }

  friend constexpr bool operator==(const Obs&, const Obs&) = default;
};

static_assert(sizeof(Obs) == 10);
static_assert(std::is_trivially_copyable_v<Obs>);

// Appends one update's messages, up to the terminating null (or the end of
// `bytes`).
inline void split(std::span<const uint8_t> bytes, std::vector<Obs>& out) {
  size_t pos = 0;
  while (pos < bytes.size()) {
    Obs o;
    o.type = ArgType(bytes[pos++]);
    if (o.type == ArgType::None) return;
    const size_t n = message_size(o.type, bytes.data() + pos);
    assert(n <= Obs::MAX_SIZE && pos + n <= bytes.size());
    std::memcpy(o.data, bytes.data() + pos, n);
    pos += n;
    out.push_back(o);
  }
}

inline std::vector<Obs> split(std::span<const uint8_t> bytes) {
  std::vector<Obs> out;
  split(bytes, out);
  return out;
}

} // namespace ataraxos
