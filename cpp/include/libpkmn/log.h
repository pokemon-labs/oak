#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <libpkmn/strings.h>

namespace PKMN::Protocol {

enum Opcode : uint8_t {
  null = 0x00,
  laststill = 0x01,
  lastmiss = 0x02,
  move = 0x03,
  switch_ = 0x04,
  cant = 0x05,
  faint = 0x06,
  turn = 0x07,
  win = 0x08,
  tie = 0x09,
  damage = 0x0A,
  heal = 0x0B,
  status = 0x0C,
  curestatus = 0x0D,
  boost = 0x0E,
  clearallboost = 0x0F,
  fail = 0x10,
  miss = 0x11,
  hitcount = 0x12,
  prepare = 0x13,
  mustrecharge = 0x14,
  activate = 0x15,
  fieldactivate = 0x16,
  start = 0x17,
  end = 0x18,
  ohko = 0x19,
  crit = 0x1A,
  supereffective = 0x1B,
  resisted = 0x1C,
  immune = 0x1D,
  transform = 0x1E,
  drag = 0x1F,
  item = 0x20,
  enditem = 0x21,
  cureteam = 0x22,
  sethp = 0x23,
  setboost = 0x24,
  copyboost = 0x25,
  sidestart = 0x26,
  sideend = 0x27,
  singlemove = 0x28,
  singleturn = 0x29,
  weather = 0x2A,
};

enum class View : std::underlying_type_t<std::byte> {
  omnicient = 0,
  p1 = 1,
  p2 = 2,
};

struct PokemonIdent {
  uint8_t player; // 1 or 2
  uint8_t slot;   // 1..6
  View view() const noexcept { return static_cast<View>(player); }
};

inline PokemonIdent decode_ident(uint8_t b) {
  PokemonIdent id;
  id.player = (b & 0b00001000) ? 2 : 1;
  id.slot = (b & 0b00000111);
  return id;
}

inline std::string ident_to_string(const PokemonIdent &id) {
  return "p" + std::to_string(id.player) + "a";
}

inline std::string ident_to_string(const PKMN::Battle &battle,
                                   const PokemonIdent &id) {
  const auto species = battle.sides[id.player - 1].pokemon[id.slot - 1].species;
  return "p" + std::to_string(id.player) +
         "a: " + PKMN::species_string(species);
}

std::string cant_reason(uint8_t reason) {
  switch (reason) {
  case 0x00:
    return "slp";
  case 0x01:
    return "frz";
  case 0x02:
    return "par";
  case 0x03:
    return "partiallytrapped";
  case 0x04:
    return "flinch";
  case 0x05:
    return "Disable";
  case 0x06:
    return "recharge";
  case 0x07:
    return "nopp";
  default:
    assert(false);
  }
  return "";
}

std::string damage_reason(uint8_t reason) {
  switch (reason) {
  case 0x00:
    return "None";
  case 0x01:
    return "psn";
  case 0x02:
    return "brn";
  case 0x03:
    return "confusion";
  case 0x04:
    return "Leech Seed";
  case 0x05:
    return "Recoil|[of]";
  case 0x06:
    return "[from] Spikes";
  default:
    assert(false);
  }
  return "";
}

inline std::string status_string(const auto status) {
  const auto byte = static_cast<uint8_t>(status);
  if (byte == 0) {
    return "";
  }
  if (byte & 7) {
    return "slp";
  }
  switch (byte) {
  case 0b00001000:
    return "psn";
  case 0b00010000:
    return "brn";
  case 0b00100000:
    return "frz";
  case 0b01000000:
    return "par";
  case 0b10001000:
    return "tox";
  default:
    assert(false);
    return "";
  };
}

template <View view = View::omnicient> struct Parser {
  const unsigned char *buf;
  size_t pos = 0;

  std::vector<std::string> log;
  std::optional<size_t> last_move_index;
  pkmn_gen1_battle battle;

  Parser(const unsigned char *b) : buf(b), battle{} {}

  uint8_t peek_u8() const { return buf[pos]; }

  auto read_u8() { return buf[pos++]; }

  uint16_t read_u16() {
    uint16_t lo = buf[pos++];
    uint16_t hi = buf[pos++];
    return lo | (hi << 8);
  }

  void push(const std::string &s) {
    log.push_back(s);
    std::cout << s << std::endl;
  }

  void annotate_last_move(const std::string &suffix) {
    if (last_move_index) {
      log[*last_move_index] += suffix;
      last_move_index.reset();
    }
  }

  void possibly_hide_hp(const PokemonIdent identity, uint16_t &hp,
                        uint16_t &max_hp) const noexcept {
    if constexpr (view != View::omnicient) {
      if (identity.view() != view && max_hp) {
        hp = 100 * hp / max_hp;
        max_hp = 100;
      }
    }
  }

  std::string condition_hp_string(uint16_t hp, uint16_t max_hp) {
    if ((hp == 0) || (max_hp == 0)) {
      return "0";
    } else {
      return std::to_string(hp) + "/" + std::to_string(max_hp);
    }
  }

  std::string condition_string(uint16_t hp, uint16_t max_hp, uint8_t status) {
    if ((hp == 0) || (max_hp == 0)) {
      return "0 fnt";
    } else {
      return std::to_string(hp) + "/" + std::to_string(max_hp) + " " +
             status_string(status);
    }
  }

  void parse() {
    while (true) {
      const auto opcode = static_cast<Opcode>(read_u8());

      switch (opcode) {
      case Opcode::null: {
        return;
      }
      case Opcode::lastmiss: {
        push("|lastmiss");
        break;
      }
      case Opcode::laststill: {
        push("|laststill");
        break;
      }
      case Opcode::move: {
        // |move|p1a: Jynx|Blizzard|p2a: Chansey|[miss]
        // |move|p1a: Cloyster|Clamp|p2a: Alakazam|[from] Clamp
        auto source = read_u8();
        auto move = read_u8();
        auto target = read_u8();
        auto reason = read_u8();
        auto from = 0;
        if (reason == 0x02) {
          from = read_u8();
          push("|move|" +
               ident_to_string(PKMN::view(battle), decode_ident(source)) + "|" +
               PKMN::move_string(move) + "|" +
               ident_to_string(PKMN::view(battle), decode_ident(target)) +
               "|[from] " + PKMN::move_string(from));
        } else {
          push("|move|" +
               ident_to_string(PKMN::view(battle), decode_ident(source)) + "|" +
               PKMN::move_string(move) + "|" +
               ident_to_string(PKMN::view(battle), decode_ident(target)));
        }
        break;
      }
      case Opcode::switch_: {
        // |switch|p1a: Starmie|Starmie|247/323
        // |switch|p2a: Starmie|Starmie|81/100
        auto ident = read_u8();
        auto species = read_u8();
        auto level = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();

        auto identity = decode_ident(ident);
        possibly_hide_hp(identity, hp, max_hp);
        std::string level_string =
            level == 100 ? "" : (std::to_string(level) + "|");
        push("|switch|" + ident_to_string(PKMN::view(battle), identity) + "|" +
             PKMN::species_string(species) + "|" + level_string +
             std::to_string(hp) + "/" + std::to_string(max_hp));
        break;
      }
      case Opcode::cant: {
        auto ident = read_u8();
        auto reason = read_u8();
        push("|cant|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)) + "|" +
             cant_reason(reason));
        break;
      }
      case Opcode::faint: {
        auto ident = read_u8();
        push("|faint|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::turn: {
        auto turn = read_u16();
        push("|turn|" + std::to_string(turn));
        break;
      }
      case Opcode::win: {
        auto player = read_u8();
        push("|win|p" + std::to_string(player));
        break;
      }
      case Opcode::tie: {
        push("|tie|");
        break;
      }
      case Opcode::damage: {
        // |-damage|p2a: Exeggutor|0 fnt
        // |-damage|p1a: Tauros|91/353
        // |-damage|p2a: Starmie|18/100 slp
        // |-damage|p1a: Jolteon|157/333|[from] Recoil|[of] p2a: Snorlax
        auto ident = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();

        auto identity = decode_ident(ident);
        possibly_hide_hp(identity, hp, max_hp);

        uint8_t of;
        if (reason == 0x05) {
          of = read_u8();
        }
        push("|-damage|" + ident_to_string(PKMN::view(battle), identity) + "|" +
             condition_string(hp, max_hp, status));
        break;
      }
      case Opcode::heal: {
        auto ident = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();
        uint8_t of;
        if (reason == 0x02) {
          of = read_u8();
        }
        push("|heal|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::status: {
        auto ident = read_u8();
        auto status = read_u8();
        auto reason = read_u8();

        auto identity = decode_ident(ident);

        uint8_t from;
        if (reason == 0x02) {
          from = read_u8();
        }
        push("|-status|" + ident_to_string(PKMN::view(battle), identity) + "|" +
             status_string(status));
        break;
      }
      case Opcode::curestatus: {
        auto ident = read_u8();
        auto status = read_u8();
        auto reason = read_u8();
        push("|curestatus|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::boost: {
        auto ident = read_u8();
        auto reason = read_u8();
        auto num = read_u8();
        push("|boost|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::clearallboost: {
        push("|clearallboost|");
        break;
      }
      case Opcode::fail: {
        auto ident = read_u8();
        auto reason = read_u8();
        push("|fail|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::miss: {
        auto ident = read_u8();
        push("|miss|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::hitcount: {
        auto ident = read_u8();
        auto num = read_u8();
        push("|hitcount|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::prepare: {
        auto ident = read_u8();
        auto move = read_u8();
        push("|prepare|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::mustrecharge: {
        auto ident = read_u8();
        push("|mustrecharge|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::activate: {
        auto ident = read_u8();
        auto reason = read_u8();
        push("|activate|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::fieldactivate: {
        push("|fieldactivate|");
        break;
      }
      case Opcode::start: {
        auto ident = read_u8();
        auto reason = read_u8();
        uint8_t move_type;
        uint8_t of;

        if (reason == 0x09) {
          move_type = read_u8();
        } else if (reason == 0x0A) {
          move_type = read_u8();
        } else if (reason == 0x0B) {
          move_type = read_u8();
        }
        push("|start|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::end: {
        auto ident = read_u8();
        auto reason = read_u8();
        push("|end|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::ohko: {
        push("|ohko|");
        break;
      }
      case Opcode::crit: {
        auto ident = read_u8();
        push("|crit|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::supereffective: {
        auto ident = read_u8();
        push("|supereffective|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::resisted: {
        auto ident = read_u8();
        push("|resisted|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::immune: {
        auto ident = read_u8();
        auto reason = read_u8(); // 0x00 none, 0x01 ohko
        push("|immune|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::transform: {
        auto source = read_u8();
        auto target = read_u8();
        push("|transform|" + std::to_string(source));
        break;
      }
      case Opcode::drag: {
        auto ident = read_u8();
        auto species = read_u8();
        auto gender = read_u8();
        auto level = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        push("|drag|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::item: {
        auto target = read_u8();
        auto item = read_u8();
        auto source = read_u8();
        push("|item|" + std::to_string(target));
        break;
      }
      case Opcode::enditem: {
        auto target = read_u8();
        auto item = read_u8();
        auto source = read_u8();
        push("|enditem|" + std::to_string(target));
        break;
      }
      case Opcode::cureteam: {
        auto ident = read_u8();
        push("|cureteam|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::sethp: {
        auto ident = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();
        push("|sethp|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::setboost: {
        auto ident = read_u8();
        auto num = read_u8();
        push("|setboost|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::copyboost: {
        auto source = read_u8();
        auto target = read_u8();
        push("|copyboost|" + std::to_string(source));
        break;
      }
      case Opcode::sidestart: {
        auto player = read_u8();
        auto reason = read_u8();
        push("|sidestart|" + std::to_string(player));
        break;
      }
      case Opcode::sideend: {
        auto player = read_u8();
        auto reason = read_u8();
        uint8_t of;
        if (reason == 0x03) {
          of = read_u8();
        }
        push("|sideend|" + std::to_string(player));
        break;
      }
      case Opcode::singlemove: {
        auto ident = read_u8();
        auto move = read_u8();
        push("|singlemove|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::singleturn: {
        auto ident = read_u8();
        auto move = read_u8();
        push("|singleturn|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case Opcode::weather: {
        auto weather = read_u8();
        auto reason = read_u8();
        push("|weather|" + std::to_string(weather));
        break;
      }
      default: {
        std::cout << "ERROR: " << std::to_string(opcode) << std::endl;
        assert(false);
      }
      }
    }
  }
};

} // namespace PKMN::Protocol