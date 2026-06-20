#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <vector>

#include <libpkmn/strings.h>

namespace PKMN::Log {

enum ArgType : uint8_t {
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
  omniscient = 0,
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
    return "";
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

std::string heal_reason(uint8_t reason) {
  switch (reason) {
  case 0x00:
    return "";
  case 0x01:
    return "|[silent]";
  case 0x02:
    return "|[from] drain|";
  case 0x03:
    assert(false);
    return "|[from] Leftovers";
  default:
    assert(false);
  }
  return "";
}

std::string boost_reason(uint8_t reason) {
  switch (reason) {
  case 0x00:
    return "atk|[from] Rage";
  case 0x01:
    return "atk";
  case 0x02:
    return "def";
  case 0x03:
    return "spe";
  case 0x04:
    return "spa";
  case 0x05:
    return "spd";
  case 0x06:
    return "accuracy";
  case 0x07:
    return "evasion";
  default:
    assert(false);
  }
  return "";
}

std::string activate_reason(uint8_t reason) {
  switch (reason) {
  case 0x00:
    return "Bide";
  case 0x01:
    return "confusion";
  case 0x02:
    return "move: Haze";
  case 0x03:
    return "move: Mist"; // TODO
  case 0x04:
    return "move: Struggle";
  case 0x05:
    return "Substitute|";
  case 0x06:
    return "||move: Splash";
  default:
    assert(false);
  }
  return "";
}

std::string start_reason(uint8_t reason) {
  switch (reason) {
  case 0x00:
    return "Bide";
  case 0x01:
    return "confusion";
  case 0x02:
    return "confusion|[silent]";
  case 0x03:
    return "move: Focus Energy";
  case 0x04:
    return "move: Leech Seed";
  case 0x05:
    return "Light Screen";
  case 0x06:
    return "Mist";
  case 0x07:
    return "Reflect";
  case 0x08:
    return "Substitute";
  case 0x09:
    return "typechange|";
  case 0x0A:
    return "Disable|";
  case 0x0B:
    return "Mimic|";
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

std::string end_reason(uint8_t reason) {
  assert(reason < 0x04 || reason > 0x0E);
  switch (reason) {
  case 0x00:
    return "Disable";
  case 0x01:
    return "confusion";
  case 0x02:
    return "Bide";
  case 0x03:
    return "Substitute";
  // Gen 2+
  case 0x04:
    return "Nightmare";
  case 0x05:
    return "Curse";
  case 0x06:
    return "Foresight";
  case 0x07:
    return "Encore";
  case 0x08:
    return "FutureSight";
  case 0x09:
    return "Leech Seed";
  case 0x0A:
    return "Bind";
  case 0x0B:
    return "Wrap";
  case 0x0C:
    return "Fire Spin";
  case 0x0D:
    return "Clamp";
  case 0x0E:
    return "Whirlpool";
  // Silent variants
  case 0x0F:
    return "Disable|[silent]";
  case 0x10:
    return "confusion|[silent]";
  case 0x11:
    return "Mist|[silent]";
  case 0x12:
    return "move: Focus Energy|[silent]";
  case 0x13:
    return "Leech Seed|[silent]";
  case 0x14:
    return "Toxic counter|[silent]";
  case 0x15:
    return "Light Screen|[silent]";
  case 0x16:
    return "Reflect|[silent]";
  case 0x17:
    return "move: Bide|[silent]";
  case 0x18:
    return "Leech Seed|[from]"; // FIXME of
  default:
    throw std::runtime_error("Bad end reason: " + std::to_string(reason));
  }
}

template <View view = View::omniscient> struct Parser {
  const unsigned char *buf;
  size_t pos = 0;

  std::vector<std::string> log;
  std::optional<size_t> last_move_index;
  pkmn_gen1_battle battle;

  Parser(const uint8_t *b) : buf(b), battle{} {}

  uint8_t peek_u8() const { return buf[pos]; }

  uint8_t read_u8() { return buf[pos++]; }

  uint16_t read_u16() {
    uint16_t lo = buf[pos++];
    uint16_t hi = buf[pos++];
    return lo | (hi << 8);
  }

  void push(const std::string &s) { log.push_back(s); }

  void annotate_last_move(const std::string &suffix) {
    if (last_move_index) {
      log[*last_move_index] += suffix;
      last_move_index.reset();
    }
  }

  void possibly_hide_hp(const PokemonIdent identity, uint16_t &hp,
                        uint16_t &max_hp) const noexcept {
    if constexpr (view != View::omniscient) {
      if (identity.view() != view && max_hp) {
        hp = std::ceil(100.0 * hp / max_hp);
        max_hp = 100;
      }
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
      const auto opcode = static_cast<ArgType>(read_u8());
      switch (opcode) {
      case ArgType::null: {
        return;
      }
      case ArgType::lastmiss: {
        for (int i = log.size() - 1; i >= 0; --i) {
          auto &line = log[i];
          auto x = line.substr(0, 6);
          if (x == "|move|") {
            line += "|[miss]";
            break;
          }
        }
        break;
      }
      case ArgType::laststill: {
        for (int i = log.size() - 1; i >= 0; --i) {
          auto &line = log[i];
          auto x = line.substr(0, 6);
          if (x == "|move|") {
            line += "|[still]";
            break;
          }
        }
        break;
      }
      case ArgType::move: {
        auto source = read_u8();
        auto move = read_u8();
        auto target = read_u8();
        auto reason = read_u8();
        switch (reason) {
        case 0x00: {
          push("|move|" +
               ident_to_string(PKMN::view(battle), decode_ident(source)) + "|" +
               PKMN::move_string(move) + "|" +
               ident_to_string(PKMN::view(battle), decode_ident(target)));
          break;
        }
        case 0x01:
        case 0x02: {
          auto from = read_u8();
          push("|move|" +
               ident_to_string(PKMN::view(battle), decode_ident(source)) + "|" +
               PKMN::move_string(move) + "|" +
               ident_to_string(PKMN::view(battle), decode_ident(target)) +
               "|[from] " + PKMN::move_string(from));
          break;
        }
        default: {
          assert(false);
        }
        }
        break;
      }
      case ArgType::switch_: {
        auto identity = decode_ident(read_u8());
        auto species = read_u8();
        auto level = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        possibly_hide_hp(identity, hp, max_hp);
        std::string level_string =
            level == 100 ? "" : (std::to_string(level) + "|");
        push("|switch|" + ident_to_string(PKMN::view(battle), identity) + "|" +
             PKMN::species_string(species) + "|" + level_string +
             std::to_string(hp) + "/" + std::to_string(max_hp));
        break;
      }
      case ArgType::cant: {
        auto identity = decode_ident(read_u8());
        auto reason = read_u8();
        std::string prefix = "|cant|" +
                             ident_to_string(PKMN::view(battle), identity) +
                             "|" + cant_reason(reason);
        if (reason == 0x05) {
          auto move = read_u8();
          prefix += "|" + PKMN::move_string(move);
        }
        push(prefix);
        break;
      }
      case ArgType::faint: {
        auto identity = decode_ident(read_u8());
        push("|faint|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::turn: {
        auto turn = read_u16();
        push("|turn|" + std::to_string(turn));
        break;
      }
      case ArgType::win: {
        auto player = read_u8();
        push("|win|p" + std::to_string(player));
        break;
      }
      case ArgType::tie: {
        push("|tie|");
        break;
      }
      case ArgType::damage: {
        auto identity = decode_ident(read_u8());
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();
        possibly_hide_hp(identity, hp, max_hp);
        std::string prefix = "|-damage|" +
                             ident_to_string(PKMN::view(battle), identity) +
                             "|" + condition_string(hp, max_hp, status);
        if (reason != 0x00) {
          prefix += "|" + damage_reason(reason);
          if (reason == 0x05) {
            auto of = decode_ident(read_u8());
            prefix += "|" + ident_to_string(PKMN::view(battle), of);
          }
        }
        push(prefix);
        break;
      }
      case ArgType::heal: {
        auto identity = decode_ident(read_u8());
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();
        possibly_hide_hp(identity, hp, max_hp);
        std::string prefix = "|-heal|" +
                             ident_to_string(PKMN::view(battle), identity) +
                             "|" + condition_string(hp, max_hp, status);
        if (reason != 0x00) {
          prefix += heal_reason(reason);
          if (reason == 0x02) {
            auto of = decode_ident(read_u8());
            prefix += "|" + ident_to_string(PKMN::view(battle), of);
          }
        }
        push(prefix);
        break;
      }
      case ArgType::status: {
        auto identity = decode_ident(read_u8());
        auto status = read_u8();
        auto reason = read_u8();
        std::string prefix = "|-status|" +
                             ident_to_string(PKMN::view(battle), identity) +
                             "|" + status_string(status);
        switch (reason) {
        case 0x00: {
          push(prefix);
          break;
        }
        case 0x01: {
          push(prefix + "|[silent]");
          break;
        }
        case 0x02: {
          auto move = read_u8();
          push(prefix + "|[from]|" + PKMN::move_string(move));
          break;
        }
        default: {
          assert(false);
        }
        }
        break;
      }
      case ArgType::curestatus: {
        auto identity = decode_ident(read_u8());
        auto status = read_u8();
        auto reason = read_u8();
        push("|-curestatus|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::boost: {
        auto identity = decode_ident(read_u8());
        auto reason = read_u8();
        auto num = read_u8();
        int boost = static_cast<int>(num) - 6;
        if (boost > 0) {
          push("|-boost|" + ident_to_string(PKMN::view(battle), identity) +
               "|" + boost_reason(reason) + "|" + std::to_string(boost));
        } else if (boost < 0) {
          assert(reason != 0);
          push("|-unboost|" + ident_to_string(PKMN::view(battle), identity) +
               "|" + boost_reason(reason) + "|" + std::to_string(-boost));
        } else {
          assert(false);
        }
        break;
      }
      case ArgType::clearallboost: {
        push("|-clearallboost|");
        break;
      }
      case ArgType::fail: {
        auto identity = decode_ident(read_u8());
        auto reason = read_u8();
        push("|-fail|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::miss: {
        auto identity = decode_ident(read_u8());
        push("|-miss|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::hitcount: {
        auto identity = decode_ident(read_u8());
        auto num = read_u8();
        push("|-hitcount|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::prepare: {
        auto identity = decode_ident(read_u8());
        auto move = read_u8();
        push("|-prepare|" + ident_to_string(PKMN::view(battle), identity) +
             "|" + PKMN::move_string(move));
        break;
      }
      case ArgType::mustrecharge: {
        auto identity = decode_ident(read_u8());
        push("|-mustrecharge|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::activate: {
        auto identity = decode_ident(read_u8());
        auto reason = read_u8();
        push("|-activate|" + ident_to_string(PKMN::view(battle), identity) +
             "|" + activate_reason(reason));
        break;
      }
      case ArgType::fieldactivate: {
        push("|-fieldactivate|");
        break;
      }
      case ArgType::start: {
        auto identity = decode_ident(read_u8());
        auto reason = read_u8();
        std::string prefix = "|-start|" +
                             ident_to_string(PKMN::view(battle), identity) +
                             "|" + start_reason(reason);
        // weird cases
        switch (reason) {
        // 9, 10, 11 have trailing '|' pipe in the reason already
        case 0x09: {
          auto types = read_u8();
          auto of = decode_ident(read_u8());
          push(prefix + std::to_string(types) + "|" +
               ident_to_string(PKMN::view(battle), of));
          break;
        }
        case 0x0A:
        case 0x0B: {
          auto move = read_u8();
          push(prefix + PKMN::move_string(move));
          break;
        }
        default: {
          // normal cases
          if (reason < 0x09) {
            push(prefix);
          } else {
            assert(false);
          }
        }
        }
        break;
      }
      case ArgType::end: {
        auto identity = decode_ident(read_u8());
        auto reason = read_u8();
        push("|-end|" + ident_to_string(PKMN::view(battle), identity) + "|" +
             end_reason(reason));
        break;
      }
      case ArgType::ohko: {
        push("|-ohko|");
        break;
      }
      case ArgType::crit: {
        auto identity = decode_ident(read_u8());
        push("|-crit|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::supereffective: {
        auto identity = decode_ident(read_u8());
        push("|-supereffective|" +
             ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::resisted: {
        auto identity = decode_ident(read_u8());
        push("|-resisted|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::immune: {
        auto identity = decode_ident(read_u8());
        auto reason = read_u8(); // 0x00 none, 0x01 ohko
        push("|-immune|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::transform: {
        auto source = decode_ident(read_u8());
        auto target = decode_ident(read_u8());
        push("|-transform|" + ident_to_string(PKMN::view(battle), source) +
             ident_to_string(PKMN::view(battle), target));
        break;
      }
      case ArgType::drag: {
        auto identity = decode_ident(read_u8());
        auto species = read_u8();
        auto gender = read_u8();
        auto level = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        push("|drag|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::item: {
        auto target = read_u8();
        auto item = read_u8();
        auto source = read_u8();
        push("|-item|" + std::to_string(target));
        break;
      }
      case ArgType::enditem: {
        auto target = read_u8();
        auto item = read_u8();
        auto source = read_u8();
        push("|-enditem|" + std::to_string(target));
        break;
      }
      case ArgType::cureteam: {
        auto identity = decode_ident(read_u8());
        push("|-cureteam|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::sethp: {
        auto identity = decode_ident(read_u8());
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();
        push("|-sethp|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::setboost: {
        auto identity = decode_ident(read_u8());
        auto num = read_u8();
        push("|-setboost|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::copyboost: {
        auto source = read_u8();
        auto target = read_u8();
        push("|-copyboost|" + std::to_string(source));
        break;
      }
      case ArgType::sidestart: {
        auto player = read_u8();
        auto reason = read_u8();
        push("|-sidestart|" + std::to_string(player));
        break;
      }
      case ArgType::sideend: {
        auto player = read_u8();
        auto reason = read_u8();
        uint8_t of;
        if (reason == 0x03) {
          of = read_u8();
        }
        push("|-sideend|" + std::to_string(player));
        break;
      }
      case ArgType::singlemove: {
        auto identity = decode_ident(read_u8());
        auto move = read_u8();
        push("|-singlemove|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::singleturn: {
        auto identity = decode_ident(read_u8());
        auto move = read_u8();
        push("|-singleturn|" + ident_to_string(PKMN::view(battle), identity));
        break;
      }
      case ArgType::weather: {
        auto weather = read_u8();
        auto reason = read_u8();
        push("|-weather|" + std::to_string(weather));
        break;
      }
      default: {
        assert(false);
      }
      }
    }
  }
};

} // namespace PKMN::Log