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

/*
FIXME leechseed From Of
Reason
Raw	Description
0x00	Disable*
0x01	confusion
0x02	Bide*
0x03	Substitute
0x04	Disable|[silent]
0x05	confusion|[silent]
0x06	mist|[silent]
0x07	focusenergy|[silent]
0x08	leechseed|[silent]
0x09	Toxic counter|[silent]
0x0A	lightscreen|[silent]
0x0B	reflect|[silent]
0x0C	move: Bide|[silent]
*0x00 corresponds to move: Disable and 0x02 to move: Bide in Generation II.
*/
std::string end_reason(uint8_t reason) {
  switch (reason) {
  case 0x00:
    return "Disable";
  case 0x01:
    return "confusion";
  case 0x02:
    return "Bide";
  case 0x03:
    return "Substitute";
  case 0x04:
    return "Disable|[silent]";
  case 0x05:
    return "confusion|[silent]";
  case 0x06:
    return "mist|[silent]";
  case 0x07:
    return "focusenergy|[silent]";
  case 0x08:
    return "leechseed|[silent]";
  case 0x09:
    return "Toxic counter|[silent]";
  case 0x0A:
    return "lightscreen|[silent]";
  case 0x0B:
    return "reflect|[silent]";
  case 0x0C:
    return "move: Bide|[silent]";
  default:
    std::cout << (int)reason << std::endl;
    assert(false);
  }
  return "";
}

template <View view = View::omniscient> struct Parser {
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
    std::cout << s << std::endl;
    log.push_back(s);
  }

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
        push("|laststill");
        break;
      }
      case ArgType::move: {
        // |move|p1a: Jynx|Blizzard|p2a: Chansey|[miss]
        // |move|p1a: Cloyster|Clamp|p2a: Alakazam|[from] Clamp
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
          // |move|p1a: Fearow|Psychic|p2a: Alakazam|[from] Mirror Move
          break;
        }
        default: {
          assert(false);
        }
        }
        break;
      }
      case ArgType::switch_: {
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
        auto ident = read_u8();
        push("|faint|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
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
        // |-damage|p2a: Exeggutor|0 fnt
        // |-damage|p1a: Tauros|91/353
        // |-damage|p2a: Starmie|18/100 slp
        // |-damage|p1a: Jolteon|157/333|[from] Recoil|[of] p2a: Snorlax
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
            prefix += "|" + ident_to_string(PKMN::view(battle), identity);
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
            auto of = read_u8();
            prefix += "|" + PKMN::move_string(of);
          }
        }
        push(prefix);
        break;
      }
      case ArgType::status: {
        auto ident = read_u8();
        auto status = read_u8();
        auto reason = read_u8();
        auto identity = decode_ident(ident);
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
        auto ident = read_u8();
        auto status = read_u8();
        auto reason = read_u8();
        push("|-curestatus|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::boost: {
        // The Pokémon identified by Ident has been (un)boosted by Num - 6 in a
        // stat indicated by the Reason.
        auto ident = read_u8();
        auto reason = read_u8();
        auto num = read_u8();
        auto identity = decode_ident(ident);
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
        auto ident = read_u8();
        auto reason = read_u8();
        push("|-fail|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::miss: {
        auto ident = read_u8();
        push("|-miss|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::hitcount: {
        auto ident = read_u8();
        auto num = read_u8();
        push("|-hitcount|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::prepare: {
        auto ident = read_u8();
        auto move = read_u8();
        auto identity = decode_ident(ident);
        push("|-prepare|" + ident_to_string(PKMN::view(battle), identity) +
             "|" + PKMN::move_string(move));
        break;
      }
      case ArgType::mustrecharge: {
        auto ident = read_u8();
        push("|-mustrecharge|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::activate: {
        auto ident = read_u8();
        auto reason = read_u8();
        auto identity = decode_ident(ident);
        push("|-activate|" + ident_to_string(PKMN::view(battle), identity) +
             "|" + activate_reason(reason));
        break;
      }
      case ArgType::fieldactivate: {
        push("|-fieldactivate|");
        break;
      }
      case ArgType::start: {
        auto ident = read_u8();
        auto reason = read_u8();
        auto identity = decode_ident(ident);
        std::string prefix = "|-start|" +
                             ident_to_string(PKMN::view(battle), identity) +
                             "|" + start_reason(reason);
        // weird cases
        switch (reason) {
        case 0x09: {
          auto types = read_u8();
          auto of = read_u8();
          auto of_ident = decode_ident(of);
          push(prefix + std::to_string(types) + "|" +
               ident_to_string(PKMN::view(battle), of_ident));
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
        std::cout << ident_to_string(PKMN::view(battle), identity) << std::endl;
        push("|-end|" + ident_to_string(PKMN::view(battle), identity) + "|" +
             end_reason(reason));
        break;
      }
      case ArgType::ohko: {
        push("|-ohko|");
        break;
      }
      case ArgType::crit: {
        auto ident = read_u8();
        push("|-crit|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::supereffective: {
        auto ident = read_u8();
        push("|-supereffective|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::resisted: {
        auto ident = read_u8();
        push("|-resisted|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::immune: {
        auto ident = read_u8();
        auto reason = read_u8(); // 0x00 none, 0x01 ohko
        push("|-immune|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::transform: {
        auto source = read_u8();
        auto target = read_u8();
        push("|-transform|" + std::to_string(source));
        break;
      }
      case ArgType::drag: {
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
        auto ident = read_u8();
        push("|-cureteam|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::sethp: {
        auto ident = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();
        push("|-sethp|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::setboost: {
        auto ident = read_u8();
        auto num = read_u8();
        push("|-setboost|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
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
        auto ident = read_u8();
        auto move = read_u8();
        push("|-singlemove|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::singleturn: {
        auto ident = read_u8();
        auto move = read_u8();
        push("|-singleturn|" +
             ident_to_string(PKMN::view(battle), decode_ident(ident)));
        break;
      }
      case ArgType::weather: {
        auto weather = read_u8();
        auto reason = read_u8();
        push("|-weather|" + std::to_string(weather));
        break;
      }
      default: {
        std::cout << "ERROR: " << static_cast<int>(opcode) << std::endl;
        assert(false);
      }
      }
    }
  }
};

} // namespace PKMN::Log