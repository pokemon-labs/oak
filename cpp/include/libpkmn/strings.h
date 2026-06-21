#pragma once

#include <pkmn.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <libpkmn/data.h>
#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/layout.h>

namespace PKMN {

using PKMN::Data::Move;
using PKMN::Data::Species;
using PKMN::Data::Status;

// TODO binding, force, recharge etc
inline std::string side_choice_string(const PKMN::Side &side,
                                      pkmn_choice choice) {
  const auto choice_type = choice & 3;
  const auto choice_data = choice >> 2;
  switch (choice_type) {
  case 0: {
    return "Pass";
  }
  case 1: {
    if (choice_data == 0) {
      return "None";
    }
    if (side.active.volatiles.rage()) {
      assert(choice_data == 1);
      return "Rage";
    }
    using namespace PKMN::Layout::Offsets;
    return move_string(side.active.moves[choice_data - 1].id);
  }
  case 2: {
    return species_string(side.get(choice_data).species);
  }
  default: {
    assert(false);
    return "";
  }
  }
}

inline bool match(const auto &A, const auto &B) {
  return std::equal(
      A.begin(), A.begin() + std::min(A.size(), B.size()), B.begin(), B.end(),
      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

inline std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

inline auto find_unique(const auto &container, const auto &value) {
  const auto matches = [&value](const auto &x) { return match(x, value); };
  auto it = std::find_if(container.begin(), container.end(), matches);
  if (it != container.end()) {
    if (auto other = std::find_if(it + 1, container.end(), matches);
        other != container.end()) {
      if (it->size() == value.size()) {
        return it;
      } else if (other->size() == value.size()) {
        return other;
      } else {
        return container.end(); // return end if not unique
      }
    }
  }
  return it;
}

inline int unique_index(const auto &container, const auto &value) {
  const auto it = find_unique(container, value);
  if (it == container.end()) {
    return -1;
  }
  return std::distance(container.begin(), it);
}

inline std::string status_string(const auto status) {
  const auto byte = static_cast<uint8_t>(status);
  if (byte == 0) {
    return "";
  }
  if (byte & 7) {
    if (byte & 128) {
      return "RST";
    } else {
      return "SLP";
    }
  }
  switch (byte) {
  case 0b00001000:
    return "PSN";
  case 0b00010000:
    return "BRN";
  case 0b00100000:
    return "FRZ";
  case 0b01000000:
    return "PAR";
  case 0b10001000:
    return "TOX";
  default:
    assert(false);
    return "";
  };
}

inline std::string pokemon_to_string(const uint8_t *const data) {
  std::stringstream sstream{};
  sstream << species_string(data[21]);
  if (data[23] != 100) {
    sstream << " (lvl " << (int)data[23] << ")";
  }
  sstream << status_string(data[20]);
  for (int m = 0; m < 4; ++m) {
    if (data[2 * m + 10] != 0) {
      sstream << move_string(data[2 * m + 10]) << ": " << (int)data[2 * m + 11]
              << " ";
    }
  }
  return sstream.str();
}

inline std::string volatiles_to_string(const PKMN::Volatiles &vol) {
  std::stringstream ss{};
  if (vol.bide())
    ss << "(bide)";
  if (vol.thrashing())
    ss << "(thrashing)";
  if (vol.multi_hit())
    ss << "(multi-hit)";
  if (vol.flinch())
    ss << "(flinch)";
  if (vol.charging())
    ss << "(charging)";
  if (vol.binding())
    ss << "(binding)";
  if (vol.invulnerable())
    ss << "(invulnerable)";
  if (vol.confusion())
    ss << "(confused)";
  if (vol.mist())
    ss << "(mist)";
  if (vol.focus_energy())
    ss << "(focus-energy)";
  if (vol.substitute())
    ss << "(substitute)";
  if (vol.recharging())
    ss << "(recharging)";
  if (vol.rage())
    ss << "(rage)";
  if (vol.leech_seed())
    ss << "(leech-seed)";
  if (vol.toxic())
    ss << "(toxic)";
  if (vol.light_screen())
    ss << "(light-screen)";
  if (vol.reflect())
    ss << "(reflect)";
  if (vol.transform())
    ss << "(transform)";
  if (vol.confusion_left())
    ss << "(confusion_left: " << (int)vol.confusion_left() << ")";
  if (vol.attacks())
    ss << "(attacks: " << (int)vol.attacks() << ")";
  if (vol.state())
    ss << "(state: " << (int)vol.state() << ")";
  if (vol.substitute_hp())
    ss << "(sub_hp: " << (int)vol.substitute_hp() << ")";
  if (vol.transform_species()) {
    const auto player_index = (vol.transform_species() & 0b1000) >> 3;
    const auto slot = vol.transform_species() & 0b111;
    ss << "(transform: p" << player_index << "s" << slot << ")";
  }
  if (vol.disable_left())
    ss << "(disable_left: " << (int)vol.disable_left() << ")";
  if (vol.disable_move())
    ss << "(disable_move: " << (int)vol.disable_move() << ")";
  if (vol.toxic_counter())
    ss << "(toxic_counter: " << (int)vol.toxic_counter() << ")";
  return ss.str();
}

inline std::string
battle_data_to_string(const pkmn_gen1_battle &battle,
                      const pkmn_gen1_chance_durations &durations) {
  std::stringstream ss{};
  const auto &b = PKMN::view(battle);
  for (auto s = 0; s < 2; ++s) {
    const auto &side = b.sides[s];
    const auto &active = side.active;
    const auto &duration = PKMN::view(durations).get(s);
    const auto &vol = side.active.volatiles;

    for (auto i = 0; i < 6; ++i) {
      const auto id = side.order[i];
      if (id == 0) {
        continue;
      }

      const auto &pokemon = side.pokemon[id - 1];
      if (i == 0) {

        // boosts
        bool equal = true;
        if (active.stats.atk != pokemon.stats.atk) {
          ss << "(atk " << pokemon.stats.atk << ">>" << active.stats.atk
             << ") ";
          equal = false;
        }
        if (active.stats.def != pokemon.stats.def) {
          ss << "(def " << pokemon.stats.def << ">>" << active.stats.def
             << ") ";
          equal = false;
        }
        if (active.stats.spe != pokemon.stats.spe) {
          ss << "(spe " << pokemon.stats.spe << ">>" << active.stats.spe
             << ") ";
          equal = false;
        }
        if (active.stats.spc != pokemon.stats.spc) {
          ss << "(spc " << pokemon.stats.spc << ">>" << active.stats.spc
             << ") ";
          equal = false;
        }
        const auto &boosts = active.boosts;
        if (boosts.atk() != 0) {
          ss << "[atk " << (int)boosts.atk() << "] ";
          equal = false;
        }
        if (boosts.def() != 0) {
          ss << "[def " << (int)boosts.def() << "] ";
          equal = false;
        }
        if (boosts.spe() != 0) {
          ss << "[spe " << (int)boosts.spe() << "] ";
          equal = false;
        }
        if (boosts.spc() != 0) {
          ss << "[spc " << (int)boosts.spc() << "] ";
          equal = false;
        }
        if (boosts.acc() != 0) {
          ss << "[acc " << (int)boosts.acc() << "] ";
          equal = false;
        }
        if (boosts.eva() != 0) {
          ss << "[eva " << (int)boosts.eva() << "] ";
          equal = false;
        }
        if (!equal) {
          ss << "\n";
        }

        // durations
        bool no_durations = true;
        if (duration.confusion()) {
          ss << "conf: " << static_cast<int>(duration.confusion());
          no_durations = false;
        }
        if (duration.disable()) {
          ss << " disable: " << static_cast<int>(duration.disable());
          no_durations = false;
        }
        if (duration.attacking()) {
          ss << " attacking: " << static_cast<int>(duration.attacking());
          no_durations = false;
        }
        if (duration.binding()) {
          ss << " binding: " << static_cast<int>(duration.binding());
          no_durations = false;
        }
        if (!no_durations) {
          ss << '\n';
        }

        // vol
        const auto vol_string = volatiles_to_string(vol);
        if (!vol_string.empty()) {
          ss << vol_string << '\n';
        }

      } else {
        ss << "  ";
      }

      // stored print
      ss << species_char_array(pokemon.species);
      if (pokemon.level != 100) {
        ss << " L" << (int)pokemon.level;
      }
      ss << ": ";
      const auto hp = pokemon.hp;
      if (hp != 0) {
        ss << pokemon.percent() << "% (" << pokemon.hp << '/'
           << pokemon.stats.hp << ") ";
      } else {
        ss << "KO " << '\n';
        continue;
      }
      const auto st = pokemon.status;
      if (st != Status::None) {
        ss << status_string(st);
        if (is_sleep(st)) {
          ss << ':';
          if (self(st)) {
            ss << (static_cast<uint32_t>(st) & 7);
          } else {
            ss << (int)duration.sleep(i);
          }
        }
        ss << ' ';
      }
      for (auto m = 0; m < 4; ++m) {
        const bool matching = (pokemon.moves[m] == side.active.moves[m]);
        if ((i > 0) || matching) {
          const auto &moveslot = pokemon.moves[m];
          ss << move_char_array(moveslot.id) << ":" << (int)moveslot.pp << ' ';
        } else {
          const auto &a = active.moves[m];
          const auto &p = pokemon.moves[m];
          ss << move_char_array(a.id) << ":" << (int)a.pp << '/';
          ss << move_char_array(p.id) << ":" << (int)p.pp << ' ';
        }
      }
      ss << '\n';
    }
    ss << "|| last_used: " << PKMN::move_string(side.last_used_move) << '/'
       << (int)side.last_used_move << ' ';
    ss << "last_selected: " << PKMN::move_string(side.last_selected_move) << '/'
       << (int)side.last_selected_move << ' ';
    ss << '(';
    for (auto o = 0; o < 6; ++o) {
      ss << (int)side.order[o] << ' ';
    }
    ss << ")\n";
    if (s == 0) {
      ss << "--- --- --- " << b.turn << " --- --- ---" << "\n";
    }
  }
  ss << "---\n";
  ss << "last_damage: " << b.last_damage << '\n';
  return ss.str();
}

inline std::string to_string(const pkmn_gen1_battle &battle) {
  return battle_data_to_string(battle, {});
}

inline std::string to_string(const auto &battle_data) {
  return battle_data_to_string(battle_data.battle, battle_data.durations);
}

inline Species string_to_species(const std::string &str) {
  const int index = unique_index(PKMN::Data::SPECIES_CHAR_ARRAY, str);
  if (index < 0) {
    const auto lower = to_lower(str);
    if (lower == "pidgeot") {
      return Species::Pidgeot;
    }
    if (lower == "paras") {
      return Species::Paras;
    }
    if (lower == "kabuto") {
      return Species::Kabuto;
    }
    if (lower == "mew") {
      return Species::Mew;
    }
    throw std::runtime_error{"Could not match string to Species"};
    return Species::None;
  } else {
    return static_cast<Species>(index);
  }
}

inline Move string_to_move(const std::string &str) {
  const int index = unique_index(PKMN::Data::MOVE_CHAR_ARRAY, str);
  if (index < 0) {
    const auto lower = to_lower(str);
    if (lower == "acid") {
      return Move::Acid;
    }
    if (lower == "bubble") {
      return Move::Bubble;
    }
    if (lower == "thunder") {
      return Move::Thunder;
    }
    throw std::runtime_error{"Could not match string to Move"};
    return Move::None;
  } else {
    return static_cast<Move>(index);
  }
}

} // namespace PKMN