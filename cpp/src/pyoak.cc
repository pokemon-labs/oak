#include <format/ou/data.h>
#include <libpkmn/data.h>
#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/init.h>
#include <libpkmn/layout.h>
#include <libpkmn/strings.h>
#include <util/load-teams.h>
#include <util/parse.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <py/libpkmn/data.h>

namespace Py::PKMN {

namespace py = pybind11;
using namespace PKMN;
using namespace PKMN::Data;

const auto solve_matrix(py::array_t<float> p1_payoffs,
                        const int discretize_factor = 256) {
  if (p1_payoffs.ndim() != 2) {
    throw std::runtime_error{"Expecting 2d array"};
  }
  const auto m = p1_payoffs.shape(0);
  const auto n = p1_payoffs.shape(1);
  auto r = p1_payoffs.unchecked<2>();
  std::vector<int> disc;
  disc.resize(m * n);
  for (auto i = 0; i < m; ++i) {
    for (auto j = 0; j < n; ++j) {
      disc[i * n + j] = static_cast<int>(r(i, j) * discretize_factor);
    }
  }
  LRSNash::FastInput solve_input{static_cast<int>(m), static_cast<int>(n),
                                 disc.data(), discretize_factor};
  std::vector<float> nash1;
  std::vector<float> nash2;
  nash1.resize(m + 2);
  nash2.resize(n + 2);
  LRSNash::FloatOneSumOutput solve_output{nash1.data(), nash2.data(), 0};
  LRSNash::solve_fast(&solve_input, &solve_output);
  auto p1 = py::array_t<float>(std::vector<ssize_t>{m});
  auto p2 = py::array_t<float>(std::vector<ssize_t>{n});
  for (int i = 0; i < m; ++i) {
    p1.mutable_unchecked<1>()(i) = nash1[i];
  }
  for (int j = 0; j < n; ++j) {
    p2.mutable_unchecked<1>()(j) = nash2[j];
  }
  return py::make_tuple(p1, p2, solve_output.value);
}

struct PokemonSet {
  uint8_t species;
  uint8_t level;
  std::array<uint8_t, 4> moves;
  PokemonSet() = default;
  PokemonSet(const PKMN::Set &s) : moves{} {
    species = static_cast<uint8_t>(s.species);
    level = static_cast<uint8_t>(s.level);
    std::transform(s.moves.begin(), s.moves.end(), moves.begin(),
                   [](const auto move) { return static_cast<uint8_t>(move); });
    // std::sort(moves.begin(), moves.end());
    std::sort(moves.begin(), moves.end(),
              [](uint8_t a, uint8_t b) { return a < b; });
  }
  bool operator==(const PokemonSet &) const = default;
  bool operator<(const PokemonSet &other) const {
    return (species == other.species) &&
           std::all_of(moves.begin(), moves.end(), [&other](const auto m) {
             return !m || (std::find(other.moves.begin(), other.moves.end(),
                                     m) != other.moves.end());
           });
  }
  bool operator<=(const PokemonSet &other) const {
    return (*this == other) || (*this < other);
  }
  static inline void hash_combine(size_t &seed, size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  static size_t hash_set(const PokemonSet &s) {
    size_t h = 0;
    hash_combine(h, s.species);
    hash_combine(h, s.level);
    for (auto m : s.moves)
      hash_combine(h, m);
    return h;
  }
};

void complete_move_set(std::array<PKMN::MoveSlot, 4> &move_slots,
                       const std::array<uint8_t, 4> &moves) {
  for (auto move_id : moves) {
    auto incoming_move = static_cast<Move>(move_id);
    if (incoming_move == Move::None) {
      continue;
    }
    auto matching_slot = std::find_if(
        move_slots.begin(), move_slots.end(),
        [incoming_move](const auto &ms) { return ms.id == incoming_move; });
    if (matching_slot == move_slots.end()) {
      auto empty =
          std::find_if(move_slots.begin(), move_slots.end(),
                       [](const auto &ms) { return ms.id == Move::None; });
      assert(empty != move_slots.end());
      empty->id = incoming_move;
      empty->pp = max_pp(incoming_move);
    }
  }
}

PYBIND11_MODULE(pyoak, m) {
  m.doc() = "oak — low-level structured view of pkmn_gen1_battle bytes.\n\n"
            "  b    = oak.Battle(raw_bytes)   # 384 bytes\n"
            "  d    = oak.Durations(dur_bytes) # 8 bytes (or Durations() for "
            "zeroed)\n"
            "  side = b.side(0)                   # P1\n"
            "  pkmn = side.pokemon(0)             # storage index 0\n"
            "  lead = side.slot(1)                # battle slot 1 (respects "
            "order[])\n"
            "  act  = side.active\n"
            "  new_bytes = b.bytes()";

  // Module-level constants
  m.attr("BATTLE_SIZE") = static_cast<int>(PKMN_GEN1_BATTLE_SIZE);
  m.attr("DURATIONS_SIZE") = static_cast<int>(PKMN_GEN1_CHANCE_DURATIONS_SIZE);

  py::class_<StatsProxy>(
      m, "Stats", "Base/active stats: hp, atk, def_, spe, spc (all uint16).")
      .def_property("hp", &StatsProxy::get_hp, &StatsProxy::set_hp)
      .def_property("atk", &StatsProxy::get_atk, &StatsProxy::set_atk)
      .def_property("def_", &StatsProxy::get_def, &StatsProxy::set_def)
      .def_property("spe", &StatsProxy::get_spe, &StatsProxy::set_spe)
      .def_property("spc", &StatsProxy::get_spc, &StatsProxy::set_spc)
      .def("__repr__", [](const StatsProxy &s) {
        return "<Stats hp=" + std::to_string(s.get_hp()) +
               " atk=" + std::to_string(s.get_atk()) +
               " def=" + std::to_string(s.get_def()) +
               " spe=" + std::to_string(s.get_spe()) +
               " spc=" + std::to_string(s.get_spc()) + ">";
      });

  py::class_<MoveSlotProxy>(m, "MoveSlot", "id (uint8) + pp (uint8).")
      .def_property("id", &MoveSlotProxy::get_id, &MoveSlotProxy::set_id,
                    "Move enum value as uint8.")
      .def_property("pp", &MoveSlotProxy::get_pp, &MoveSlotProxy::set_pp)
      .def("name", &MoveSlotProxy::name, "Move name string.")
      .def("__repr__", [](const MoveSlotProxy &ms) {
        return "<MoveSlot " + ms.name() + " pp=" + std::to_string(ms.get_pp()) +
               ">";
      });

  py::class_<BoostsProxy>(
      m, "Boosts",
      "Stat boosts (int8, range -6..+6). raw exposes the underlying uint32.")
      .def_property("atk", &BoostsProxy::get_atk, &BoostsProxy::set_atk)
      .def_property("def_", &BoostsProxy::get_def, &BoostsProxy::set_def)
      .def_property("spe", &BoostsProxy::get_spe, &BoostsProxy::set_spe)
      .def_property("spc", &BoostsProxy::get_spc, &BoostsProxy::set_spc)
      .def_property("acc", &BoostsProxy::get_acc, &BoostsProxy::set_acc)
      .def_property("eva", &BoostsProxy::get_eva, &BoostsProxy::set_eva)
      .def_property("raw", &BoostsProxy::get_raw, &BoostsProxy::set_raw,
                    "Raw uint32 of the 4 packed boost bytes.")
      .def("__repr__", [](const BoostsProxy &b) {
        return "<Boosts atk=" + std::to_string(b.get_atk()) +
               " def=" + std::to_string(b.get_def()) +
               " spe=" + std::to_string(b.get_spe()) +
               " spc=" + std::to_string(b.get_spc()) +
               " acc=" + std::to_string(b.get_acc()) +
               " eva=" + std::to_string(b.get_eva()) + ">";
      });

  py::class_<VolatilesProxy>(
      m, "Volatiles",
      "Active-pokemon volatile flags and counters backed by a single uint64.\n"
      "All 18 boolean flags, counters with C++ setters, and the remaining\n"
      "bit-packed fields (state, substitute_hp, transform_species,\n"
      "disable_move, toxic_counter) are individually writable.\n"
      "`bits` exposes the raw uint64 directly.")
      // boolean flags
      .def_property("bide", &VolatilesProxy::get_bide,
                    &VolatilesProxy::set_bide)
      .def_property("thrashing", &VolatilesProxy::get_thrashing,
                    &VolatilesProxy::set_thrashing)
      .def_property("multi_hit", &VolatilesProxy::get_multi_hit,
                    &VolatilesProxy::set_multi_hit)
      .def_property("flinch", &VolatilesProxy::get_flinch,
                    &VolatilesProxy::set_flinch)
      .def_property("charging", &VolatilesProxy::get_charging,
                    &VolatilesProxy::set_charging)
      .def_property("binding", &VolatilesProxy::get_binding,
                    &VolatilesProxy::set_binding)
      .def_property("invulnerable", &VolatilesProxy::get_invulnerable,
                    &VolatilesProxy::set_invulnerable)
      .def_property("confusion", &VolatilesProxy::get_confusion,
                    &VolatilesProxy::set_confusion)
      .def_property("mist", &VolatilesProxy::get_mist,
                    &VolatilesProxy::set_mist)
      .def_property("focus_energy", &VolatilesProxy::get_focus_energy,
                    &VolatilesProxy::set_focus_energy)
      .def_property("substitute", &VolatilesProxy::get_substitute,
                    &VolatilesProxy::set_substitute)
      .def_property("recharging", &VolatilesProxy::get_recharging,
                    &VolatilesProxy::set_recharging)
      .def_property("rage", &VolatilesProxy::get_rage,
                    &VolatilesProxy::set_rage)
      .def_property("leech_seed", &VolatilesProxy::get_leech_seed,
                    &VolatilesProxy::set_leech_seed)
      .def_property("toxic", &VolatilesProxy::get_toxic,
                    &VolatilesProxy::set_toxic)
      .def_property("light_screen", &VolatilesProxy::get_light_screen,
                    &VolatilesProxy::set_light_screen)
      .def_property("reflect", &VolatilesProxy::get_reflect,
                    &VolatilesProxy::set_reflect)
      .def_property("transform", &VolatilesProxy::get_transform,
                    &VolatilesProxy::set_transform)
      // counters (C++ setters exist)
      .def_property("confusion_left", &VolatilesProxy::get_confusion_left,
                    &VolatilesProxy::set_confusion_left)
      .def_property("attacks", &VolatilesProxy::get_attacks,
                    &VolatilesProxy::set_attacks)
      .def_property("disable_left", &VolatilesProxy::get_disable_left,
                    &VolatilesProxy::set_disable_left)
      // bit-packed fields (manual setters via bf_set)
      .def_property("state", &VolatilesProxy::get_state,
                    &VolatilesProxy::set_state,
                    "16-bit Bide/Rage/binding damage accumulator (bits 39:24).")
      .def_property("substitute_hp", &VolatilesProxy::get_substitute_hp,
                    &VolatilesProxy::set_substitute_hp,
                    "Substitute HP (bits 47:40).")
      .def_property("transform_species", &VolatilesProxy::get_transform_species,
                    &VolatilesProxy::set_transform_species,
                    "Transformed-into species index (bits 51:48).")
      .def_property("disable_move", &VolatilesProxy::get_disable_move,
                    &VolatilesProxy::set_disable_move,
                    "Disabled move slot index 0-3 (bits 58:56).")
      .def_property("toxic_counter", &VolatilesProxy::get_toxic_counter,
                    &VolatilesProxy::set_toxic_counter,
                    "Toxic damage counter N (bits 63:59).")
      // raw
      .def_property("bits", &VolatilesProxy::get_bits,
                    &VolatilesProxy::set_bits,
                    "Raw uint64 — all 64 bits at once.")
      .def("__str__", &VolatilesProxy::to_string)
      .def("__repr__", [](const VolatilesProxy &v) {
        return "<Volatiles 0x" + [&] {
          char buf[17];
          std::snprintf(buf, sizeof(buf), "%016llx",
                        (unsigned long long)v.get_bits());
          return std::string(buf);
        }() + ">";
      });

  py::class_<MoveDetailsProxy>(m, "MoveDetails")
      .def_property("index", &MoveDetailsProxy::get_index,
                    &MoveDetailsProxy::set_index,
                    py::return_value_policy::reference_internal)
      .def_property("counterable", &MoveDetailsProxy::get_counterable,
                    &MoveDetailsProxy::set_counterable,
                    py::return_value_policy::reference_internal);

  py::class_<PokemonProxy>(m, "Pokemon")
      .def("stats", &PokemonProxy::stats,
           py::return_value_policy::reference_internal)
      .def("move", &PokemonProxy::move, py::arg("index"), "Move slot 0-3.",
           py::return_value_policy::reference_internal)
      .def_property("hp", &PokemonProxy::get_hp, &PokemonProxy::set_hp)
      .def_property("status", &PokemonProxy::get_status,
                    &PokemonProxy::set_status,
                    "Status byte (see Data::Status enum values).")
      .def_property("species", &PokemonProxy::get_species,
                    &PokemonProxy::set_species, "Species enum value as uint8.")
      .def_property("types", &PokemonProxy::get_types, &PokemonProxy::set_types,
                    "Packed type byte.")
      .def_property("level", &PokemonProxy::get_level, &PokemonProxy::set_level)
      .def("percent", &PokemonProxy::percent, "Current HP as a percentage.")
      .def("species_name", &PokemonProxy::species_name, "Species name string.")
      .def("status_name", &PokemonProxy::status_name,
           "Status abbreviation string.")
      .def("__str__", &PokemonProxy::to_string)
      .def("__repr__", [](const PokemonProxy &p) {
        return "<Pokemon " + p.species_name() +
               " hp=" + std::to_string(p.get_hp()) + ">";
      });

  // ---- ActivePokemon -------------------------------------------------------
  py::class_<ActivePokemonProxy>(
      m, "ActivePokemon",
      "The in-battle Pokemon for a side: has stats, boosts, volatiles, moves.")
      .def("stats", &ActivePokemonProxy::stats,
           py::return_value_policy::reference_internal)
      .def("boosts", &ActivePokemonProxy::boosts,
           py::return_value_policy::reference_internal)
      .def("volatiles", &ActivePokemonProxy::volatiles,
           py::return_value_policy::reference_internal)
      .def("move", &ActivePokemonProxy::move, py::arg("index"),
           "Move slot 0-3.", py::return_value_policy::reference_internal)
      .def_property("species", &ActivePokemonProxy::get_species,
                    &ActivePokemonProxy::set_species)
      .def_property("types", &ActivePokemonProxy::get_types,
                    &ActivePokemonProxy::set_types)
      .def("species_name", &ActivePokemonProxy::species_name)
      .def("__repr__", [](const ActivePokemonProxy &a) {
        return "<ActivePokemon " + a.species_name() + ">";
      });

  py::class_<SideProxy>(m, "Side")
      .def("pokemon", &SideProxy::pokemon, py::arg("index"),
           "Pokemon at storage index 0-5 (independent of battle order).",
           py::return_value_policy::reference_internal)
      .def("slot", &SideProxy::slot, py::arg("slot"),
           "Pokemon in battle slot 1-6, respecting order[].",
           py::return_value_policy::reference_internal)
      .def_property_readonly("active", &SideProxy::active,
                             py::return_value_policy::reference_internal)
      .def("stored", &SideProxy::stored,
           "Shorthand for slot(1) — the currently active (stored) Pokemon.",
           py::return_value_policy::reference_internal)
      .def_property(
          "order", &SideProxy::get_order, &SideProxy::set_order,
          "Six-element array mapping slot (1-indexed) to storage index.")
      .def_property("last_selected_move", &SideProxy::get_last_selected_move,
                    &SideProxy::set_last_selected_move)
      .def_property("last_used_move", &SideProxy::get_last_used_move,
                    &SideProxy::set_last_used_move)
      .def("__repr__", [](const SideProxy &) { return "<Side>"; });

  py::class_<DurationProxy>(m, "Duration",
                            "Per-side duration counters for one player.")
      .def("sleep", &DurationProxy::sleep, py::arg("slot"),
           "Sleep counter for Pokemon at order slot 0-5.")
      .def("set_sleep", &DurationProxy::set_sleep, py::arg("slot"),
           py::arg("value"))
      .def_property("confusion", &DurationProxy::get_confusion,
                    &DurationProxy::set_confusion)
      .def_property("disable", &DurationProxy::get_disable,
                    &DurationProxy::set_disable)
      .def_property("attacking", &DurationProxy::get_attacking,
                    &DurationProxy::set_attacking)
      .def_property("binding", &DurationProxy::get_binding,
                    &DurationProxy::set_binding)
      .def_property("raw", &DurationProxy::get_raw, &DurationProxy::set_raw,
                    "Raw uint32 of all packed duration bits.")
      .def("__repr__", [](const DurationProxy &) { return "<Duration>"; });

  py::class_<DurationsView>(m, "Durations",
                            "8-byte chance durations for both sides.\n\n"
                            "  d = oak.Durations(dur_bytes)  # from bytes\n"
                            "  d = oak.Durations()           # zeroed\n"
                            "  d.get(0).confusion = 3\n"
                            "  dur_bytes = d.bytes()")
      .def(py::init<>(), "Construct zeroed Durations.")
      .def(py::init<py::bytes>(), py::arg("data"),
           "Construct from 8 raw bytes.")
      .def("get", &DurationsView::get, py::arg("side"),
           "Duration for side 0 or 1.",
           py::return_value_policy::reference_internal)
      .def("bytes", &DurationsView::bytes,
           "Return current (possibly mutated) 8-byte representation.")
      .def("__repr__", [](const DurationsView &) { return "<Durations>"; })
      .def(py::pickle([](const DurationsView &d) { return d.bytes(); },
                      [](py::bytes b) { return DurationsView(b); }));

  py::class_<BattleView>(m, "Battle",
                         "Structured view of a 384-byte pkmn_gen1_battle.\n\n"
                         "  b = oak.Battle(raw_bytes)\n"
                         "  b.side(0).slot(1).hp = 200\n"
                         "  new_bytes = b.bytes()")
      .def(py::init<>(), "Construct zeroed Battle.")
      .def(py::init<py::bytes>(), py::arg("data"),
           "Construct from 384 raw bytes.")
      .def("side", &BattleView::side, py::arg("index"),
           "Side 0 (P1) or Side 1 (P2).",
           py::return_value_policy::reference_internal)
      .def("last_move", &BattleView::last_move, py::arg("index"),
           "last_moves[_], 0 or 1", py::return_value_policy::reference_internal)
      .def_property("turn", &BattleView::get_turn, &BattleView::set_turn)
      .def_property("last_damage", &BattleView::get_last_damage,
                    &BattleView::set_last_damage)
      .def_property("rng", &BattleView::get_rng, &BattleView::set_rng,
                    "RNG seed as uint64.")
      .def("bytes", &BattleView::bytes,
           "Return the current (possibly mutated) 384-byte battle "
           "representation.")
      .def("__str__", &BattleView::to_string)
      .def("__repr__",
           [](const BattleView &b) {
             return "<Battle turn=" + std::to_string(b.get_turn()) + ">";
           })
      .def(py::pickle([](const BattleView &b) { return b.bytes(); },
                      [](py::bytes b) { return BattleView(b); }));

  py::class_<PokemonSet>(m, "Set")
      .def(py::init<>(), "Construct zeroed Set.")
      .def_readwrite("species", &PokemonSet::species)
      .def_readwrite("level", &PokemonSet::level)
      .def_readwrite("moves", &PokemonSet::moves)
      .def(py::pickle(
          [](const PokemonSet &s) {
            return py::make_tuple(s.species, s.level, s.moves);
          },
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error("PokemonSet: invalid pickle state");
            PokemonSet s;
            s.species = t[0].cast<uint8_t>();
            s.level = t[1].cast<uint8_t>();
            s.moves = t[2].cast<std::array<uint8_t, 4>>();
            return s;
          }))
      .def("__eq__",
           [](const PokemonSet &a, const PokemonSet &b) { return a == b; })
      .def("__lt__",
           [](const PokemonSet &a, const PokemonSet &b) { return a < b; })
      .def("__le__",
           [](const PokemonSet &a, const PokemonSet &b) { return a <= b; })
      .def("__gt__",
           [](const PokemonSet &a, const PokemonSet &b) { return b < a; })
      .def("__ge__",
           [](const PokemonSet &a, const PokemonSet &b) { return b <= a; })
      .def("__hash__", [](const PokemonSet &s) {
        return static_cast<py::ssize_t>(PokemonSet::hash_set(s));
      });

  m.def(
      "species_id", [](const int x) { return PKMN::species_string(x); },
      py::arg("number"));
  m.def(
      "move_id", [](const int x) { return PKMN::move_string(x); },
      py::arg("number"));
  m.def(
      "id_to_species",
      [](const std::string &str) {
        return static_cast<uint8_t>(PKMN::string_to_species(str));
      },
      py::arg("species_id"));
  m.def(
      "id_to_move",
      [](const std::string &str) {
        return static_cast<uint8_t>(PKMN::string_to_move(str));
      },
      py::arg("move_id"));

  m.def(
      "parse_battle",
      [](const std::string &battle_string, uint64_t seed = 0x123456) {
        auto [battle, durations] = Parse::parse_battle(battle_string, seed);
        MCTS::Input input{};
        input.battle = battle;
        input.durations = durations;
        input.result = PKMN::result(battle);
        return py::make_tuple(BattleView{battle}, DurationsView{durations},
                              input.result);
      },
      py::arg("battle_string"), py::arg("seed") = 0x123456);

  m.def(
      "update",
      [](BattleView &battle, DurationsView &durations, uint8_t c1, uint8_t c2) {
        auto options = PKMN::options();
        pkmn_gen1_chance_options chance_options{};
        chance_options.durations = durations.raw;
        PKMN::set(options, chance_options);
        auto result = PKMN::update(battle.raw, c1, c2, options);
        durations.raw = PKMN::durations(options);
        return result;
      },
      py::arg("battle"), py::arg("durations"), py::arg("c1"), py::arg("c2"));

  m.def(
      "battle_string",
      [](const BattleView &battle, const DurationsView &durations) {
        return PKMN::battle_data_to_string(battle.raw, durations.raw);
      },
      py::arg("battle"), py::arg("durations"));

  m.def("load_teams", [](const std::string &path) {
    std::vector<std::vector<PokemonSet>> res{};
    const auto out = _load_teams(path);
    for (const auto &t : out) {
      std::vector<PokemonSet> team{};
      for (const auto &s : t) {
        team.emplace_back(s);
      }
      res.emplace_back(team);
    }
    return res;
  });

  m.def(
      "stats",
      [](const uint8_t species, uint8_t level = 100,
         std::array<uint8_t, 5> evs = {255, 255, 255, 255, 255},
         std::array<uint8_t, 5> dvs = {15, 15, 15, 15, 15}) {
        std::array<uint16_t, 5> stats;
        const auto base_stats =
            get_species_data(static_cast<Species>(species)).base_stats;
        stats[0] =
            Init::compute_stat(base_stats.hp, true, level, evs[0], dvs[0]);
        stats[1] =
            Init::compute_stat(base_stats.atk, false, level, evs[1], dvs[1]);
        stats[2] =
            Init::compute_stat(base_stats.def, false, level, evs[2], dvs[2]);
        stats[3] =
            Init::compute_stat(base_stats.spe, false, level, evs[3], dvs[3]);
        stats[4] =
            Init::compute_stat(base_stats.spc, false, level, evs[4], dvs[4]);
        return stats;
      },
      py::arg("species"), py::arg("level"), py::arg("EVs"), py::arg("DVs"));

  m.def(
      "complete_pokemon_from_set",
      [](PokemonProxy &proxy, const PokemonSet &set) -> void {
        PKMN::Pokemon &pokemon = *proxy.p;

        if (pokemon.species == Species::None) {
          pokemon.species = static_cast<Species>(set.species);
        }
        if (pokemon.level == 0) {
          pokemon.level = set.level;
        }
        if (pokemon.stats == PKMN::Stats{}) {
          const auto base_stats = get_species_data(pokemon.species).base_stats;
          pokemon.stats.hp =
              Init::compute_stat(base_stats.hp, true, pokemon.level);
          pokemon.stats.atk =
              Init::compute_stat(base_stats.atk, false, pokemon.level);
          pokemon.stats.def =
              Init::compute_stat(base_stats.def, false, pokemon.level);
          pokemon.stats.spe =
              Init::compute_stat(base_stats.spe, false, pokemon.level);
          pokemon.stats.spc =
              Init::compute_stat(base_stats.spc, false, pokemon.level);
          pokemon.hp = pokemon.stats.hp;
        }

        if (pokemon.types == 0) {
          const auto types = get_types(pokemon.species);
          pokemon.types = static_cast<uint8_t>(types[0]) |
                          (static_cast<uint8_t>(types[1]) << 4);
        }

        complete_move_set(pokemon.moves, set.moves);
      },
      "Fully initialize empty or partially revealed Pokemon slot from a Set");

  m.def("complete_pokemon_moves",
        [](PokemonProxy &proxy, const PokemonSet &set) -> void {
          PKMN::Pokemon &pokemon = *proxy.p;
          complete_move_set(pokemon.moves, set.moves);
        });

  m.def("complete_active_moves",
        [](ActivePokemonProxy &proxy, const PokemonSet &set) -> void {
          PKMN::ActivePokemon &active = *proxy.p;
          complete_move_set(active.moves, set.moves);
        });

  m.def("copy_moves_to_active",
        [](const PokemonProxy &pokemon, ActivePokemonProxy &active) {
          active.p->moves = pokemon.p->moves;
        });

  m.def("choice_label", [](const SideProxy &side, int choice) {
    return PKMN::side_choice_string(*side.p, static_cast<pkmn_choice>(choice));
  });

  // Fills any Move::None slots in `set.moves` with randomly sampled legal OU
  // moves. Does not duplicate moves already present.
  m.def("fill_random_moveset", [](PokemonSet &set, uint64_t seed = 0) {
    if (seed == 0) {
      seed = std::random_device{}();
    }
    std::mt19937 rng{seed};
    const auto &pool = Format::OU::move_pool(set.species);
    const auto pool_size = Format::OU::move_pool_size(set.species);
    const auto &moves = set.moves;
    std::vector<Move> remaining{};
    for (auto i = 0; i < pool_size; ++i) {
      const auto move = pool[i];
      if (std::find(moves.begin(), moves.end(), static_cast<uint8_t>(move)) ==
          moves.end()) {
        remaining.push_back(move);
      }
    }
    std::shuffle(remaining.begin(), remaining.end(), rng);
    auto it = remaining.begin();
    for (auto &slot : set.moves) {
      if (slot == 0 && it != remaining.end()) {
        slot = static_cast<uint8_t>(*it++);
      }
    }
  });

  m.def("choices",
        [](const BattleView &battle, int result)
            -> std::pair<std::vector<pkmn_choice>, std::vector<pkmn_choice>> {
          return PKMN::choices(battle.raw, static_cast<pkmn_result>(result));
        });

  m.def("result_type",
        [](int result) -> int { return pkmn_result_type(result); });

  m.def("solve_matrix", &solve_matrix, py::arg("row_payoff"),
        py::arg("discretize_factor"));

  m.def(
      "switch_in",
      [](const PokemonProxy &stored, ActivePokemonProxy &active) -> void {
        *active.p = PKMN::switch_in(*stored.p);
      },
      py::arg("pokemon"), py::arg("active"));

  m.def(
      "move_data",
      [](int m) {
        py::dict d;
        const auto &move_data = PKMN::move_data(m);
        d["effect"] = static_cast<int>(move_data.effect);
        d["bp"] = static_cast<int>(move_data.bp);
        d["type"] = static_cast<int>(move_data.type);
        d["accuracy"] = static_cast<int>(move_data.accuracy);
        return d;
      },
      py::arg("move"));

  m.def("get_effectiveness", [](int attacking, int defending) -> int {
    const auto a = static_cast<PKMN::Data::Type>(attacking);
    const auto d = static_cast<PKMN::Data::Type>(defending);
    return static_cast<int>(PKMN::Data::get_effectiveness(a, d));
  });
}

} // namespace Py::PKMN