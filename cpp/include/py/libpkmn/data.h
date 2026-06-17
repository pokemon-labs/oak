#pragma once

#include <libpkmn/data.h>

#include <string>
#include <cstring>
#include <cstdint>
#include <stdexcept>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

namespace py = pybind11;
using namespace PKMN;
using namespace PKMN::Data;

static inline std::string_view bytes_sv(const py::bytes &b) {
  return {PyBytes_AS_STRING(b.ptr()),
          static_cast<size_t>(PyBytes_GET_SIZE(b.ptr()))};
}

// Read an arbitrary N-bit unsigned field from `bits` starting at `shift`.
template <int N> static constexpr uint64_t bf_get(uint64_t bits, int shift) {
  return (bits >> shift) & ((1ULL << N) - 1);
}

// Write an arbitrary N-bit unsigned field into `bits` at `shift`.
template <int N>
static constexpr uint64_t bf_set(uint64_t bits, int shift, uint64_t val) {
  const uint64_t mask = ((1ULL << N) - 1) << shift;
  return (bits & ~mask) | ((val & ((1ULL << N) - 1)) << shift);
}
struct StatsProxy {
  Stats *p;

  uint16_t get_hp() const { return p->hp; }
  uint16_t get_atk() const { return p->atk; }
  uint16_t get_def() const { return p->def; }
  uint16_t get_spe() const { return p->spe; }
  uint16_t get_spc() const { return p->spc; }

  void set_hp(uint16_t v) { p->hp = v; }
  void set_atk(uint16_t v) { p->atk = v; }
  void set_def(uint16_t v) { p->def = v; }
  void set_spe(uint16_t v) { p->spe = v; }
  void set_spc(uint16_t v) { p->spc = v; }
};

struct MoveSlotProxy {
  MoveSlot *p;

  uint8_t get_id() const { return static_cast<uint8_t>(p->id); }
  uint8_t get_pp() const { return p->pp; }
  void set_id(uint8_t v) { p->id = static_cast<Move>(v); }
  void set_pp(uint8_t v) { p->pp = v; }

  std::string name() const { return PKMN::move_string(p->id); }
};

struct BoostsProxy {
  Boosts *p;

  int8_t get_atk() const { return p->atk(); }
  int8_t get_def() const { return p->def(); }
  int8_t get_spe() const { return p->spe(); }
  int8_t get_spc() const { return p->spc(); }
  int8_t get_acc() const { return p->acc(); }
  int8_t get_eva() const { return p->eva(); }

  void set_atk(int8_t v) { p->set_atk(v); }
  void set_def(int8_t v) { p->set_def(v); }
  void set_spe(int8_t v) { p->set_spe(v); }
  void set_spc(int8_t v) { p->set_spc(v); }
  void set_acc(int8_t v) { p->set_acc(v); }
  void set_eva(int8_t v) { p->set_eva(v); }

  uint32_t get_raw() const {
    uint32_t out;
    std::memcpy(&out, p->bytes, 4);
    return out;
  }
  void set_raw(uint32_t v) { std::memcpy(p->bytes, &v, 4); }
};

struct VolatilesProxy {
  Volatiles *p;

  bool get_bide() const { return p->bide(); }
  bool get_thrashing() const { return p->thrashing(); }
  bool get_multi_hit() const { return p->multi_hit(); }
  bool get_flinch() const { return p->flinch(); }
  bool get_charging() const { return p->charging(); }
  bool get_binding() const { return p->binding(); }
  bool get_invulnerable() const { return p->invulnerable(); }
  bool get_confusion() const { return p->confusion(); }
  bool get_mist() const { return p->mist(); }
  bool get_focus_energy() const { return p->focus_energy(); }
  bool get_substitute() const { return p->substitute(); }
  bool get_recharging() const { return p->recharging(); }
  bool get_rage() const { return p->rage(); }
  bool get_leech_seed() const { return p->leech_seed(); }
  bool get_toxic() const { return p->toxic(); }
  bool get_light_screen() const { return p->light_screen(); }
  bool get_reflect() const { return p->reflect(); }
  bool get_transform() const { return p->transform(); }

  void set_bide(bool v) { p->set_bide(v); }
  void set_thrashing(bool v) { p->set_thrashing(v); }
  void set_multi_hit(bool v) { p->set_multi_hit(v); }
  void set_flinch(bool v) { p->set_flinch(v); }
  void set_charging(bool v) { p->set_charging(v); }
  void set_binding(bool v) { p->set_binding(v); }
  void set_invulnerable(bool v) { p->set_invulnerable(v); }
  void set_confusion(bool v) { p->set_confusion(v); }
  void set_mist(bool v) { p->set_mist(v); }
  void set_focus_energy(bool v) { p->set_focus_energy(v); }
  void set_substitute(bool v) { p->set_substitute(v); }
  void set_recharging(bool v) { p->set_recharging(v); }
  void set_rage(bool v) { p->set_rage(v); }
  void set_leech_seed(bool v) { p->set_leech_seed(v); }
  void set_toxic(bool v) { p->set_toxic(v); }
  void set_light_screen(bool v) { p->set_light_screen(v); }
  void set_reflect(bool v) { p->set_reflect(v); }
  void set_transform(bool v) { p->set_transform(v); }

  uint8_t get_confusion_left() const { return p->confusion_left(); }
  uint8_t get_attacks() const { return p->attacks(); }
  uint8_t get_disable_left() const { return p->disable_left(); }

  void set_confusion_left(uint8_t v) { p->set_confusion_left(v); }
  void set_attacks(uint8_t v) { p->set_attacks(v); }
  void set_disable_left(uint8_t v) { p->set_disable_left(v); }

  uint16_t get_state() const { return p->state(); }
  uint8_t get_substitute_hp() const { return p->substitute_hp(); }
  uint8_t get_transform_species() const { return p->transform_species(); }
  uint8_t get_disable_move() const { return p->disable_move(); }
  uint8_t get_toxic_counter() const { return p->toxic_counter(); }

  void set_state(uint16_t v) {
    p->bits = bf_set<16>(p->bits, 24, static_cast<uint64_t>(v));
  }
  void set_substitute_hp(uint8_t v) {
    p->bits = bf_set<8>(p->bits, 40, static_cast<uint64_t>(v));
  }
  void set_transform_species(uint8_t v) {
    p->bits = bf_set<4>(p->bits, 48, static_cast<uint64_t>(v));
  }
  void set_disable_move(uint8_t v) {
    p->bits = bf_set<3>(p->bits, 56, static_cast<uint64_t>(v));
  }
  void set_toxic_counter(uint8_t v) {
    p->bits = bf_set<5>(p->bits, 59, static_cast<uint64_t>(v));
  }

  uint64_t get_bits() const { return p->bits; }
  void set_bits(uint64_t v) { p->bits = v; }

  std::string to_string() const { return volatiles_to_string(*p); }
};

struct PokemonProxy {
  Pokemon *p;

  StatsProxy stats() { return {&p->stats}; }
  MoveSlotProxy move(int i) {
    if (i < 0 || i > 3)
      throw std::out_of_range("move index must be 0-3");
    return {&p->moves[static_cast<size_t>(i)]};
  }

  uint16_t get_hp() const { return p->hp; }
  uint8_t get_status() const { return static_cast<uint8_t>(p->status); }
  uint8_t get_species() const { return static_cast<uint8_t>(p->species); }
  uint8_t get_types() const { return p->types; }
  uint8_t get_level() const { return p->level; }

  void set_hp(uint16_t v) { p->hp = v; }
  void set_status(uint8_t v) { p->status = static_cast<Status>(v); }
  void set_species(uint8_t v) { p->species = static_cast<Species>(v); }
  void set_types(uint8_t v) { p->types = v; }
  void set_level(uint8_t v) { p->level = v; }

  int percent() const { return p->percent(); }

  std::string species_name() const {
    return PKMN::species_string(p->species);
  }
  std::string status_name() const { return status_string(p->status); }
  std::string to_string() const {
    return pokemon_to_string(reinterpret_cast<const uint8_t *>(p));
  }
};

struct ActivePokemonProxy {
  ActivePokemon *p;

  StatsProxy stats() { return {&p->stats}; }
  BoostsProxy boosts() { return {&p->boosts}; }
  VolatilesProxy volatiles() { return {&p->volatiles}; }
  MoveSlotProxy move(int i) {
    if (i < 0 || i > 3)
      throw std::out_of_range("move index must be 0-3");
    return {&p->moves[static_cast<size_t>(i)]};
  }

  uint8_t get_species() const { return static_cast<uint8_t>(p->species); }
  uint8_t get_types() const { return p->types; }
  void set_species(uint8_t v) { p->species = static_cast<Species>(v); }
  void set_types(uint8_t v) { p->types = v; }

  std::string species_name() const {
    return PKMN::species_string(p->species);
  }
};

struct SideProxy {
  Side *p;

  PokemonProxy pokemon(int i) {
    if (i < 0 || i > 5)
      throw std::out_of_range("pokemon index must be 0-5");
    return {&p->pokemon[static_cast<size_t>(i)]};
  }
  // slot is 1-indexed, uses order[] just like the C++ Side::get()
  PokemonProxy slot(int s) {
    if (s < 1 || s > 6)
      throw std::out_of_range("slot must be 1-6");
    return {&p->get(s)};
  }
  ActivePokemonProxy active() { return {&p->active}; }
  PokemonProxy stored() { return {&p->stored()}; }

  std::array<uint8_t, 6> get_order() const { return p->order; }
  void set_order(std::array<uint8_t, 6> v) { p->order = v; }

  uint8_t get_last_selected_move() const {
    return static_cast<uint8_t>(p->last_selected_move);
  }
  uint8_t get_last_used_move() const {
    return static_cast<uint8_t>(p->last_used_move);
  }
  void set_last_selected_move(uint8_t v) {
    p->last_selected_move = static_cast<Move>(v);
  }
  void set_last_used_move(uint8_t v) {
    p->last_used_move = static_cast<Move>(v);
  }
};

struct DurationProxy {
  Duration *p;

  uint8_t sleep(int slot) const {
    if (slot < 0 || slot > 5)
      throw std::out_of_range("slot must be 0-5");
    return p->sleep(slot);
  }
  void set_sleep(int slot, uint8_t v) {
    if (slot < 0 || slot > 5)
      throw std::out_of_range("slot must be 0-5");
    p->set_sleep(slot, v);
  }

  uint8_t get_confusion() const { return p->confusion(); }
  uint8_t get_disable() const { return p->disable(); }
  uint8_t get_attacking() const { return p->attacking(); }
  uint8_t get_binding() const { return p->binding(); }

  void set_confusion(uint8_t v) { p->set_confusion(v); }
  void set_disable(uint8_t v) { p->set_disable(v); }
  void set_attacking(uint8_t v) { p->set_attacking(v); }
  void set_binding(uint8_t v) { p->set_binding(v); }

  uint32_t get_raw() const { return p->data; }
  void set_raw(uint32_t v) { p->data = v; }
};

struct DurationsView {
  pkmn_gen1_chance_durations raw{};

  explicit DurationsView() = default;
  explicit DurationsView(pkmn_gen1_chance_durations &durations)
      : raw{durations} {}
  explicit DurationsView(py::bytes b) {
    auto sv = bytes_sv(b);
    if (sv.size() != PKMN_GEN1_CHANCE_DURATIONS_SIZE)
      throw std::runtime_error("Durations: expected " +
                               std::to_string(PKMN_GEN1_CHANCE_DURATIONS_SIZE) +
                               " bytes, got " + std::to_string(sv.size()));
    std::memcpy(&raw, sv.data(), PKMN_GEN1_CHANCE_DURATIONS_SIZE);
  }

  Durations &durations() { return view(raw); }
  const Durations &durations() const { return view(raw); }

  DurationProxy get(int i) {
    if (i < 0 || i > 1)
      throw std::out_of_range("index must be 0 or 1");
    return {&durations().get(i)};
  }

  py::bytes bytes() const {
    return py::bytes(reinterpret_cast<const char *>(&raw),
                     PKMN_GEN1_CHANCE_DURATIONS_SIZE);
  }
};

struct BattleView {
  pkmn_gen1_battle raw{};

  explicit BattleView() = default;
  explicit BattleView(pkmn_gen1_battle battle) : raw{battle} {}
  explicit BattleView(py::bytes b) {
    auto sv = bytes_sv(b);
    if (sv.size() != PKMN_GEN1_BATTLE_SIZE)
      throw std::runtime_error("Battle: expected " +
                               std::to_string(PKMN_GEN1_BATTLE_SIZE) +
                               " bytes, got " + std::to_string(sv.size()));
    std::memcpy(&raw, sv.data(), PKMN_GEN1_BATTLE_SIZE);
  }

  Battle &battle() { return view(raw); }
  const Battle &battle() const { return view(raw); }

  SideProxy side(int i) {
    if (i < 0 || i > 1)
      throw std::out_of_range("side index must be 0 or 1");
    return {&battle().sides[static_cast<size_t>(i)]};
  }

  uint16_t get_turn() const { return battle().turn; }
  uint16_t get_last_damage() const { return battle().last_damage; }
  uint64_t get_rng() const { return battle().rng; }

  void set_turn(uint16_t v) { battle().turn = v; }
  void set_last_damage(uint16_t v) { battle().last_damage = v; }
  void set_rng(uint64_t v) { battle().rng = v; }

  py::bytes bytes() const {
    return py::bytes(reinterpret_cast<const char *>(&raw),
                     PKMN_GEN1_BATTLE_SIZE);
  }

  std::string to_string() const { return PKMN::to_string(raw); }
};

}