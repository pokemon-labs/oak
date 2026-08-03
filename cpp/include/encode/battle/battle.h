#pragma once

#include <libpkmn/data.h>
#include <libpkmn/data/status.h>
#include <libpkmn/init.h>
#include <libpkmn/pkmn.h>

#include <array>
#include <cassert>

namespace Encode::Battle {

using PKMN::Data::Move;
using PKMN::Data::Species;
using PKMN::Data::Type;

namespace Status {
constexpr auto get_status_index(auto status, uint8_t sleeps = 0) {
  // brn, par, psn(vol encodes tox), frz
  if (status == PKMN::Data::Status::None) {
    return 0;
  }
  if (!is_sleep(status)) {
    const auto n = std::countr_zero(static_cast<uint8_t>(status)) - 2;
    assert(n >= 1 && n <= 4);
    return n;
  } else {
    if (!self(status)) {
      assert(sleeps > 0);
      return 4 + sleeps;
    } else {
      const auto s = static_cast<uint8_t>(status) & 7;
      assert(s > 0 && s <= 3);
      return 15 - s;
    }
  }
}
static_assert(get_status_index(PKMN::Data::Status::None) == 0);
static_assert(get_status_index(PKMN::Data::Status::Poison) == 1);
static_assert(get_status_index(PKMN::Data::Status::Toxic) == 1);
static_assert(get_status_index(PKMN::Data::Status::Burn) == 2);
static_assert(get_status_index(PKMN::Data::Status::Freeze) == 3);
static_assert(get_status_index(PKMN::Data::Status::Paralysis) == 4);
// further down corresponds to more likely to wake up
static_assert(get_status_index(PKMN::Data::Status::Sleep1, 1) == 5);
static_assert(get_status_index(PKMN::Data::Status::Sleep1, 2) == 6);
static_assert(get_status_index(PKMN::Data::Status::Sleep1, 3) == 7);
static_assert(get_status_index(PKMN::Data::Status::Sleep1, 4) == 8);
static_assert(get_status_index(PKMN::Data::Status::Sleep1, 5) == 9);
static_assert(get_status_index(PKMN::Data::Status::Sleep1, 6) == 10);
static_assert(get_status_index(PKMN::Data::Status::Sleep1, 7) == 11);
static_assert(get_status_index(PKMN::Data::Status::Rest3, 1) == 12);
static_assert(get_status_index(PKMN::Data::Status::Rest2, 2) == 13);
static_assert(get_status_index(PKMN::Data::Status::Rest1, 3) == 14);
constexpr auto n_dim = 15;
} // namespace Status

namespace Pokemon {
constexpr auto n_dim = 151 + Status::n_dim + 1;
constexpr float *write(const PKMN::Pokemon &pokemon, uint8_t sleep, float *t) {
  auto species = static_cast<uint8_t>(pokemon.species);
  assert(species > 0);
  assert(species <= 151);
  t[species - 1] = 1.0;
  t += 151;
  t[Status::get_status_index(pokemon.status, sleep)] = 1.0;
  t += Status::n_dim;
  t[0] = pokemon.hp / (float)pokemon.stats.hp;
  t += 1;
  return t;
}
constexpr void write(const PKMN::Pokemon &pokemon, uint8_t sleep, float *t,
                     uint16_t *index) {
  auto species = static_cast<uint8_t>(pokemon.species);
  *t++ = 1.0;
  *index++ = species - 1;
  *t++ = 1.0;
  *index++ = 151 + Status::get_status_index(pokemon.status, sleep);
  *t++ = pokemon.hp / (float)pokemon.stats.hp;
  *index++ = 151 + Status::n_dim;
}
} // namespace Pokemon

inline constexpr uint8_t ceil_log2_u8(uint8_t x) {
  return (uint8_t)(31 - __builtin_clz((2 * x + 1)));
}

inline consteval bool is_bucket_step(const auto i) {
  return (ceil_log2_u8(i) + 1) == ceil_log2_u8(i + 1);
}

// This determines all values since its an increasing function
static_assert(ceil_log2_u8(0) == 0);
static_assert(is_bucket_step(0));
static_assert(is_bucket_step(1));
static_assert(is_bucket_step(3));
static_assert(is_bucket_step(7));
static_assert(is_bucket_step(15));
static_assert(is_bucket_step(31));
static_assert(ceil_log2_u8(61) == 6);

namespace Moves {
constexpr auto n_moves = static_cast<uint8_t>(Move::Struggle) - 1;
constexpr auto n_pp = 6; // 7 buckets when we allow for 0pp but we don't encode
                         // anything in that case
constexpr auto n_dim = n_moves * n_pp;
static_assert(ceil_log2_u8(61) == 6);
constexpr float *write(const std::array<PKMN::MoveSlot, 4> &moves, float *t) {
  for (auto [id, pp] : moves) {
    if (id != Move::Struggle && id != Move::None && static_cast<bool>(pp)) {
      auto p = ceil_log2_u8(pp) - 1; // no 0 pp
      assert(p >= 0);
      assert(p < n_pp);
      auto i = (static_cast<uint16_t>(id) - 1) * n_pp + p;
      t[i] = 1.0;
    }
  }
  return t + n_dim;
}
constexpr auto write(const std::array<PKMN::MoveSlot, 4> &moves, float *t,
                     uint16_t *index) {
  auto n = 0;
  for (const auto [id, pp] : moves) {
    if (id != Move::Struggle && id != Move::None && static_cast<bool>(pp)) {
      auto p = ceil_log2_u8(pp) - 1;
      assert(p >= 0);
      assert(p < n_pp);
      auto i = (static_cast<uint16_t>(id) - 1) * n_pp + p;
      *index++ = i;
      *t++ = 1.0;
      ++n;
    }
  }
  return n;
}
inline consteval auto get_dim_labels() {
  std::array<std::array<char, 13>, n_dim> result{};
  for (auto i = 0; i < n_moves; ++i) {
    for (auto p = 0; p < n_pp; ++p) {
      auto index = i * n_pp + p;
      result[index] = PKMN::Data::MOVE_CHAR_ARRAY[i + 1];
      result[index][11] = '0' + (p + 1); // one before last cus null terminated?
    }
  }
  return result;
}
constexpr auto dim_labels = get_dim_labels();
} // namespace Moves

namespace Stats {
constexpr float max_stat_value = 999;
constexpr float max_hp_value = PKMN::Init::compute_stat(
    get_species_data(Species::Chansey).base_stats.hp, true);
constexpr auto n_dim = 4;
constexpr float *write(const PKMN::Stats &stats, float *t) {
  t[0] = std::min(1.0f, stats.atk / max_stat_value);
  t[1] = std::min(1.0f, stats.def / max_stat_value);
  t[2] = std::min(1.0f, stats.spe / max_stat_value);
  t[3] = std::min(1.0f, stats.spc / max_stat_value);
  // t[4] = stats.hp / max_hp_value;
  return t + n_dim;
}
constexpr void write(const PKMN::Stats &stats, float *&t, uint16_t *&index,
                     uint16_t &offset) {
  *t++ = std::min(1.0f, stats.atk / max_stat_value);
  *index++ = offset + 0;
  *t++ = std::min(1.0f, stats.def / max_stat_value);
  *index++ = offset + 1;
  *t++ = std::min(1.0f, stats.spe / max_stat_value);
  *index++ = offset + 2;
  *t++ = std::min(1.0f, stats.spc / max_stat_value);
  *index++ = offset + 3;
  // *t++ = stats.hp / max_hp_value;
  // *index++ = offset + 4;
  offset += n_dim;
}
inline consteval auto get_dim_labels() {
  return std::array<std::array<char, 4>, n_dim>{{"ATK", "DEF", "SPE", "SPC"}};
}
constexpr auto dim_labels = get_dim_labels();
} // namespace Stats

namespace Types {
constexpr auto n_dim = static_cast<uint8_t>(Type::Dragon) + 1;
static_assert(n_dim == 15);
constexpr float *write(const uint8_t types, float *t) {
  const uint8_t type_1 = types % 16;
  const uint8_t type_2 = types / 16;
  assert(type_1 < n_dim);
  assert(type_2 < n_dim);
  t[type_1] = 1;
  t[type_2] = 1;
  return t + n_dim;
}
constexpr void write(const uint8_t types, float *&t, uint16_t *&index,
                     uint16_t &offset) {
  const uint8_t type_1 = types % 16;
  const uint8_t type_2 = types / 16;
  assert(type_1 < n_dim);
  assert(type_2 < n_dim);
  *t++ = 1.0f;
  *index++ = offset + type_1;
  if (type_2 != type_1) {
    *t++ = 1.0f;
    *index++ = offset + type_2;
  }
  offset += n_dim;
}
} // namespace Types

namespace Volatiles {
constexpr auto n_dim = 18;
constexpr float *write(const PKMN::Volatiles &vol, float *t) {
  constexpr float chansey_sub = Stats::max_hp_value / 4 + 1;
  // See data layout in extern/engine/src/lib/gen1/readme.md
  // hidden data is replaced with normalized durations
  t[0] = vol.bide();
  t[1] = vol.thrashing();
  t[2] = vol.charging();
  t[3] = vol.binding();
  t[4] = vol.invulnerable();
  t[5] = vol.confusion();
  t[6] = vol.mist();
  t[7] = vol.focus_energy();
  t[8] = vol.substitute();
  t[9] = vol.recharging();
  t[10] = vol.rage();
  t[11] = vol.leech_seed();
  t[12] = vol.toxic();
  t[13] = vol.light_screen();
  t[14] = vol.reflect();
  t[15] = vol.transform();
  // confusion_left
  // attacks (thrashing/binding) left
  // state = (bide damage and rage accuracy)
  t[16] = vol.substitute_hp() / chansey_sub;
  // transform id
  // disable left
  // disable move slot; just zero out the move encoding
  t[17] = vol.toxic_counter() / 16.0;
  return t + n_dim;
}
constexpr void write(const PKMN::Volatiles &vol, float *&t, uint16_t *&index,
                     uint16_t &offset) {
  constexpr float chansey_sub = Stats::max_hp_value / 4 + 1;
  const float vals[n_dim] = {static_cast<float>(vol.bide()),
                             static_cast<float>(vol.thrashing()),
                             static_cast<float>(vol.charging()),
                             static_cast<float>(vol.binding()),
                             static_cast<float>(vol.invulnerable()),
                             static_cast<float>(vol.confusion()),
                             static_cast<float>(vol.mist()),
                             static_cast<float>(vol.focus_energy()),
                             static_cast<float>(vol.substitute()),
                             static_cast<float>(vol.recharging()),
                             static_cast<float>(vol.rage()),
                             static_cast<float>(vol.leech_seed()),
                             static_cast<float>(vol.toxic()),
                             static_cast<float>(vol.light_screen()),
                             static_cast<float>(vol.reflect()),
                             static_cast<float>(vol.transform()),
                             vol.substitute_hp() / chansey_sub,
                             vol.toxic_counter() / 16.0f};
  for (uint16_t i = 0; i < n_dim; ++i) {
    assert(vals[i] >= 0.0);
    assert(vals[i] <= 1.0);
    if (vals[i] != 0.0f) {
      *t++ = vals[i];
      *index++ = offset + i;
    }
  }
  offset += n_dim;
}
inline consteval auto get_dim_labels() {
  return std::array<std::array<char, 13>, n_dim>{
      {"bide", "thrashing", "charging", "binding", "invulner", "confusion",
       "mist", "focus_energy", "substitute", "recharging", "rage", "leech_seed",
       "toxic", "light_screen", "reflect", "transform", "sub_hp"}};
}
constexpr auto dim_labels = get_dim_labels();
} // namespace Volatiles

namespace Duration {
constexpr auto n_confusion = 5;
constexpr auto n_disable = 8;
constexpr auto n_attacking = 3; // bide = thrashing
constexpr auto n_binding = 4;
constexpr auto n_dim = n_confusion + n_disable + n_attacking + n_binding;

constexpr float *write(const PKMN::Duration &duration, float *t) {
  if (const auto confusion = duration.confusion()) {
    assert(confusion <= n_confusion);
    t[confusion - 1] = 1;
  }
  t += n_confusion;

  if (const auto disable = duration.disable()) {
    assert(disable <= n_disable);
    t[disable - 1] = 1;
  }
  t += n_disable;

  if (const auto attacking = duration.attacking()) {
    assert(attacking <= n_attacking);
    t[attacking - 1] = 1;
  }
  t += n_attacking;

  if (const auto binding = duration.binding()) {
    assert(binding <= n_binding);
    t[binding - 1] = 1;
  }
  t += n_binding;
  return t;
}
constexpr void write(const PKMN::Duration &duration, float *&t,
                     uint16_t *&index, uint16_t &offset) {

  if (const auto c = duration.confusion()) {
    *t++ = 1.0f;
    *index++ = offset + (c - 1);
  }
  offset += n_confusion;

  if (const auto d = duration.disable()) {
    *t++ = 1.0f;
    *index++ = offset + (d - 1);
  }
  offset += n_disable;

  if (const auto a = duration.attacking()) {
    *t++ = 1.0f;
    *index++ = offset + (a - 1);
  }
  offset += n_attacking;

  if (const auto b = duration.binding()) {
    *t++ = 1.0f;
    *index++ = offset + (b - 1);
  }
  offset += n_binding;
}

inline consteval auto get_dim_labels() {
  std::array<std::array<char, 10>, n_dim> result;

  auto index = 0;
  for (auto i = 0; i < n_confusion; ++i) {
    result[index + i] = {"confusion"};
    result[index + i][9] = static_cast<char>('1' + i);
  }
  index += n_confusion;

  for (auto i = 0; i < n_disable; ++i) {
    result[index + i] = {"disable"};
    result[index + i][7] = static_cast<char>('1' + i);
  }
  index += n_disable;

  for (auto i = 0; i < n_attacking; ++i) {
    result[index + i] = {"attacking"};
    result[index + i][9] = static_cast<char>('1' + i);
  }
  index += n_attacking;

  for (auto i = 0; i < n_binding; ++i) {
    result[index + i] = {"binding"};
    result[index + i][7] = static_cast<char>('1' + i);
  }
  return result;
}
constexpr auto dim_labels = get_dim_labels();
} // namespace Duration

namespace Active {
constexpr auto n_dim =
    Stats::n_dim + Types::n_dim + Volatiles::n_dim + Duration::n_dim;
constexpr float *write(const PKMN::ActivePokemon &active,
                       const PKMN::Duration &duration, float *t) {
  t = Stats::write(active.stats, t);
  t = Types::write(active.types, t);
  t = Volatiles::write(active.volatiles, t);
  t = Duration::write(duration, t);
  return t;
}
constexpr void write(const PKMN::ActivePokemon &active,
                     const PKMN::Duration &duration, float *&t,
                     uint16_t *&index, uint16_t &offset) {
  Stats::write(active.stats, t, index, offset);
  Types::write(active.types, t, index, offset);
  Volatiles::write(active.volatiles, t, index, offset);
  Duration::write(duration, t, index, offset);
}
inline consteval auto get_dim_labels() {
  std::array<std::array<char, 13>, n_dim> result{};

  const auto copy = [](const auto &src, auto &dest) {
    for (auto i = 0; i < src.size(); ++i) {
      dest[i] = src[i];
    }
  };
  auto index = 0;
  for (auto i = 0; i < Stats::n_dim; ++i) {
    copy(Stats::dim_labels[i], result[index + i]);
  }
  index += Stats::n_dim;
  for (auto i = 0; i < Types::n_dim; ++i) {
    copy(PKMN::Data::TYPE_CHAR_ARRAY[i], result[index + i]);
  }
  index += Types::n_dim;
  for (auto i = 0; i < Volatiles::n_dim; ++i) {
    copy(Volatiles::dim_labels[i], result[index + i]);
  }
  index += Volatiles::n_dim;
  for (auto i = 0; i < Duration::n_dim; ++i) {
    copy(Duration::dim_labels[i], result[index + i]);
  }
  index += Duration::n_dim;
  return result;
}
constexpr auto dim_labels = get_dim_labels();
} // namespace Active

} // namespace Encode::Battle
