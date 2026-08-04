#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/strings.h>

#include <cassert>

namespace Encode::Battle::Policy {

static constexpr int n_dim =
    static_cast<int>(PKMN::Data::Species::Mew) +
    (static_cast<int>(PKMN::Data::Move::Struggle) - 1); // no Struggle, None

inline uint16_t get_index(const PKMN::Side &side, auto choice) {
  const auto choice_type = choice & 3;
  const auto choice_data = choice >> 2;
  switch (choice_type) {
  case 1: {
    assert(choice_data >= 0 && choice_data <= 4);
    auto moveid =
        static_cast<uint16_t>(side.stored().moves[choice_data - 1].id);
    // assert(moveid < static_cast<uint16_t>(Data::Move::Struggle));
    if (moveid == 0) {
      return 0;
    }
    return moveid - 1;
  }
  case 2: {
    assert(choice_data >= 2 && choice_data <= 6);
    const auto &pokemon = side.get(choice_data);
    auto species = static_cast<uint16_t>(pokemon.species);
    assert(species <= 151);
    return 164 + species - 1;
  }
  case 0: {
    return 0;
  }
  default: {
    assert(false);
    return 0;
  }
  }
}

inline consteval auto get_dim_labels() {
  std::array<std::array<char, 13>, n_dim> result{};
  const auto copy = [](const auto &src, auto &dest) {
    for (auto i = 0; i < src.size(); ++i) {
      dest[i] = src[i];
    }
  };
  auto index = 0;
  for (auto i = 0; i < 164; ++i) {
    copy(PKMN::Data::MOVE_CHAR_ARRAY[i + 1], result[index + i]);
  }
  index += 164;
  for (auto i = 0; i < 151; ++i) {
    copy(PKMN::Data::SPECIES_CHAR_ARRAY[i + 1], result[index + i]);
  }
  return result;
}

constexpr auto dim_labels = get_dim_labels();

} // namespace Encode::Battle::Policy