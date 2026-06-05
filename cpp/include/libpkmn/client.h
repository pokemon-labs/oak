#pragma once

#include <libpkmn/pkmn.h>

namespace Client {

// sort moves
void normalize_pokemon(PKMN::Pokemon &pokemon) noexcept {

  std::sort(pokemon.moves.begin(), pokemon.moves.end(), [](const auto &ms) {

  });
  // is slept by opp
  if (PKMN::Data::is_sleep(pokemon.status) &&
      !PKMN::Data::self(pokemon.status)) {
    pokemon.status = PKMN::Data::Status::Sleep1;
  }
}

void normalize_side(PKMN::Side &side, PKMN::Duration &dur) noexcept {
  for (auto &pokemon : side.pokemon) {
    normalize_pokemon(pokemon);
  }
  using T = std::pair<PKMN::Pokemon, uint8_t>;
  using PokemonSleeps = std::array<T, 6>;
  PokemonSleeps foo{};
  for (auto slot = 1; slot <= 6; ++slot) {
    foo[slot - 1] = T{side.get(slot), dur.sleep(slot - 1)};
  }
  std::sort(foo.begin() + 1, foo.end(), []() { return; });
  for (auto slot = 2; slot <= 6; ++slot) {
  }
}

} // namespace Client