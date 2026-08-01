#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/key.h>
#include <libpkmn/data/status.h>
#include <nn/affine.h>
#include <nn/ffn.h>

#include <map>
#include <memory>

namespace NN::Battle {

template <typename T> struct SideCache {

  using Embedding = std::unique_ptr<T[]>;
  static constexpr auto max_active_input_size = 28;
  static constexpr auto max_active_moves_input_size = 4;
  static constexpr bool is_integral = std::is_integral_v<T>;

  template <Activation activation>
  static void propagate(auto &network, float const *input,
                        uint16_t const *indices, T *embedding, uint16_t n) {
    static thread_local std::vector<float> output;
    const auto dim = network.template layer<1>().out_dim;
    output.reserve(dim);
    if constexpr (is_integral) {
      network.template propagate<activation, activation>(input, indices,
                                                         output.data(), n);
      std::transform(output.begin(), output.begin() + dim, embedding,
                     [](const auto f) { return static_cast<T>(127 * f); });
    } else {
      network.template propagate<activation, activation>(input, indices,
                                                         embedding, n);
    }
  }

  struct PokemonCache {
    std::array<Embedding, Encode::Battle::Key::n_pokemon> data;
    const T *get(const PKMN::Pokemon &pokemon, uint8_t sleep) const {
      auto key = Encode::Battle::Key::get_key(pokemon, sleep);
      return data[key];
    }
  };
  struct PokemonMovesCache {
    std::array<Embedding, Encode::Battle::Key::n_moves> data;
    const T *get(const PKMN::Pokemon &pokemon) const {
      auto key = Encode::Battle::Key::get_key(pokemon.moves);
      return data[key];
    }
  };
  struct ActiveMovesCache {
    using Moves = std::array<PKMN::MoveSlot, 4>;
    std::map<Moves, Embedding> data;
    std::array<float, max_active_moves_input_size> encoding_input;
    std::array<uint16_t, max_active_moves_input_size> encoding_indices;
  };

  struct ActiveCache {
    std::map<Encode::Battle::Key::ActiveKey, Embedding> data;
    std::array<float, max_active_input_size> encoding_input;
    std::array<uint16_t, max_active_input_size> encoding_indices;

    template <Activation activation>
    const T *get(auto &active_net, const PKMN::ActivePokemon &active,
                 const PKMN::Duration &duration) {
      const auto dim = active_net.template layer<1>().out_dim;
      auto key = Encode::Battle::Key::get_key(active, duration);
      if (data.find(key) != data.end()) {
        return data[key];
      } else {
        auto *input = encoding_input.data();
        auto *indices = encoding_indices.data();
        uint16_t offset = 0;
        const auto n =
            Encode::Battle::Moves::write(active.moves, input, indices, offset);
        data[key] = std::make_unique<T[]>(dim);
        auto *embedding = data(key);
        SideCache::propagate<activation>(active_net, input, indices, embedding);
        encoding_input = {};
        encoding_indices = {};
      }
    }
  };

  // consistency
  template <typename U> using Side = std::array<U, 6>;
  Side<PKMN::Pokemon> reference;
  // caches
  Side<ActiveCache> active;
  Side<ActiveMovesCache> active_moves;
  Side<PokemonCache> pokemon;
  Side<PokemonMovesCache> pokemon_moves;
  // work shit
  std::vector<float> embedding;

  template <Activation activation>
  void precompute(const PKMN::Side &side, auto index, auto &pokemon_net) {
    reference[index] = side.pokemon[index];
    // pokemon
    const auto get_entry = [this, &pokemon_net](const auto &pokemon,
                                                const auto sleep) {
      std::array<uint16_t, Encode::Battle::Pokemon::n_dim> encoding_indices{};
      std::array<float, Encode::Battle::Pokemon::n_dim> encoding_input{};
      float *input = encoding_input.data();
      uint16_t *indices = encoding_indices.data();
      uint16_t _ = 0;
      Encode::Battle::Pokemon::write(pokemon, sleep, input, indices);
      uint32_t n = std::distance(encoding_input.data(), input);
      auto *embedding =
          this->data(Encode::Battle::Key::get_key(pokemon, sleep));
      SideCache::propagate<activation>(pokemon_net, input, indices, embedding,
                                       n);
    };

    using PKMN::Data::Status;

    constexpr std::array<Status, 8> status_array{
        Status::None,      Status::Poison, Status::Burn,  Status::Freeze,
        Status::Paralysis, Status::Rest1,  Status::Rest2, Status::Rest3};

    for (auto hp = 1; hp <= 50; ++hp) {
      // TODO
      pokemon.hp = pokemon.stats.hp * hp / 50;
      // non slept status conditions
      for (const auto status : status_array) {
        pokemon.status = status;
        get_entry(pokemon, 0);
      }
      // slept
      pokemon.status = Status::Sleep1;
      for (auto sleep = 1; sleep <= 7; ++sleep) {
        get_entry(pokemon, sleep);
      }
    }
  }

  void clear() {
    active.data.clear();
    active_moves.data.clear();
    for (auto &embedding : pokemon.data) {
      embedding.reset();
    }
    for (auto &embedding : pokemon_moves.data) {
      embedding.reset();
    }
  }
};

auto quantize_cache(const SideCache<float> &cache) {

  const auto copy_array = [](const auto &src, auto &dest, auto dim) {
    std::transform(src.begin(), src.end(), dest.begin(), dest.end(),
                   dest.begin(), [dim](auto s, auto &d) {
                     delete[] d;
                     d = new uint8_t[dim];
                     std::transform(s, s + dim, d, [](float x) {
                       return static_cast<uint8_t>(x * 127);
                     });
                   });
  };

  SideCache<uint8_t> quantized;
}

} // namespace NN::Battle
