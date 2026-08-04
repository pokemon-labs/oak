#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/key.h>
#include <libpkmn/data/status.h>
#include <nn/affine.h>
#include <nn/battle/network.h>
#include <nn/ffn.h>

#include <map>
#include <memory>

namespace NN::Battle {

template <typename T> struct SideCache {
  using value_type = T;
  using Embedding = std::unique_ptr<T[]>;
  static constexpr auto max_active_input_size = 28;
  static constexpr auto max_active_moves_input_size = 4;

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

    const T *get(auto &network, const Moves &moves) {
      const auto dim = network.moves_net.template layer<1>().out_dim;
      auto key = Encode::Battle::Key::get_key_active(moves);
      auto [it, inserted] = data.try_emplace(key);
      if (inserted) {
        const auto n = Encode::Battle::Moves::write(
            moves, encoding_input.data(), encoding_indices.data());
        it->second = std::make_unique<T[]>(dim);
        network.template propagate_embedding<Embedding_::Moves, T>(
            encoding_input.data(), encoding_indices.data(), it->second.get(),
            n);
      } else {
        return it->second;
      }
    }
  };

  struct ActiveCache {
    std::map<Encode::Battle::Key::ActiveKey, Embedding> data;
    std::array<float, max_active_input_size> encoding_input;
    std::array<uint16_t, max_active_input_size> encoding_indices;

    const T *get(auto &network, const PKMN::ActivePokemon &active,
                 const PKMN::Duration &duration) {
      const auto dim = network.active_net.template layer<1>().out_dim;
      auto key = Encode::Battle::Key::get_key(active, duration);
      if (data.find(key) != data.end()) {
        return data[key];
      } else {
        auto *input = encoding_input.data();
        auto *indices = encoding_indices.data();
        const auto n =
            Encode::Battle::Moves::write(active.moves, input, indices);
        data[key] = std::make_unique<T[]>(dim);
        auto *embedding = data(key);
        network.template propagate_embedding<Embedding_::Active, T>(
            network, input, indices, embedding);
        encoding_input = {};
        encoding_indices = {};
      }
    }
  };

  // consistency
  template <typename U> using Side = std::array<U, 6>;
  Side<PKMN::Pokemon> reference;
  // caches
  Side<ActiveCache> active_cache;
  Side<ActiveMovesCache> active_moves_cache;
  Side<PokemonCache> pokemon_cache;
  Side<PokemonMovesCache> pokemon_moves_cache;
  // work shit
  std::vector<float> embedding;

  void precompute(auto &network, const PKMN::Side &side, auto index) {
    auto pokemon = side.pokemon[index];
    reference[index] = pokemon;

    auto &pokemon_data = pokemon_cache[index].data;
    const auto dim = network.pokemon_net.template layer<1>().out_dim;
    // pokemon
    const auto get_entry = [&pokemon_data, &network, dim](const auto &pokemon,
                                                          const auto sleep) {
      std::array<uint16_t, Encode::Battle::Pokemon::n_dim> encoding_indices{};
      std::array<float, Encode::Battle::Pokemon::n_dim> encoding_input{};
      float *input = encoding_input.data();
      uint16_t *indices = encoding_indices.data();
      Encode::Battle::Pokemon::write(pokemon, sleep, input, indices);
      const auto key = Encode::Battle::Key::get_key(pokemon, sleep);
      auto &u = pokemon_data[key];
      u.reset(new T[dim]);
      auto *embedding = u.get();
      assert(embedding != nullptr);
      network.template propagate_embedding<Embedding_::Pokemon, T>(
          input, indices, embedding, 3);
      // std::cout << "dim: " << dim << " ";
      // for (auto i = 0; i < dim; ++i) {
      //   std::cout << (float)embedding[i] << ' ';
      // }
      // std::cout << '\n';
      std::cout << (int)key << ' ';
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

    // assert(std::all_of(data.begin(), data.end(),
    //                    [](auto x) { return static_cast<bool>(x); }));
  }

  void clear() {
    active_cache.data.clear();
    active_moves_cache.data.clear();
    for (auto &embedding : pokemon_cache.data) {
      embedding.reset();
    }
    for (auto &embedding : pokemon_moves_cache.data) {
      embedding.reset();
    }
  }
};

auto quantize_cache(const SideCache<float> &cache, uint32_t pokemon_dim,
                    uint32_t active_dim, uint32_t moves_dim) {
  const auto copy_array = [](const auto &src, auto &dest, auto dim) {
    for (std::size_t i = 0; i < src.size(); ++i) {
      dest[i].reset();
      if (!src[i]) {
        continue;
      }
      dest[i] = std::make_unique<uint8_t[]>(dim);
      std::transform(src[i].get(), src[i].get() + dim, dest[i].get(),
                     [](float x) { return static_cast<uint8_t>(x * 127); });
    }
  };

  const auto copy_map = [](const auto &src, auto &dest, auto dim) {
    for (const auto &[key, embedding] : src) {
      auto quantized = std::make_unique<uint8_t[]>(dim);
      std::transform(embedding.get(), embedding.get() + dim, quantized.get(),
                     [](float x) { return static_cast<uint8_t>(x * 127); });
      dest.emplace(key, std::move(quantized));
    }
  };

  SideCache<uint8_t> quantized{};
  quantized.reference = cache.reference;
  for (auto index = 0; index < 6; ++index) {
    copy_array(cache.pokemon_cache[index].data,
               quantized.pokemon_cache[index].data, pokemon_dim);
    copy_array(cache.pokemon_moves_cache[index].data,
               quantized.pokemon_moves_cache[index].data, moves_dim);
    copy_map(cache.active_cache[index].data, quantized.active_cache[index].data,
             active_dim);
    copy_map(cache.active_moves_cache[index].data,
             quantized.active_moves_cache[index].data, moves_dim);
  }
  return quantized;
}

} // namespace NN::Battle
