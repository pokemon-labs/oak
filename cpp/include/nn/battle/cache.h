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

  void precompute(auto &network, const PKMN::Side &side, auto index) {
    auto &pokemon_data = pokemon_cache[index].data;
    const auto pokemon_dim = network.pokemon_net.template layer<1>().out_dim;
    const auto get_pokemon_embedding =
        [&pokemon_data, &network, pokemon_dim](
            const auto &pokemon, const auto bucket, const auto sleep) {
          using namespace Encode::Battle::Pokemon;
          uint16_t indices[n_nonzero];
          float input[n_nonzero];
          Encode::Battle::Pokemon::write(pokemon, sleep, input, indices);
          // const auto key = Encode::Battle::Key::get_key(pokemon, sleep);
          const auto key =
              Encode::Battle::Status::get_status_index(pokemon.status, sleep) +
              (bucket - 1) * Encode::Battle::Status::n_dim;
          auto &u = pokemon_data[key];
          u.reset(new T[pokemon_dim]);
          auto *embedding = u.get();
          network.template propagate_embedding<Embedding_::Pokemon, T>(
              input, indices, embedding, n_nonzero);
        };
    auto pokemon = side.pokemon[index];
    reference[index] = pokemon;
    using PKMN::Data::Status;
    constexpr std::array<Status, 8> status_array{
        Status::None,      Status::Poison, Status::Burn,  Status::Freeze,
        Status::Paralysis, Status::Rest1,  Status::Rest2, Status::Rest3};
    for (auto bucket = 1; bucket <= 50; ++bucket) {
      pokemon.hp = pokemon.stats.hp * bucket / 50;
      for (const auto status : status_array) {
        pokemon.status = status;
        get_pokemon_embedding(pokemon, bucket, 0);
      }
      pokemon.status = Status::Sleep1;
      for (auto sleep = 1; sleep <= 7; ++sleep) {
        get_pokemon_embedding(pokemon, bucket, sleep);
      }
    }

    auto &moves_data = pokemon_moves_cache[index].data;
    const auto moves_dim = network.moves_net.template layer<1>().out_dim;
    const auto get_moves_embedding = [&moves_data, &network,
                                      moves_dim](const auto &moves) {
      using namespace Encode::Battle::Moves;
      uint16_t indices[n_nonzero];
      float input[n_nonzero];
      const auto n = write(moves, input, indices);
      const auto key = Encode::Battle::Key::get_key(moves);
      auto &u = moves_data[key];
      u.reset(new T[moves_dim]);
      auto *embedding = u.get();
      network.template propagate_embedding<Embedding_::Moves, T>(input, indices,
                                                                 embedding, n);
    };

    using Encode::Battle::Key::n_moves;
    using Encode::Battle::Key::n_pp;
    for (auto m = 0; m < n_moves; ++m) {
      auto m_ = m;
      for (auto i = 0; i < 4; ++i) {
        auto pp_bucket = m_ % n_pp;
        uint8_t repr = std::pow(2, pp_bucket) - 1;
        pokemon.moves[i].pp = repr;
        m_ /= n_pp;
      }
      get_moves_embedding(pokemon.moves);
    }

    assert(std::all_of(pokemon_data.begin(), pokemon_data.end(),
                       [](auto &x) { return static_cast<bool>(x); }));
    assert(std::all_of(moves_data.begin(), moves_data.end(),
                       [](auto &x) { return static_cast<bool>(x); }));
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
