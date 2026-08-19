#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/key.h>
#include <libpkmn/data/status.h>
#include <nn/affine.h>
#include <nn/ffn.h>

#include <map>
#include <memory>

namespace NN::Battle {

template <typename T> using EmbeddingT = std::unique_ptr<T[]>;

using PKMN::Data::Status;

template <typename T> struct PokemonCache {

  // Encode does not have a dimension for no status
  static constexpr auto n_status = Encode::Battle::Status::n_dim + 1;
  // We only encode whether the move has pp, so 2^4 for the moveset
  static constexpr auto n_pp = 16;
  // For a stored pokemon, the move pp and status features are the only ones
  // that can change over the course of the game (hp is not a part of the input
  // to the pokemon embedding)
  using Key = uint8_t;
  static constexpr Key n_embeddings = n_status * n_pp;
  // All status consditions that dont use sleep duration
  static constexpr std::array<Status, 8> status_array{
      Status::None,      Status::Poison, Status::Burn,  Status::Freeze,
      Status::Paralysis, Status::Rest1,  Status::Rest2, Status::Rest3};

  static constexpr bool is_integral{std::is_integral_v<T>};
  using Embedding = EmbeddingT<T>;

  uint32_t dim;
  std::array<Embedding, n_embeddings> embeddings;
  // work
  std::vector<float> embedding;

  PokemonCache(uint32_t dim = 0) : dim{dim}, embedding{} {
    for (auto &embedding : embeddings) {
      embedding = std::make_unique<T[]>(dim);
    }
    if constexpr (is_integral) {
      embedding.resize(dim);
    }
  }

  PokemonCache &operator=(const PokemonCache &other) {
    dim = other.dim;
    for (auto i = 0; i < n_embeddings; ++i) {
      embeddings[i] = std::make_unique<T[]>(dim);
      const auto *source = other.embeddings[i].get();
      std::copy(source, source + dim, embeddings[i].get());
    }
    embedding = other.embedding;
    return *this;
  }

  template <typename U> PokemonCache &operator=(const PokemonCache<U> &other) {
    dim = other.dim;
    for (auto i = 0; i < n_embeddings; ++i) {
      embeddings[i] = std::make_unique<T[]>(dim);
      const auto *source = other.embeddings[i].get();
      constexpr float scale =
          std::is_floating_point_v<U> && std::is_integral_v<T> ? 127.0f : 1.0f;
      std::transform(source, source + dim, embeddings[i].get(),
                     [scale](const U x) { return static_cast<T>(x * scale); });
    }
    embedding = other.embedding;
    return *this;
  }

  inline T *data(Key key) const { return embeddings[key].get(); }

  // iterate through all move pp/status combinations for a pokemon and store
  // embedding
  template <Activation activation>
  void fill(EmbeddingNet &pokemon_net, const PKMN::Pokemon &base_pokemon) {
    assert(dim == pokemon_net.layer<1>().out_dim);

    auto pokemon = base_pokemon;
    // iterate through all 16 has-pp combinations
    for (auto m = 0; m < n_pp; ++m) {
      // set move pp based on bits of m
      for (auto i = 0; i < 4; ++i) {
        pokemon.moves[i].pp = m & (1 << i);
      }

      const auto get_entry = [this, &pokemon_net](const auto &pokemon,
                                                  const auto sleep) {
        std::array<uint16_t, Encode::Battle::Pokemon::n_dim> encoding_indices{};
        std::array<float, Encode::Battle::Pokemon::n_dim> encoding_input{};
        float *input = encoding_input.data();
        uint16_t *indices = encoding_indices.data();
        Encode::Battle::Pokemon::write(pokemon, sleep, input, indices);
        uint32_t n = std::distance(encoding_input.data(), input);
        auto *embedding_data =
            this->data(Encode::Battle::pokemon_key(pokemon, sleep));
        if constexpr (is_integral) {
          pokemon_net.propagate<activation, activation>(encoding_input.data(),
                                                        encoding_indices.data(),
                                                        embedding.data(), n);
          std::transform(embedding.begin(), embedding.end(), embedding_data,
                         [](const auto f) { return static_cast<T>(127 * f); });
        } else {
          pokemon_net.propagate<activation, activation>(encoding_input.data(),
                                                        encoding_indices.data(),
                                                        embedding_data, n);
        }
      };

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

  const T *get(const auto &pokemon, const auto sleep) const {
    const auto key = Encode::Battle::pokemon_key(pokemon, sleep);
    return data(key);
  }
};

template <typename T> struct ActivePokemonCache {

  static constexpr bool is_integral{std::is_integral_v<T>};
  using Embedding = EmbeddingT<T>;
  using PokemonKey = PokemonCache<T>::Key;
  using Key = std::pair<PKMN::ActivePokemon, PokemonKey>;

  uint32_t dim;
  std::map<Key, Embedding> embeddings;
  // workspace
  std::array<float, Encode::Battle::ActivePokemon::n_dim> encoding_input;
  std::array<uint16_t, Encode::Battle::ActivePokemon::n_dim> encoding_indices;
  std::vector<float> embedding;

  ActivePokemonCache(uint32_t dim = 0) : dim{dim} {
    if constexpr (is_integral) {
      embedding.resize(dim);
    }
  }

  ActivePokemonCache &operator=(const ActivePokemonCache &other) {
    dim = other.dim;
    for (const auto &p : other.embeddings) {
      embeddings[p.first] = std::make_unique<T[]>(dim);
      const auto *source = p.second.get();
      std::copy(source, source + dim, embeddings[p.first].get());
    }
    if constexpr (is_integral) {
      embedding.resize(dim);
    }
    return *this;
  }

  template <typename U>
  ActivePokemonCache &operator=(const ActivePokemonCache<U> &other) {
    dim = other.dim;
    for (const auto &p : other.embeddings) {
      embeddings[p.first] = std::make_unique<T[]>(dim);
      const auto *source = p.second.get();
      constexpr float scale =
          std::is_floating_point_v<U> && std::is_integral_v<T> ? 127.0f : 1.0f;
      std::transform(source, source + dim, embeddings[p.first].get(),
                     [scale](const U x) { return static_cast<T>(x * scale); });
    }
    if constexpr (is_integral) {
      embedding.resize(dim);
    }
    return *this;
  }

  auto *data(const auto key) { return embeddings[key].get(); }

  template <Activation activation>
  const T *get(EmbeddingNet &active_net, const auto &active,
               const auto &pokemon, const auto &duration) {
    const auto key =
        Key{active, Encode::Battle::pokemon_key(pokemon, duration.sleep(0))};
    if (embeddings.find(key) != embeddings.end()) {
      const auto embedding_data = data(key);
      assert(embedding_data != nullptr);
      return embedding_data;
    } else {
      auto *input = encoding_input.data();
      auto *indices = encoding_indices.data();
      Encode::Battle::ActivePokemon::write(pokemon, active, duration, input,
                                           indices);
      const auto n = std::distance(encoding_input.data(), input);

      embeddings[key] = std::make_unique<T[]>(dim);
      auto *embedding_data = data(key);

      if constexpr (is_integral) {
        active_net.propagate<activation, activation>(encoding_input.data(),
                                                     encoding_indices.data(),
                                                     embedding.data(), n);
        std::transform(embedding.begin(), embedding.end(), embedding_data,
                       [](const auto f) { return static_cast<T>(127 * f); });
      } else {
        active_net.propagate<activation, activation>(
            encoding_input.data(), encoding_indices.data(), embedding_data, n);
      }

      std::fill(encoding_input.begin(), encoding_input.begin() + n, 0);
      std::fill(encoding_indices.begin(), encoding_indices.begin() + n, 0);

      return embedding_data;
    }
  }
};

template <typename T> struct BattleCache {

  template <typename U> using SideSet = std::array<U, 6>;
  template <typename U> using BattleSet = std::array<std::array<U, 6>, 2>;
  BattleSet<PokemonCache<T>> pokemon;
  BattleSet<ActivePokemonCache<T>> active;

  using Team = std::array<PKMN::Set, 6>;
  std::array<Team, 2> teams;

  BattleCache(uint32_t pod = 0, uint32_t aod = 0)
      : pokemon{SideSet<PokemonCache<T>>{pod, pod, pod, pod, pod, pod},
                {pod, pod, pod, pod, pod, pod}},
        active{SideSet<ActivePokemonCache<T>>{aod, aod, aod, aod, aod, aod},
               {aod, aod, aod, aod, aod, aod}} {}

  BattleCache(const BattleCache &other) {
    for (auto s = 0; s < 2; ++s) {
      for (auto p = 0; p < 6; ++p) {
        pokemon[s][p] = other.pokemon[s][p];
        active[s][p] = other.active[s][p];
      }
    }
  }

  BattleCache &operator=(const BattleCache &other) = default;
  template <typename U> BattleCache &operator=(const BattleCache<U> &other) {
    for (auto s = 0; s < 2; ++s) {
      for (auto p = 0; p < 6; ++p) {
        pokemon[s][p] = other.pokemon[s][p];
        active[s][p] = other.active[s][p];
      }
    }
    return *this;
  }

  template <Activation activation>
  void fill(EmbeddingNet &pokemon_net, const PKMN::Battle &battle) {
    for (auto s = 0; s < 2; ++s) {
      for (auto p = 0; p < 6; ++p) {
        const auto &poke = battle.sides[s].pokemon[p];
        pokemon[s][p].template fill<activation>(pokemon_net, poke);
        // teams[s][p] = PKMN::Set{poke.species, poke.moves};
        // std::transform(teams[s][p])
      }
    }
  }
};

} // namespace NN::Battle
