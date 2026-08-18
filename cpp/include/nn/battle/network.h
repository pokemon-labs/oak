#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/policy.h>
#include <nn/battle/main-net.h>
#include <nn/battle/quantized/main-net.h>
#include <nn/default-hyperparameters.h>
#include <nn/ffn.h>
#include <util/random.h>

namespace NN::Battle {

enum class Embedding_ {
  Pokemon,
  Active,
  Moves,
};

struct NetworkBase {
  virtual std::tuple<int, int, int, int> shape() const noexcept = 0;
  virtual ~NetworkBase() = default;
  virtual MainNet *main_net_float() = 0;
  virtual int activation_type() const = 0;
  EmbeddingNet pokemon_net;
  EmbeddingNet active_net;
  EmbeddingNet moves_net;
  uint32_t pokemon_out_dim() const noexcept {
    return this->pokemon_net.layer<1>().out_dim;
  }
  uint32_t active_out_dim() const noexcept {
    return this->active_net.layer<1>().out_dim;
  }
  uint32_t moves_out_dim() const noexcept {
    return this->moves_net.layer<1>().out_dim;
  }
  uint32_t side_embedding_dim() const {
    return active_out_dim() + 6 * (pokemon_out_dim() + moves_out_dim());
  }
  void resize(uint32_t ph, uint32_t po, uint32_t ah, uint32_t ao, uint32_t mh,
              uint32_t mo, uint32_t h, uint32_t value, uint32_t policy) {
    if (auto *main = main_net_float()) {
      pokemon_net.resize(Encode::Battle::Pokemon::n_dim, ph, po);
      active_net.resize(Encode::Battle::Active::n_dim, ah, ao);
      moves_net.resize(Encode::Battle::Moves::n_dim, mh, mo);
      main->resize(2 * side_embedding_dim(), h, value, policy);
    } else {
      throw std::runtime_error{"Attempting to resize quantized network"};
    }
  }
  void initialize(auto &device) {
    if (auto *main = main_net_float()) {
      pokemon_net.initialize(device);
      active_net.initialize(device);
      moves_net.initialize(device);
      main->initialize(device);
    } else {
      throw std::runtime_error{"Attempting to initialize quantized network"};
    }
  }
  template <Embedding_ emb, typename T, Activation activation>
  void propagate_embedding(float *input, uint16_t *indices, T *embedding,
                           uint16_t n) {
    static thread_local std::vector<float> temp;
    const auto go = [&](EmbeddingNet &net) {
      if constexpr (std::is_integral_v<T>) {
        const auto dim = net.layer<1>().out_dim;
        temp.reserve(dim);
        net.propagate<activation, activation>(input, indices, temp.data(), n);
        std::transform(temp.begin(), temp.begin() + dim, embedding,
                       [](const auto f) { return static_cast<T>(127 * f); });
      } else {
        net.propagate<activation, activation>(input, indices, embedding, n);
      }
    };
    if constexpr (emb == Embedding_::Pokemon) {
      go(pokemon_net);
    } else if constexpr (emb == Embedding_::Active) {
      go(active_net);
    } else if constexpr (emb == Embedding_::Moves) {
      go(moves_net);
    } else {
      static_assert(emb != emb);
    }
  }
};

template <typename Main, Activation activation>
class NetworkImpl : public NetworkBase {
public:
  static_assert(activation == Activation::relu ||
                activation == Activation::clamp ||
                activation == Activation::relu_scaled);
  using T = typename Main::T;
  static constexpr auto act = activation;
  Main main_net;
  std::tuple<int, int, int, int> shape() const noexcept {
    return main_net.shape();
  }
  MainNet *main_net_float() override {
    if constexpr (std::is_same_v<Main, MainNet>) {
      return &main_net;
    } else {
      return nullptr;
    }
  }
  int activation_type() const { return static_cast<int>(act); }
};

template <Activation activation>
using FNetwork = NetworkImpl<MainNet, activation>;
using Network = FNetwork<Activation::relu>;
using NetworkClamped = FNetwork<Activation::clamp>;
using NetworkScaled = FNetwork<Activation::relu_scaled>;
template <int In, int Hidden, int ValueHidden, int PolicyHidden>
using QNetwork =
    NetworkImpl<Quantized::MainNet<In, Hidden, ValueHidden, PolicyHidden>,
                Activation::clamp>;

#define Q32
#define Q64
#define Q128

namespace Impl {
inline auto invalid(const std::string &msg) -> std::shared_ptr<NetworkBase> {
  throw std::runtime_error{"Invalid layer size for quantized net " + msg +
                           " (check code for valid sizes)."};
}

template <int In, int Hidden, int ValueHidden, int PolicyHidden>
auto visit_network_4(const auto &F, std::shared_ptr<NetworkBase> network) {
  if constexpr (Hidden < ValueHidden) {
    return Impl::invalid("Value hidden cannot be larger than hidden.");
  } else if constexpr (Hidden < PolicyHidden) {
    return Impl::invalid("Policy hidden cannot be larger than hidden.");
  } else {
    using Net = QNetwork<In, Hidden, ValueHidden, PolicyHidden>;
    if (!network) {
      network = std::make_shared<Net>();
    }
    if (auto *net = dynamic_cast<Net *>(network.get())) {
      F(*net);
    } else {
      throw std::runtime_error{"Invalid discrete cast."};
    }
    return network;
  }
}

template <int In, int Hidden, int ValueHidden>
auto visit_network_3(int policy_hidden, const auto &F,
                     std::shared_ptr<NetworkBase> network) {
  switch (policy_hidden) {
#ifdef Q32
  case 32:
    return visit_network_4<In, Hidden, ValueHidden, 32>(F, std::move(network));
#endif
#ifdef Q64
  case 64:
    return visit_network_4<In, Hidden, ValueHidden, 64>(F, std::move(network));
#endif
#ifdef Q128
  case 128:
    return visit_network_4<In, Hidden, ValueHidden, 128>(F, std::move(network));
#endif
  default:
    return Impl::invalid("Policy hidden: " + std::to_string(policy_hidden));
  }
}

template <int In, int Hidden>
auto visit_network_2(int value_hidden, int policy_hidden, const auto &F,
                     std::shared_ptr<NetworkBase> network) {
  switch (value_hidden) {
#ifdef Q32
  case 32:
    return visit_network_3<In, Hidden, 32>(policy_hidden, F,
                                           std::move(network));
#endif
#ifdef Q64
  case 64:
    return visit_network_3<In, Hidden, 64>(policy_hidden, F,
                                           std::move(network));
#endif
#ifdef Q128
  case 128:
    return visit_network_3<In, Hidden, 128>(policy_hidden, F,
                                            std::move(network));
#endif
  default:
    return Impl::invalid("Value hidden: " + std::to_string(value_hidden));
  }
}

template <int In>
auto visit_network_1(int hidden, int value_hidden, int policy_hidden,
                     const auto &F, std::shared_ptr<NetworkBase> network) {
  switch (hidden) {
#ifdef Q32
  case 32:
    return visit_network_2<In, 32>(value_hidden, policy_hidden, F,
                                   std::move(network));
#endif
#ifdef Q64
  case 64:
    return visit_network_2<In, 64>(value_hidden, policy_hidden, F,
                                   std::move(network));
#endif
#ifdef Q128
  case 128:
    return visit_network_2<In, 128>(value_hidden, policy_hidden, F,
                                    std::move(network));
#endif
  default:
    return Impl::invalid("Hidden: " + std::to_string(hidden));
  }
}
} // namespace Impl

inline auto visit_quantized_network(int in, int hidden, int value_hidden,
                                    int policy_hidden, const auto &F,
                                    std::shared_ptr<NetworkBase> network = {}) {
  switch (in) {
  case 768:
    return Impl::visit_network_1<768>(hidden, value_hidden, policy_hidden, F,
                                      std::move(network));
  default:
    return Impl::invalid("Side dim: " + std::to_string(in));
  }
}

void visit_network(std::shared_ptr<NetworkBase> network, const auto &F) {

  if (!network) {
    throw std::runtime_error{"Attempting to use uninitialized network."};
  }

  if (auto net = std::dynamic_pointer_cast<Network>(network)) {
    F(*net);
  } else if (auto net = std::dynamic_pointer_cast<NetworkClamped>(network)) {
    F(*net);
  } else if (network->main_net_float() == nullptr) {
    // quantized
    const auto [i, h, v, p] = network->shape();
    visit_quantized_network(i, h, v, p, F, network);
  } else {
    assert(false);
  }
}

template <typename T, Activation activation, typename... Caches>
T *write_pokemon(T *embedding, uint8_t index, const PKMN::Pokemon &pokemon,
                 uint8_t sleep, NetworkBase &network, Caches &...cache_pack) {
  static_assert(sizeof...(Caches) <= 1,
                "write_pokemon takes zero or one cache");
  using namespace Encode::Battle::Pokemon;
  const auto pokemon_dim = network.pokemon_out_dim();
  if constexpr (sizeof...(Caches) == 1) {
    auto &cache = std::get<0>(std::tie(cache_pack...)).pokemon_cache[index];
    const T *t = cache.get(pokemon, sleep);
    std::copy(t, t + pokemon_dim, embedding);
  } else {
    uint16_t indices[n_nonzero];
    float input[n_nonzero];
    Encode::Battle::Pokemon::write(pokemon, sleep, input, indices);
    network.template propagate_embedding<Embedding_::Pokemon, T, activation>(
        input, indices, embedding, n_nonzero);
  }
  return embedding + pokemon_dim;
}

template <typename T, Activation activation, typename... Caches>
T *write_pokemon_moves(T *embedding, uint8_t index, const auto &moves,
                       NetworkBase &network, Caches &...cache_pack) {
  static_assert(sizeof...(Caches) <= 1,
                "write_active_moves takes zero or one cache");
  using namespace Encode::Battle::Moves;
  const auto moves_dim = network.moves_out_dim();
  if constexpr (sizeof...(Caches) == 1) {
    auto &cache =
        std::get<0>(std::tie(cache_pack...)).pokemon_moves_cache[index];
    const T *t = cache.get(moves);
    std::copy(t, t + moves_dim, embedding);
  } else {
    uint16_t indices[n_nonzero];
    float input[n_nonzero];
    auto n = Encode::Battle::Moves::write(moves, input, indices);
    network.template propagate_embedding<Embedding_::Moves, T, activation>(
        input, indices, embedding, n);
  }
  return embedding + moves_dim;
}

template <typename T, Activation activation, typename... Caches>
T *write_active_moves(T *embedding, uint8_t index, const auto &moves,
                      NetworkBase &network, Caches &...cache_pack) {
  static_assert(sizeof...(Caches) <= 1,
                "write_active_moves takes zero or one cache");
  using namespace Encode::Battle::Moves;
  const auto moves_dim = network.moves_out_dim();
  if constexpr (sizeof...(Caches) == 1) {
    auto &cache =
        std::get<0>(std::tie(cache_pack...)).active_moves_cache[index];
    const T *t = cache.template get<activation>(network, moves);
    std::copy(t, t + moves_dim, embedding);
  } else {
    uint16_t indices[n_nonzero];
    float input[n_nonzero];
    auto n = Encode::Battle::Moves::write(moves, input, indices);
    network.template propagate_embedding<Embedding_::Moves, T, activation>(
        input, indices, embedding, n);
  }
  return embedding + moves_dim;
}

template <typename T, Activation activation, typename... Caches>
T *write_active(T *embedding, uint8_t index, const PKMN::ActivePokemon &active,
                const PKMN::Duration &duration, NetworkBase &network,
                Caches &...cache_pack) {
  static_assert(sizeof...(Caches) <= 1, "write_active takes zero or one cache");
  using namespace Encode::Battle::Active;
  const auto active_dim = network.active_out_dim();

  if constexpr (sizeof...(Caches) == 1) {
    auto &cache = std::get<0>(std::tie(cache_pack...)).active_cache[index];
    const T *t = cache.template get<activation>(network, active, duration);
    std::copy(t, t + active_dim, embedding);
  } else {
    uint16_t indices[n_nonzero];
    float input[n_nonzero];
    const auto n =
        Encode::Battle::Active::write(active, duration, input, indices);
    network.template propagate_embedding<Embedding_::Active, T, activation>(
        input, indices, embedding, n);
  }
  return embedding + active_dim;
}

template <typename T, Activation activation, typename... Caches>
T *write_side_embedding(T *embedding, const PKMN::Side &side,
                        const PKMN::Duration &duration, NetworkBase &network,
                        Caches &...caches) {
  static_assert(sizeof...(Caches) <= 1,
                "write_side_embedding takes zero or one cache");
  const auto pokemon_dim = network.pokemon_out_dim();
  const auto active_dim = network.active_out_dim();
  const auto moves_dim = network.moves_out_dim();
  // const auto side_dim = (active_dim + 6 * (pokemon_dim + moves_dim));
  // const auto *e = embedding;
  const auto write_zero = [&embedding](auto dim) {
    std::fill_n(embedding, dim, T{0});
    return embedding + dim;
  };
  const auto &stored = side.stored();
  if (stored.hp == 0) {
    embedding = write_zero(active_dim + pokemon_dim + moves_dim);
  } else {
    const auto index = side.order[0] - 1;
    embedding = write_active<T, activation>(embedding, index, side.active,
                                            duration, network, caches...);
    embedding = write_pokemon<T, activation>(
        embedding, index, stored, duration.sleep(0), network, caches...);
    if (side.active.moves == stored.moves) {
      embedding = write_pokemon_moves<T, activation>(
          embedding, index, side.active.moves, network, caches...);
    } else {
      embedding = write_active_moves<T, activation>(
          embedding, index, side.active.moves, network, caches...);
    }
  }
  for (auto slot = 2; slot <= 6; ++slot) {
    const auto index = side.order[slot - 1];
    if (index == 0) {
      embedding = write_zero(pokemon_dim + moves_dim);
    } else {
      const auto &pokemon = side.pokemon[index - 1];
      if (pokemon.hp == 0) {
        embedding = write_zero(pokemon_dim + moves_dim);
      } else {
        embedding = write_pokemon<T, activation>(embedding, index - 1, pokemon,
                                                 duration.sleep(slot - 1),
                                                 network, caches...);
        embedding = write_pokemon_moves<T, activation>(
            embedding, index - 1, pokemon.moves, network, caches...);
      }
    }
  }
  return embedding;
}

} // namespace NN::Battle