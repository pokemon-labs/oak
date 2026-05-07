#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/policy.h>
#include <nn/battle/cache.h>
#include <nn/battle/main-net.h>
#include <nn/battle/quantized/main-net.h>
#include <nn/default-hyperparameters.h>
#include <nn/ffn.h>
#include <util/random.h>

namespace NN::Battle {

inline constexpr float sigmoid(const float x) { return 1 / (1 + std::exp(-x)); }

struct NetworkBase {
  virtual std::tuple<int, int, int, int> shape() const noexcept = 0;
  virtual std::unique_ptr<NetworkBase> clone() const noexcept = 0;
  virtual ~NetworkBase() = default;
};

template <typename Main, Activation activation>
class NetworkImpl : public NetworkBase {
public:
  static_assert(activation == Activation::relu ||
                activation == Activation::clamp ||
                activation == Activation::relu_scaled);
  using T = typename Main::T;

  EmbeddingNet pokemon_net;
  EmbeddingNet active_net;
  BattleCache<T> battle_cache;
  Main main_net;

  uint32_t pokemon_out_dim;
  uint32_t active_out_dim;
  uint32_t side_embedding_dim;
  std::vector<T> battle_embedding;

public:
  std::tuple<int, int, int, int> shape() const noexcept {
    return main_net.shape();
  }

  std::unique_ptr<NetworkBase> clone() const noexcept {
    return std::make_unique<NetworkImpl>(*this);
  }

  void fill_cache(const pkmn_gen1_battle &battle) noexcept {
    battle_cache.template fill<Activation::relu>(pokemon_net,
                                                 PKMN::view(battle));
  }

  bool read_parameters(std::istream &stream) {
    const bool ok = pokemon_net.read_parameters(stream) &&
                    active_net.read_parameters(stream) &&
                    main_net.read_parameters(stream);
    if (!ok) {
      return false;
    }
    char dummy;
    if (stream.read(&dummy, 1)) {
      return false;
    } else {
      pokemon_out_dim = pokemon_net.layer<1>().out_dim;
      active_out_dim = active_net.layer<1>().out_dim;
      side_embedding_dim = (1 + active_out_dim) + 5 * (1 + pokemon_out_dim);
      battle_embedding.resize(2 * side_embedding_dim);
      battle_cache = BattleCache<T>{pokemon_out_dim, active_out_dim};
      return true;
    }
  }

  float value_inference(const pkmn_gen1_battle &b,
                        const pkmn_gen1_chance_durations &d) {
    write_battle_embedding(b, d);
    const auto value = sigmoid(
        main_net.template propagate<activation>(battle_embedding.data()));
    assert(!std::isnan(value));
    return value;
  }

  void policy_inference(const pkmn_gen1_battle &b,
                        const pkmn_gen1_chance_durations &d, const auto m,
                        const auto n, const auto *p1_choice,
                        const auto *p2_choice, float *p1, float *p2) {
    static thread_local uint16_t p1_choice_index[9];
    static thread_local uint16_t p2_choice_index[9];
    const auto &battle = PKMN::view(b);
    for (auto i = 0; i < m; ++i) {
      p1_choice_index[i] =
          Encode::Battle::Policy::get_index(battle.sides[0], p1_choice[i]);
    }
    for (auto i = 0; i < n; ++i) {
      p2_choice_index[i] =
          Encode::Battle::Policy::get_index(battle.sides[1], p2_choice[i]);
    }
    write_battle_embedding(b, d);
    main_net.template propagate<false, activation>(battle_embedding.data(), m,
                                                   n, p1_choice_index,
                                                   p2_choice_index, p1, p2);
  }

  auto value_policy_inference(const pkmn_gen1_battle &b,
                              const pkmn_gen1_chance_durations &d, const auto m,
                              const auto n, const auto *p1_choice,
                              const auto *p2_choice, float *p1, float *p2) {
    static thread_local uint16_t p1_choice_index[9];
    static thread_local uint16_t p2_choice_index[9];
    const auto &battle = PKMN::view(b);
    for (auto i = 0; i < m; ++i) {
      p1_choice_index[i] =
          Encode::Battle::Policy::get_index(battle.sides[0], p1_choice[i]);
    }
    for (auto i = 0; i < n; ++i) {
      p2_choice_index[i] =
          Encode::Battle::Policy::get_index(battle.sides[1], p2_choice[i]);
    }
    write_battle_embedding(b, d);
    const auto value = sigmoid(main_net.template propagate<true, activation>(
        battle_embedding.data(), m, n, p1_choice_index, p2_choice_index, p1,
        p2));
    assert(!std::isnan(value));
    return value;
  }

private:
  auto side_embedding_index(auto i) const noexcept {
    assert(i > 0);
    return (1 + active_out_dim) + (i - 1) * (1 + pokemon_out_dim);
  }

  void write_battle_embedding(const pkmn_gen1_battle &b,
                              const pkmn_gen1_chance_durations &d) noexcept {
    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    for (auto s = 0; s < 2; ++s) {
      const auto &side = battle.sides[s];
      const auto &duration = durations.get(s);
      const auto &stored = side.stored();

      auto *side_embedding =
          battle_embedding.data() + s * side_embedding_index(6);

      if (stored.hp == 0) {
        std::fill_n(side_embedding, active_out_dim + 1, 0);
      } else {
        const auto percent = (float)stored.hp / stored.stats.hp;
        side_embedding[0] = std::is_integral_v<T> ? percent * 127 : percent;
        const T *embedding = battle_cache.active[s][side.order[0] - 1]
                                 .template get<Activation::relu>(
                                     active_net, side.active, stored, duration);
        std::copy_n(embedding, active_out_dim, side_embedding + 1);
      }

      for (auto slot = 2; slot <= 6; ++slot) {
        auto *slot_embedding = side_embedding + side_embedding_index(slot - 1);
        // const auto id = side.order[permute[slot - 2]];
        const auto id = side.order[slot - 1];
        if (id == 0) {
          std::fill_n(slot_embedding, pokemon_out_dim + 1, 0);
        } else {
          const auto &pokemon = side.pokemon[id - 1];
          if (pokemon.hp == 0) {
            std::fill_n(slot_embedding, pokemon_out_dim + 1, 0);
          } else {
            const auto percent = (float)pokemon.hp / pokemon.stats.hp;
            slot_embedding[0] = std::is_integral_v<T> ? percent * 127 : percent;
            const auto sleep = duration.sleep(slot - 1);
            const T *embedding =
                battle_cache.pokemon[s][id - 1].get(pokemon, sleep);
            std::copy_n(embedding, pokemon_out_dim, slot_embedding + 1);
          }
        }
      }
    }
  }
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
inline auto invalid(const std::string &msg) -> std::unique_ptr<NetworkBase> {
  throw std::runtime_error{"Invalid layer size for quantized net " + msg +
                           " (check code for valid sizes)."};
}

template <int In, int Hidden, int ValueHidden, int PolicyHidden>
auto visit_network_4(const auto &F, std::unique_ptr<NetworkBase> network) {
  if constexpr (Hidden < ValueHidden) {
    return Impl::invalid("Value hidden cannot be larger than hidden.");
  } else if constexpr (Hidden < PolicyHidden) {
    return Impl::invalid("Policy hidden cannot be larger than hidden.");
  } else {
    using Net = QNetwork<In, Hidden, ValueHidden, PolicyHidden>;
    if (!network) {
      network = std::make_unique<Net>();
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
                     std::unique_ptr<NetworkBase> network) {
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
                     std::unique_ptr<NetworkBase> network) {
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
                     const auto &F, std::unique_ptr<NetworkBase> network) {
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
                                    std::unique_ptr<NetworkBase> network = {}) {
  switch (in) {
  case 768:
    return Impl::visit_network_1<768>(hidden, value_hidden, policy_hidden, F,
                                      std::move(network));
  default:
    return Impl::invalid("Side dim: " + std::to_string(in));
  }
}
} // namespace NN::Battle