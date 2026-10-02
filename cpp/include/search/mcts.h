#pragma once

#include <libpkmn/layout.h>
#include <libpkmn/strings.h>
#include <nn/battle/network.h>
#include <search/durations.h>
#include <search/hash.h>
#include <search/poke-engine-evaluate.h>
#include <search/util/softmax.h>
#include <util/random.h>

#include <chrono>
#include <cmath>
#include <span>
#include <iostream>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "../extern/lrsnash/src/lib.h"

namespace MCTS {
template <typename T> void debug_print(const T &x) {
#ifndef NDEBUG
  std::cout << x;
#endif
}

template <typename F, typename... Args> void debug_do(F &&f, Args &&...args) {
#ifndef NDEBUG
  std::forward<F>(f)(std::forward<Args>(args)...);
#endif
}

struct MonteCarlo {
  bool forbid_switches;
  bool forbid_status;
  MonteCarlo(bool forbid_switches = false, bool forbid_status = false)
      : forbid_switches{forbid_switches}, forbid_status{forbid_status} {}
};
} // namespace MCTS

namespace TypeTraits {
template <typename T>
inline constexpr bool is_node =
    requires(std::remove_cvref_t<T> &heap) { heap.stats; };

template <typename T>
inline constexpr bool is_table =
    requires(std::remove_cvref_t<T> &heap) { heap.entries; };

template <typename T>
inline constexpr bool is_network =
    std::is_base_of_v<NN::Battle::NetworkBase, std::remove_cvref_t<T>>;

template <typename T>
inline constexpr bool is_poke_engine =
    std::is_same_v<PokeEngine::Eval, std::remove_cvref_t<T>>;

template <typename T>
inline constexpr bool is_monte_carlo =
    std::is_same_v<MCTS::MonteCarlo, std::remove_cvref_t<T>>;

template <typename T>
inline constexpr bool is_contextual_bandit =
    requires(typename std::remove_cvref_t<T>::Stats &stats,
             const std::remove_cvref_t<T> &bandit,
             const float *logits) { stats.softmax_logits(bandit, logits); };

template <typename T>
inline constexpr bool is_matrix_ucb =
    requires(std::remove_cvref_t<T> &params) { params.bandit; };
template <typename T>
inline constexpr bool is_matrix_ucb_stats =
    requires(std::remove_cvref_t<T> &stats) { stats.matrix; };

// TODO static asserts at end of file
} // namespace TypeTraits

namespace MCTS {
using namespace TypeTraits;

const auto get_bandit_params = [](const auto &params) -> const auto & {
  if constexpr (requires { params.bandit; }) {
    return params.bandit;
  } else {
    return params;
  }
};

struct Input {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
};

struct Output {
  struct Side {
    uint8_t k;
    std::array<pkmn_choice, 9> choices;
    std::array<double, 9> logit;
    std::array<double, 9> prior;
    std::array<double, 9> empirical;
    std::array<double, 9> nash;
    std::array<double, 9> total;
  };

  std::array<std::array<size_t, 9>, 9> visit_matrix;
  std::array<std::array<double, 9>, 9> value_matrix;
  // TODO M2
  std::array<std::array<double, 9>, 9> variance_matrix;

  size_t iterations;
  std::chrono::microseconds duration;
  size_t total_depth;

  double initial_value;
  double empirical_value;
  double nash_value;
  Side p1;
  Side p2;
};

// for std::map compatibility
using Obs = std::array<uint8_t, 16>;

template <typename strategy_type = float> struct MatrixUCBStats {
  struct Side {
    uint8_t k;
    std::array<strategy_type, 9> policy;
    uint8_t sample(auto &device) const noexcept {
      if constexpr (std::is_integral_v<strategy_type>) {
        auto n = device.random_int(
            static_cast<size_t>(std::numeric_limits<strategy_type>::max()) + 1);
        for (auto i = 0; i < policy.size(); ++i) {
          if (policy[i] >= n) {
            return i;
          }
          n -= policy[i];
        }
        return 0;
      } else {
        return device.sample_pdf(std::span<const strategy_type>{policy.data(), k});
      }
    }
  };
  struct Entry {
    size_t visits;
    double total_value;
  };

  Side p1;
  Side p2;
  std::array<std::array<Entry, 9>, 9> matrix;
  float ucb_weight;
  size_t total_visits;

  void init(const auto m, const auto n) noexcept {
    ucb_weight = std::log(2 * m * n);
    total_visits = 0;
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        matrix[i][j] = {};
      }
    }
    constexpr strategy_type uniform =
        std::is_integral_v<strategy_type>
            ? std::numeric_limits<strategy_type>::max()
            : 1;
    p1.k = m;
    p2.k = n;
    std::fill_n(p1.policy.begin(), m, uniform / m);
    std::fill_n(p2.policy.begin(), n, uniform / n);
  }

  bool is_init() const { return p1.k; }

  void softmax_logits(const auto &params, const float *p1_logits,
                      const float *p2_logits) {
    softmax_k(p1.policy.data(), p1_logits, p1.k);
    softmax_k(p2.policy.data(), p2_logits, p2.k);
  }

  void solve_ucb_matrix(auto &device, auto &params) noexcept {
    std::array<int, 9 * 9> p1_ucb_matrix;
    std::array<int, 9 * 9> p2_ucb_matrix;
    const float log_T = std::log(total_visits + 1);
    const float exp = params.c * std::sqrt(2 * (2 * log_T + ucb_weight));
    for (auto i = 0; i < p1.k; ++i) {
      for (auto j = 0; j < p2.k; ++j) {
        const auto &entry = matrix[i][j];
        float p1_entry = 0.5;
        float p2_entry = 0.5;
        uint32_t visits = 1;
        if (entry.visits > 0) {
          p1_entry = entry.total_value / entry.visits;
          p2_entry = 1 - p1_entry;
          visits = entry.visits;
        }
        const float inv_exploration_den = 1.0 / std::sqrt(visits);
        const float exploration = exp * inv_exploration_den;
        p1_entry += exploration;
        p2_entry += exploration;
        p1_ucb_matrix[i * p2.k + j] = p1_entry * params.discretize_factor;
        p2_ucb_matrix[i * p2.k + j] = p2_entry * params.discretize_factor;
      }
    }
    std::array<float, 9 + 2> p1_nash;
    std::array<float, 9 + 2> p2_nash;
    LRSNash::Input solve_input{static_cast<int>(p1.k), static_cast<int>(p2.k),
                               static_cast<int>(params.discretize_factor),
                               p1_ucb_matrix.data(), p2_ucb_matrix.data()};
    LRSNash::FloatOneSumOutput solve_output{p1_nash.data(), p2_nash.data(), 0};
    LRSNash::solve_full(&solve_input, &solve_output);

    // policy = w * old + (1 - w) * nash; w = 0 is a plain overwrite
    const float w = params.old_weight;
    constexpr float scale =
        std::is_integral_v<strategy_type>
            ? static_cast<float>(std::numeric_limits<strategy_type>::max())
            : 1.0f;
    for (auto i = 0; i < p1.k; ++i) {
      p1.policy[i] = w * p1.policy[i] + (1 - w) * scale * p1_nash[i];
    }
    for (auto j = 0; j < p2.k; ++j) {
      p2.policy[j] = w * p2.policy[j] + (1 - w) * scale * p2_nash[j];
    }
  }

  void select(auto &device, const auto &params, auto &outcome) {
    if ((total_visits + 1) % params.interval == 0) {
      solve_ucb_matrix(device, params);
    }
    outcome.first.index = p1.sample(device);
    outcome.second.index = p2.sample(device);
    // do this here so grow=0 will never proc.
    ++total_visits;
  }

  void update(const auto &params, const auto &outcome) {
    auto &entry = matrix[outcome.first.index][outcome.second.index];
    ++entry.visits;
    entry.total_value += outcome.first.value;
  }
};

template <typename Bandit> struct Node {
  using Key = std::tuple<uint8_t, uint8_t, Obs>;
  using JointStats = typename Bandit::JointStats;
  JointStats stats;
  std::map<Key, Node<Bandit>> children;
};

template <typename Bandit> struct MatrixUCBNode {
  using Key = std::tuple<uint8_t, uint8_t, Obs>;
  using Both = std::variant<Node<Bandit>, MatrixUCBNode<Bandit>>;
  MatrixUCBStats<> stats;
  std::map<Key, Both> children;
};

template <typename Bandit> struct Table {
  using Key = uint64_t;
  using JointStats = typename Bandit::JointStats;
  Hash::Battle hasher;
  std::unordered_map<Key, JointStats> entries;
};

template <typename Bandit> struct MatrixUCBParams {
  Bandit bandit;
  float c;
  uint32_t interval;
  uint32_t grow;
  int discretize_factor = 256;
  float old_weight = 0;
};

struct SearchOptions {
  size_t root_rolls;
  size_t other_rolls;
  bool debug_print;
  // dependent
  bool rolls_same;
  bool clamping;

  constexpr SearchOptions(size_t root_rolls = 39, size_t other_rolls = 39,
                          bool debug_print = false)
      : root_rolls{root_rolls}, other_rolls{other_rolls},
        debug_print{debug_print}, rolls_same{root_rolls == other_rolls},
        clamping{(root_rolls != 39) || (other_rolls != 39)} {}
};

constexpr SearchOptions default_search{3, 1};

// runtime
struct RuntimeOptions {
  uint32_t max_depth;
  uint32_t rollout_depth;
  float rollout_temp;
  RuntimeOptions(uint32_t max_depth = 0, uint32_t rollout_depth = 0,
                 float rollout_temp = 1)
      : max_depth{max_depth}, rollout_depth{rollout_depth},
        rollout_temp{rollout_temp} {}
};

template <SearchOptions Options = default_search> struct Search {

  pkmn_gen1_battle_options options;
  pkmn_gen1_chance_options chance_options;
  pkmn_gen1_calc_options calc_options;
  std::array<pkmn_choice, 9> p1_choices;
  std::array<pkmn_choice, 9> p2_choices;
  // network
  std::vector<float> battle_embedding;
  std::vector<uint8_t> battle_embedding_quantized;
  float p1_logits[9];
  float p2_logits[9];
  uint16_t p1_choice_index[9];
  uint16_t p2_choice_index[9];

  template <typename... Caches>
  Output run(auto &device, const auto budget, const auto &params, auto &heap,
             auto &eval, const Input &input, Output output,
             const RuntimeOptions &live_options, Caches &...caches) noexcept {

    // reset data members
    *this = {};

    constexpr bool is_node_ = is_node<decltype(heap)>;
    constexpr bool is_contextual_bandit_ =
        is_contextual_bandit<decltype(params)>;
    constexpr bool is_matrix_ucb_ = is_matrix_ucb<decltype(params)>;

    if constexpr (is_poke_engine<decltype(eval)>) {
      eval.get_root_score(input.battle);
    } else if constexpr (is_network<decltype(eval)>) {
      battle_embedding.resize(2 * eval.side_embedding_dim(true));
      battle_embedding_quantized.resize(2 * eval.side_embedding_dim(true));
    }

    if (!heap.stats.is_init()) {
      auto [m, n] = get_choices(input.battle, input.result);
      if (!heap.stats.is_init()) {
        heap.stats.init(m, n);
      }
      if constexpr (is_network<decltype(eval)>) {
        constexpr bool policy_inference =
            is_matrix_ucb_ || is_contextual_bandit_;
        chance_options.durations = input.durations;
        pkmn_gen1_battle_options_set(&options, nullptr, &chance_options,
                                     nullptr);
        write_battle_embedding(eval, input.battle, caches...);
        output.initial_value = network_inference<true, policy_inference, true>(
            eval, input.battle, m, n, caches...);
        if constexpr (policy_inference) {
          heap.stats.softmax_logits(params, p1_logits, p2_logits);
        }
      }
    }

    const auto start = std::chrono::high_resolution_clock::now();
    // time duration
    if constexpr (requires {
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        budget);
                  }) {
      const auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(budget);
      std::chrono::microseconds elapsed{};
      while (elapsed < duration) {
        run_root_iteration(device, params, heap, input, eval, output,
                           live_options, caches...);
        ++output.iterations;
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);
      }
      // run while boolean flag is set
    } else if constexpr (requires { *budget; }) {
      while (*budget) {
        run_root_iteration(device, params, heap, input, eval, output,
                           live_options, caches...);
        ++output.iterations;
      }
      // number of iterations
    } else {
      for (auto i = 0; i < budget; ++i) {
        run_root_iteration(device, params, heap, input, eval, output,
                           live_options, caches...);
        ++output.iterations;
      }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    output.duration +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    process_output(input, heap, output);
    return output;
  }

  template <typename... Caches>
  void run_root_iteration(auto &device, const auto &params, auto &heap,
                          const auto &input, auto &eval, Output &output,
                          const RuntimeOptions &live_options,
                          Caches &...caches) noexcept {
    auto copy = input;
    auto *rng = reinterpret_cast<uint64_t *>(
        copy.battle.bytes + PKMN::Layout::Offsets::Battle::rng);
    rng[0] = device.uniform_64();
    chance_options.durations = copy.durations;
    randomize_hidden_variables(copy.battle, copy.durations);
    pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);
    run_iteration(device, params, heap, copy, eval, output, live_options, 0,
                  caches...);
  }

  // typical recursive mcts function
  // we return value for each player because it's slightly faster than calcing
  // 1 - value for each bandit update
  template <typename... Caches>
  std::pair<float, float>
  run_iteration(auto &device, const auto &params, auto &heap, auto &input,
                auto &eval, Output &output, const RuntimeOptions &live_options,
                size_t depth, Caches &...caches) noexcept {
    constexpr bool is_matrix_ucb_ = is_matrix_ucb<decltype(params)>;
    const auto &bandit = [&params]() {
      if constexpr (is_matrix_ucb_) {
        return params.bandit;
      } else {
        return params;
      }
    }();
    using Bandit = std::remove_cvref_t<decltype(bandit)>;
    constexpr bool is_matrix_ucb_stats_ =
        is_matrix_ucb_stats<decltype(heap.stats)>;
    static_assert(is_matrix_ucb_ == is_matrix_ucb_stats_);
    constexpr bool is_contextual_bandit_ = is_contextual_bandit<Bandit>;
    constexpr bool policy_inference = is_matrix_ucb_ || is_contextual_bandit_;

    auto &battle = input.battle;
    auto &result = input.result;
    auto &stats = heap.stats;

    const bool terminal =
        live_options.max_depth > 0 && depth >= live_options.max_depth;

    if (stats.is_init() && !terminal) {
      using Outcome = typename Bandit::Outcome;
      using JointOutcome = std::pair<Outcome, Outcome>;

      JointOutcome outcome;
      stats.select(device, params, outcome);

      // grow MatrixUCB subtree
      if constexpr (is_matrix_ucb_) {
        const auto &entry =
            stats.matrix[outcome.first.index][outcome.second.index];
        if (entry.visits == params.grow) {
          debug_print("GROW " + std::to_string(depth) + "\n");
          for (auto &[key, both] : heap.children) {
            if (std::get<0>(key) == outcome.first.index &&
                std::get<1>(key) == outcome.second.index) {
              if (std::holds_alternative<Node<Bandit>>(both)) {
                auto &node = std::get<Node<Bandit>>(both);
                if (node.stats.is_init()) {
                  // node is not init iff state is terminal
                  MatrixUCBNode<Bandit> new_node{};
                  new_node.stats.init(node.stats.p1.k, node.stats.p2.k);
                  // TODO use bandit stats to warm start side policies
                  for (auto &[key_, value_] : node.children) {
                    new_node.children[key_] = std::move(value_);
                  }
                  both = std::move(new_node);
                }
              }
            }
          }
        }
      }

      const auto [m_, n_] = get_choices(battle, result);
      const auto c1 = p1_choices[outcome.first.index];
      const auto c2 = p2_choices[outcome.second.index];

      battle_options_set(battle, depth);
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);

      const auto &obs = *reinterpret_cast<const Obs *>(
          pkmn_gen1_battle_options_chance_actions(&options));
      auto &child =
          heap.children[{outcome.first.index, outcome.second.index, obs}];
      const auto values = [&]() {
        if constexpr (is_matrix_ucb_) {
          using node_t = Node<Bandit>;
          using matrix_ucb_node_t = MatrixUCBNode<Bandit>;
          if (std::holds_alternative<node_t>(child)) {
            return run_iteration(device, bandit, std::get<node_t>(child), input,
                                 eval, output, live_options, depth + 1,
                                 caches...);
          } else {
            return run_iteration(
                device, params, std::get<matrix_ucb_node_t>(child), input, eval,
                output, live_options, depth + 1, caches...);
          }
        } else {
          return run_iteration(device, bandit, child, input, eval, output,
                               live_options, depth + 1, caches...);
        }
      }();
      outcome.first.value = values.first;
      outcome.second.value = values.second;
      stats.update(params, outcome);

      if constexpr (!is_matrix_ucb_) {
        if (depth == 0) {
          ++output.visit_matrix[outcome.first.index][outcome.second.index];
          output.value_matrix[outcome.first.index][outcome.second.index] +=
              values.first;
          if constexpr (requires { outcome.first.policy; }) {
            for (auto i = 0; i < output.p1.k; ++i) {
              output.p1.total[i] += outcome.first.policy[i];
            }
            for (auto i = 0; i < output.p2.k; ++i) {
              output.p2.total[i] += outcome.second.policy[i];
            }
          } else {
            // static_assert(!std::is_same_v<Bandit, RegretMatching>);
          }
        }
      }

      return values;
    }

    output.total_depth += depth;

    auto result_type = pkmn_result_type(result);
    if (!result_type) {
      using Eval = decltype(eval);
      float leaf_value;
      if constexpr (is_monte_carlo<Eval>) {
        leaf_value = init_stats_and_rollout(stats, device, battle, result);
      } else {
        auto [m, n] = get_choices(battle, result);
        stats.init(m, n);
        pkmn_choice c1, c2;

        if constexpr (is_network<Eval>) {
          const auto neg_inf = -std::numeric_limits<float>::infinity();
          std::array<float, 9> p1_rollout;
          std::array<float, 9> p2_rollout;
          write_battle_embedding(eval, battle, caches...);
          if constexpr (policy_inference) {
            network_inference<false, true, true>(eval, battle, m, n, caches...);
            stats.softmax_logits(params, p1_logits, p2_logits);
          }
          for (auto rollout = 0; rollout < live_options.rollout_depth;
               ++rollout) {
            if (!policy_inference || rollout) {
              network_inference<false, true, true>(eval, battle, m, n,
                                                   caches...);
            }
            std::fill(p1_logits + m, p1_logits + 9, neg_inf);
            std::fill(p2_logits + n, p2_logits + 9, neg_inf);
            softmax_9_temp(p1_rollout, p1_logits, live_options.rollout_temp);
            softmax_9_temp(p2_rollout, p2_logits, live_options.rollout_temp);
            c1 = p1_choices[device.sample_pdf(p1_rollout)];
            c2 = p2_choices[device.sample_pdf(p2_rollout)];
            pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
            result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
            if (result_type = pkmn_result_type(result)) {
              return terminal_values(result_type);
            }
            update_battle_embedding(eval, battle, c1, c2, caches...);
            std::tie(m, n) = get_choices(battle, result);
          }

          // we can skip the main trunk if we didn't update
          if constexpr (policy_inference) {
            leaf_value = live_options.rollout_depth
                             ? network_inference<true, false, true>(
                                   eval, battle, m, n, caches...)
                             : network_inference<true, false, false>(
                                   eval, battle, m, n, caches...);
          } else {
            leaf_value = network_inference<true, false, true>(eval, battle, m,
                                                              n, caches...);
          }
          auto embedding_copy = embedding_clone(eval);
          assert(
              std::equal(embedding_copy.data(),
                         embedding_copy.data() + eval.side_embedding_dim(true),
                         embedding(eval)));
        } else if constexpr (is_poke_engine<Eval>) {
          leaf_value = eval.evaluate(battle);
        } else {
          static_assert(!std::is_same_v<Eval, Eval>, "Invalid eval type!");
        }
      }
      return {leaf_value, 1 - leaf_value};
    } else {
      return terminal_values(result_type);
    }
  }

  std::pair<float, float> terminal_values(auto result_type) {
    switch (result_type) {
    case PKMN_RESULT_WIN: {
      return {1, 0};
    }
    case PKMN_RESULT_LOSE: {
      return {0, 1};
    }
    case PKMN_RESULT_TIE: {
      return {.5, .5};
    }
    default: {
      assert(false);
      return {.5, .5};
    }
    }
  }

  void process_output(const Input &input, const auto &heap,
                      Output &output) noexcept {
    std::tie(output.p1.k, output.p2.k) =
        get_choices(input.battle, input.result);
    std::copy_n(p1_choices.begin(), output.p1.k, output.p1.choices.begin());
    std::copy_n(p2_choices.begin(), output.p2.k, output.p2.choices.begin());

    if constexpr (is_matrix_ucb_stats<decltype(heap.stats)>) {
      for (auto i = 0; i < output.p1.k; ++i) {
        for (auto j = 0; j < output.p2.k; ++j) {
          const auto &entry = heap.stats.matrix[i][j];
          output.visit_matrix[i][j] = entry.visits;
          output.value_matrix[i][j] = entry.total_value;
        }
      }
    }

    constexpr int discretize_factor = 256;
    double total_value = 0;
    output.p1.empirical = {};
    output.p2.empirical = {};
    std::array<int, 9 * 9> solve_matrix;
    for (int i = 0; i < output.p1.k; ++i) {
      for (int j = 0; j < output.p2.k; ++j) {
        total_value += output.value_matrix[i][j];
        auto n = output.visit_matrix[i][j];
        output.p1.empirical[i] += n;
        output.p2.empirical[j] += n;
        double e = 0.5;
        if (n) {
          e = output.value_matrix[i][j] / n;
        }
        solve_matrix[output.p2.k * i + j] = e * discretize_factor;
      }
    }

    output.empirical_value = total_value / output.iterations;
    LRSNash::FastInput solve_input{static_cast<int>(output.p1.k),
                                   static_cast<int>(output.p2.k),
                                   solve_matrix.data(), discretize_factor};
    std::array<float, 9 + 2> nash1{}, nash2{};
    LRSNash::FloatOneSumOutput solve_output{nash1.data(), nash2.data(), 0};
    LRSNash::solve_fast(&solve_input, &solve_output);

    for (auto i = 0; i < output.p1.k; ++i) {
      output.p1.empirical[i] /= (double)output.iterations;
      output.p1.nash[i] = nash1[i];
    }
    for (auto j = 0; j < output.p2.k; ++j) {
      output.p2.empirical[j] /= (double)output.iterations;
      output.p2.nash[j] = nash2[j];
    }
    output.nash_value = solve_output.value;
  }

  float init_stats_and_rollout(auto &stats, auto &device,
                               pkmn_gen1_battle &battle,
                               pkmn_result result) noexcept {
    auto seed = device.uniform_64();
    auto [m, n] = get_choices(battle, result);
    auto c1 = p1_choices[seed % m];
    seed >>= 32;
    auto c2 = p2_choices[seed % n];
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    if (!stats.is_init()) {
      stats.init(m, n);
    }
    while (!pkmn_result_type(result)) {
      seed = device.uniform_64();
      std::tie(m, n) = get_choices(battle, result);
      c1 = p1_choices[seed % m];
      seed >>= 32;
      c2 = p2_choices[seed % n];
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    }
    return terminal_values(pkmn_result_type(result)).first;
  }

  auto get_choices(const pkmn_gen1_battle &battle,
                   pkmn_result result) noexcept {
    const auto m = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), p1_choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    const auto n = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), p2_choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    assert(m > 0 && m <= 9);
    assert(n > 0 && n <= 9);
    return std::pair<uint8_t, uint8_t>{m, n};
  }

  void battle_options_set(const pkmn_gen1_battle &battle, size_t depth) {
    if constexpr (!Options.clamping) {
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    } else {
      // last two bytes of battle rng
      const auto *rand = battle.bytes + PKMN::Layout::Offsets::Battle::rng + 6;
      auto *over = this->calc_options.overrides.bytes;
      if constexpr (Options.rolls_same) {
        over[0] = roll_byte<Options.root_rolls>(rand[0]);
        over[8] = roll_byte<Options.root_rolls>(rand[1]);
      } else {
        if (depth == 0) {
          over[0] = roll_byte<Options.root_rolls>(rand[0]);
          over[8] = roll_byte<Options.root_rolls>(rand[1]);
        } else {
          over[0] = roll_byte<Options.other_rolls>(rand[0]);
          over[8] = roll_byte<Options.other_rolls>(rand[1]);
        }
      }
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, &calc_options);
    }
  }

  template <size_t n_rolls>
  inline static constexpr uint8_t roll_byte(const uint8_t seed) noexcept {
    constexpr uint8_t lowest_roll{217};
    constexpr uint8_t middle_roll{236};
    if constexpr (n_rolls == 1) {
      return middle_roll;
    } else {
      static_assert((n_rolls == 2) || (n_rolls == 3) || (n_rolls == 20));
      constexpr uint8_t step = 38 / (n_rolls - 1);
      return lowest_roll + step * (seed % n_rolls);
    }
  }

  const auto &durations() const noexcept {
    return *pkmn_gen1_battle_options_chance_durations(&options);
  }

  // Network helpers

  auto embedding(auto &eval) noexcept {
    using Eval = std::remove_cvref_t<decltype(eval)>;
    using T = typename Eval::T;
    if constexpr (std::is_same_v<T, uint8_t>) {
      return battle_embedding_quantized.data();
    } else if constexpr (std::is_same_v<T, float>) {
      return battle_embedding.data();
    } else {
      static_assert(!std::is_same_v<T, T>);
    }
  }

  auto embedding_clone(auto &eval) const noexcept {
    using Eval = std::remove_cvref_t<decltype(eval)>;
    using T = typename Eval::T;
    if constexpr (std::is_same_v<T, uint8_t>) {
      return battle_embedding_quantized;
    } else if constexpr (std::is_same_v<T, float>) {
      return battle_embedding;
    } else {
      static_assert(!std::is_same_v<T, T>);
    }
  }

  template <typename Eval, typename... Caches>
  void write_battle_embedding(Eval &eval, const auto &b, Caches &...caches) {
    using T = typename Eval::T;
    constexpr auto activation = Eval::act;
    constexpr bool rewrite_hp = Eval::rewrite_hp;
    auto *e = embedding(eval);
    const auto &battle = PKMN::view(b);
    const auto &d = PKMN::view(durations());
    if constexpr (sizeof...(Caches) == 2) {
      auto &p1_cache = std::get<0>(std::tie(caches...));
      auto &p2_cache = std::get<1>(std::tie(caches...));
      e = NN::Battle::write_side_embedding<T, activation, rewrite_hp>(
          e, battle.sides[0], d.get(0), eval, p1_cache);
      e = NN::Battle::write_side_embedding<T, activation, rewrite_hp>(
          e, battle.sides[1], d.get(1), eval, p2_cache);
    } else {
      e = NN::Battle::write_side_embedding<T, activation, rewrite_hp>(
          e, battle.sides[0], d.get(0), eval);
      e = NN::Battle::write_side_embedding<T, activation, rewrite_hp>(
          e, battle.sides[1], d.get(1), eval);
    }
    assert(std::distance(embedding(eval), e) ==
           2 * eval.side_embedding_dim(rewrite_hp));
  }

  template <typename Eval, typename... Caches>
  void update_battle_embedding(Eval &eval, const auto &b, pkmn_choice p1_choice,
                               pkmn_choice p2_choice, Caches &...caches) {
    using T = typename Eval::T;
    constexpr auto activation = Eval::act;
    constexpr bool rewrite_hp = Eval::rewrite_hp;
    const auto &battle = PKMN::view(b);
    const auto &d = PKMN::view(durations());
    if constexpr (sizeof...(Caches) == 2) {
      auto &p1_cache = std::get<0>(std::tie(caches...));
      auto &p2_cache = std::get<1>(std::tie(caches...));
      NN::Battle::rewrite_side_embedding<T, activation, rewrite_hp>(
          embedding(eval), battle.sides[0], d.get(0), p1_choice, eval,
          p1_cache);
      NN::Battle::rewrite_side_embedding<T, activation, rewrite_hp>(
          embedding(eval) + eval.side_embedding_dim(rewrite_hp),
          battle.sides[1], d.get(1), p2_choice, eval, p2_cache);
    } else {
      NN::Battle::rewrite_side_embedding<T, activation, rewrite_hp>(
          embedding(eval), battle.sides[0], d.get(0), p1_choice, eval);
      NN::Battle::rewrite_side_embedding<T, activation, rewrite_hp>(
          embedding(eval) + eval.side_embedding_dim(rewrite_hp),
          battle.sides[1], d.get(1), p2_choice, eval);
    }
  }

  template <bool use_value, bool use_policy, bool use_trunk, typename... Caches>
  auto network_inference(auto &eval, const auto &b, auto m, auto n,
                         Caches &...caches) {
    static_assert(use_value || use_policy);
    using Eval = std::remove_cvref_t<decltype(eval)>;
    constexpr auto activation = Eval::act;
    const auto &battle = PKMN::view(b);
    if constexpr (use_policy) {
      for (auto i = 0; i < m; ++i) {
        p1_choice_index[i] =
            Encode::Battle::Policy::get_index(battle.sides[0], p1_choices[i]);
      }
      for (auto i = 0; i < n; ++i) {
        p2_choice_index[i] =
            Encode::Battle::Policy::get_index(battle.sides[1], p2_choices[i]);
      }
      if constexpr (use_value) {
        return NN::Battle::sigmoid(
            eval.main_net.template propagate<true, activation, use_trunk>(
                embedding(eval), m, n, p1_choice_index, p2_choice_index,
                p1_logits, p2_logits));
      } else {
        return eval.main_net.template propagate<false, activation, use_trunk>(
            embedding(eval), m, n, p1_choice_index, p2_choice_index, p1_logits,
            p2_logits);
      }
    } else {
      return NN::Battle::sigmoid(
          eval.main_net.template propagate<activation, use_trunk>(
              embedding(eval)));
    }
  }
};

} // namespace MCTS