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
#include <iostream>
#include <random>
#include <type_traits>
#include <unordered_map>

#include "../extern/lrsnash/src/lib.h"

namespace MCTS {
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
    std::array<double, 9> beta;
  };

  std::array<std::array<size_t, 9>, 9> visit_matrix;
  std::array<std::array<double, 9>, 9> value_matrix;

  size_t iterations;
  std::chrono::microseconds duration;

  double initial_value;
  double empirical_value;
  double nash_value;
  Side p1;
  Side p2;
};

// for std::map compatibility
using Obs = std::array<uint8_t, 16>;

// template <typename JointBandit, typename strategy_type = uint16_t>
// struct MatrixUCBNode {
//   struct Side {
//     uint8_t k;
//     std::array<strategy_type, 9> policy;
//     void init(const auto k) { this->k = k; }
//   };
//   using Key = std::tuple<uint8_t, uint8_t, Obs>;
//   using Node = std::variant<Node<JointBandit>, MatrixUCBNode<JointBandit>>;

//   std::map<Key, Node> children;
//   struct Stats {
//     Side p1;
//     Side p2;
//   };
//   Stats stats;

//   void init(const auto m,children const auto n) noexcept {
//     p1.init(m);
//     p2.init(n);
//   }

//   bool is_init() const { return p1.k; }

//   void select(auto &device, const Params &params,
//               JointOutcome &outcome) const noexcept {
//     p1.select(device, params, outcome.first);
//     p2.select(device, params, outcome.second);
//   }

//   auto sample(auto &device, const auto &s) {
//     if constexpr (std::is_integral_v<strategy_type>) {
//       auto n = device.random_int(
//           static_cast<size_t>(std::numeric_limits<strategy_type>::max()) +
//           1);
//       for (auto i = 0; i < s.size(); ++i) {
//         if (s[i] >= n) {
//           return i;
//         }
//         n -= s[i];
//       }
//       return 0;
//     } else {
//       return device.sample_pdf(s);
//     }
//   }

//   struct Entry {
//     uint64_t visits;
//     double total_value;
//   };
//   std::array<std::array<Entry, 9>, 9> matrix;

//   void update(const auto &outcome) {
//     auto entry = matrix[outcome.first.index][outcome.second.index];
//     ++entry.visits;
//     total_value += outcome.value;
//   }
// };

template <typename Bandit> struct Node {
  using Key = std::tuple<uint8_t, uint8_t, Obs>;
  using JointStats = typename Bandit::JointStats;
  JointStats stats;
  std::map<Key, Node<Bandit>> children;
};

template <typename Bandit> struct Table {
  using Key = uint64_t;
  using JointStats = typename Bandit::JointStats;
  Hash::Battle hasher;
  std::unordered_map<Key, JointStats> entries;
};

// wrapper to use for enabling matrix ucb at root heap
template <typename Bandit> struct MatrixUCBParams {
  Bandit bandit;
  uint32_t delay;
  uint32_t interval;
  uint32_t minimum;
  float c;
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

template <SearchOptions Options = default_search> struct Search {

  pkmn_gen1_battle_options options;
  pkmn_gen1_chance_options chance_options;
  pkmn_gen1_calc_options calc_options;
  std::array<pkmn_choice, 9> p1_choices;
  std::array<pkmn_choice, 9> p2_choices;
  Hash::State root_hash_state;

  // matrix ucb
  bool initial_solve;
  float ucb_weight;
  std::array<float, 9 + 2> p1_nash;
  std::array<float, 9 + 2> p2_nash;

  // beta
  size_t beta_n = 10;

  // info
  size_t total_depth;
  size_t errors;

  template <typename... Caches>
  Output run(auto &device, const auto budget, const auto &params, auto &heap,
             auto &eval, const Input &input, Output output,
             Caches &...caches) noexcept {

    // reset data members
    *this = {};

    // get choices data here for matrix ucb
    output.p1.k = pkmn_gen1_battle_choices(
        &input.battle, PKMN_PLAYER_P1, pkmn_result_p1(input.result),
        output.p1.choices.data(), PKMN_GEN1_MAX_CHOICES);
    output.p2.k = pkmn_gen1_battle_choices(
        &input.battle, PKMN_PLAYER_P2, pkmn_result_p2(input.result),
        output.p2.choices.data(), PKMN_GEN1_MAX_CHOICES);
    if constexpr (requires { params.bandit; }) {
      ucb_weight = std::log(2 * output.p1.k * output.p2.k);
    }

    if constexpr (is_poke_engine<decltype(eval)>) {
      eval.get_root_score(input.battle);
    }

    auto &stats = [&]() -> auto & {
      if constexpr (is_node<decltype(heap)>) {
        return heap.stats;
      } else {
        heap.hasher.init(input.battle, input.durations);
        root_hash_state = heap.hasher.state();
        return heap.entries[heap.hasher.last()];
      }
    }();

    if (!stats.is_init()) {

      stats.init(output.p1.k, output.p2.k);

      if constexpr (is_contextual_bandit<decltype(get_bandit_params(params))>) {
        static_assert(is_network<decltype(eval)>);
        using output_type = std::remove_cvref_t<decltype(eval)>::T;
        using Eval = std::remove_cvref_t<decltype(eval)>;
        constexpr auto activation = Eval::act;
        constexpr bool rewrite_hp = Eval::rewrite_hp;
        static thread_local uint16_t p1_choice_index[9];
        static thread_local uint16_t p2_choice_index[9];
        static thread_local float p1_logits[9];
        static thread_local float p2_logits[9];
        chance_options.durations = input.durations;
        pkmn_gen1_battle_options_set(&options, nullptr, &chance_options,
                                     nullptr);
        static thread_local std::vector<output_type> battle_embedding;
        battle_embedding.reserve(2 * eval.side_embedding_dim(rewrite_hp));
        write_battle_embedding<activation, rewrite_hp>(
            battle_embedding.data(), PKMN::view(input.battle), eval, caches...);
        for (auto i = 0; i < output.p1.k; ++i) {
          p1_choice_index[i] = Encode::Battle::Policy::get_index(
              PKMN::view(input.battle).sides[0], output.p1.choices[i]);
        }
        for (auto i = 0; i < output.p2.k; ++i) {
          p2_choice_index[i] = Encode::Battle::Policy::get_index(
              PKMN::view(input.battle).sides[1], output.p2.choices[i]);
        }
        output.initial_value = NN::Battle::sigmoid(
            eval.main_net.template propagate<true, activation>(
                battle_embedding.data(), output.p1.k, output.p2.k,
                p1_choice_index, p2_choice_index, p1_logits, p2_logits));
        stats.softmax_logits(get_bandit_params(params), p1_logits, p2_logits);
        std::copy_n(p1_logits, output.p1.k, output.p1.logit.data());
        std::copy_n(p2_logits, output.p2.k, output.p2.logit.data());
        softmax(output.p1.prior.data(), p1_logits, output.p1.k);
        softmax(output.p2.prior.data(), p2_logits, output.p2.k);
        // use policy inference instead of solving
        softmax(p1_nash.data(), p1_logits, output.p1.k);
        softmax(p2_nash.data(), p2_logits, output.p2.k);
        initial_solve = true;
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
                           caches...);
        ++output.iterations;
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);
      }
      // run while boolean flag is set
    } else if constexpr (requires { *budget; }) {
      while (*budget) {
        run_root_iteration(device, params, heap, input, eval, output,
                           caches...);
        ++output.iterations;
      }
      // number of iterations
    } else {
      for (auto i = 0; i < budget; ++i) {
        run_root_iteration(device, params, heap, input, eval, output,
                           caches...);
        ++output.iterations;
      }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    output.duration +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    process_output(output, beta_n);
    return output;
  }

  template <typename... Caches>
  void run_root_iteration(auto &device, const auto &params, auto &heap,
                          const auto &input, auto &eval, Output &output,
                          Caches &...caches) noexcept {

    auto copy = input;
    auto *rng = reinterpret_cast<uint64_t *>(
        copy.battle.bytes + PKMN::Layout::Offsets::Battle::rng);
    rng[0] = device.uniform_64();
    chance_options.durations = copy.durations;
    randomize_hidden_variables(copy.battle, copy.durations);
    pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);
    if constexpr (is_table<decltype(heap)>) {
      heap.hasher.set(root_hash_state);
    }

    if constexpr (!is_matrix_ucb<decltype(params)>) {
      run_iteration(device, params, heap, copy, eval, output, 0, caches...);
    } else {
      if ((output.iterations < params.delay)) {
        run_iteration(device, params.bandit, heap, copy, eval, output, 0,
                      caches...);
      } else {
        const auto [p1_index, p2_index] =
            solve_root_matrix_and_sample(device, params, copy, output);
        const auto c1 = output.p1.choices[p1_index];
        const auto c2 = output.p2.choices[p2_index];
        battle_options_set(copy.battle, 0);
        copy.result = pkmn_gen1_battle_update(&copy.battle, c1, c2, &options);

        const auto value = [&]() {
          if constexpr (is_node<decltype(heap)>) {
            const auto &obs = *reinterpret_cast<const Obs *>(
                pkmn_gen1_battle_options_chance_actions(&options));
            auto &child = heap.children[{p1_index, p2_index, obs}];
            return run_iteration(device, params.bandit, child, copy, eval,
                                 output, 1, caches...);
          } else {
            return run_iteration(device, params.bandit, heap, copy, eval,
                                 output, 1, caches...);
          }
        }();

        // TODO error check
        ++output.visit_matrix[p1_index][p2_index];
        output.value_matrix[p1_index][p2_index] += value.first;
      }
    }
  }

  // typical recursive mcts function
  // we return value for each player because it's slightly faster than calcing
  // 1 - value for each bandit update
  template <typename... Caches>
  std::pair<float, float> run_iteration(auto &device, const auto &bandit,
                                        auto &heap, auto &input, auto &eval,
                                        Output &output, size_t depth,
                                        Caches &...caches) noexcept {
    static constexpr size_t max_depth = 100;

    bool error = false;
    if constexpr (is_table<decltype(heap)>) {
      if (depth >= max_depth) {
        set_turn_limit(input.battle);
        ++errors;
        error = true;
      }
    }

    auto &battle = input.battle;
    auto &result = input.result;
    auto &stats = [&]() -> auto & {
      if constexpr (is_node<decltype(heap)>) {
        return heap.stats;
      } else {
        return heap.entries[heap.hasher.last()];
      }
    }();

    if (stats.is_init() && !error) {
      using Bandit = std::remove_reference_t<decltype(bandit)>;
      using Outcome = typename Bandit::Outcome;
      using JointOutcome = std::pair<Outcome, Outcome>;

      // do bandit
      JointOutcome outcome;
      stats.select(device, bandit, outcome);
      assert(outcome.first.index < 9);
      assert(outcome.second.index < 9);

      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c1 = p1_choices[outcome.first.index];
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c2 = p2_choices[outcome.second.index];

      if constexpr (is_node<decltype(heap)>) {
        battle_options_set(battle, depth);
      } else {
        // slightly stronger to not clamp rolls when using a table?
        // battle_options_set(battle, depth);
        pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      }
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);

      if constexpr (is_table<decltype(heap)>) {
        heap.hasher.update(battle, durations(), c1, c2);
      }

      const auto value = [&]() {
        if constexpr (is_node<decltype(heap)>) {
          const auto &obs = *reinterpret_cast<const Obs *>(
              pkmn_gen1_battle_options_chance_actions(&options));
          auto &child =
              heap.children[{outcome.first.index, outcome.second.index, obs}];
          return run_iteration(device, bandit, child, input, eval, output,
                               depth + 1, caches...);
        } else {
          return run_iteration(device, bandit, heap, input, eval, output,
                               depth + 1, caches...);
        }
      }();
      outcome.first.value = value.first;
      outcome.second.value = value.second;

      if constexpr (is_node<decltype(heap)>) {
        stats.update(outcome);
      } else {
        if (!turn_limit(battle)) {
          stats.update(outcome);
        } else {
          stats.update(outcome);
          // return {0.5, 0.5};
        }
      }

      if (depth == 0) {
        ++output.visit_matrix[outcome.first.index][outcome.second.index];
        output.value_matrix[outcome.first.index][outcome.second.index] +=
            value.first;
      }

      return value;
    }

    total_depth += depth;

    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_NONE:
      [[likely]] {
        using T = decltype(eval);
        float value;
        if constexpr (is_monte_carlo<T>) {
          value = init_stats_and_rollout(stats, device, battle, result);
        } else {
          const auto m = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
              p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
          const auto n = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
              p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
          if (!stats.is_init()) {
            stats.init(m, n);
          }

          if constexpr (is_network<T>) {
            using Eval = std::remove_cvref_t<T>;
            using output_type = Eval::T;
            constexpr auto activation = Eval::act;
            constexpr bool rewrite_hp = Eval::rewrite_hp;

            static thread_local std::vector<output_type> battle_embedding;
            battle_embedding.reserve(2 * eval.side_embedding_dim(rewrite_hp));
            write_battle_embedding<activation, rewrite_hp>(
                battle_embedding.data(), PKMN::view(battle), eval, caches...);

            if constexpr (is_contextual_bandit<decltype(bandit)>) {
              static thread_local uint16_t p1_choice_index[9];
              static thread_local uint16_t p2_choice_index[9];
              static thread_local float p1_logits[9];
              static thread_local float p2_logits[9];
              for (auto i = 0; i < m; ++i) {
                p1_choice_index[i] = Encode::Battle::Policy::get_index(
                    PKMN::view(battle).sides[0], p1_choices[i]);
              }
              for (auto i = 0; i < n; ++i) {
                p2_choice_index[i] = Encode::Battle::Policy::get_index(
                    PKMN::view(battle).sides[1], p2_choices[i]);
              }
              value = NN::Battle::sigmoid(
                  eval.main_net.template propagate<true, activation>(
                      battle_embedding.data(), m, n, p1_choice_index,
                      p2_choice_index, p1_logits, p2_logits));
              stats.softmax_logits(bandit, p1_logits, p2_logits);
            } else {
              value = NN::Battle::sigmoid(
                  eval.main_net.template propagate<activation>(
                      battle_embedding.data()));
            }
            assert(std::isfinite(value));
          } else if constexpr (is_poke_engine<T>) {
            value = eval.evaluate(battle);
          } else {
            static_assert(!std::is_same_v<T, T>);
          }
        }
        return {value, 1 - value};
      }

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
    };
  }

  float init_stats_and_rollout(auto &stats, auto &device,
                               pkmn_gen1_battle &battle,
                               pkmn_result result) noexcept {

    auto seed = device.uniform_64();
    auto m = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1,
                                      pkmn_result_p1(result), p1_choices.data(),
                                      PKMN_GEN1_MAX_CHOICES);
    auto c1 = p1_choices[seed % m];
    auto n = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2,
                                      pkmn_result_p2(result), p2_choices.data(),
                                      PKMN_GEN1_MAX_CHOICES);
    seed >>= 32;
    auto c2 = p2_choices[seed % n];
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    if (!stats.is_init()) {
      stats.init(m, n);
    }
    while (!pkmn_result_type(result)) {
      seed = device.uniform_64();
      m = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1,
                                   pkmn_result_p1(result), p1_choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      c1 = p1_choices[seed % m];
      n = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2,
                                   pkmn_result_p2(result), p2_choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      seed >>= 32;
      c2 = p2_choices[seed % n];
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    }
    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_WIN: {
      return 1;
    }
    case PKMN_RESULT_LOSE: {
      return 0;
    }
    case PKMN_RESULT_TIE: {
      return 0.5;
    }
    default: {
      assert(false);
      return 0.5;
    }
    };
  }

  inline auto solve_root_matrix_and_sample(auto &device, auto &params,
                                           auto &copy,
                                           const auto &output) noexcept {
    uint8_t p1_index{}, p2_index{};
    const bool periodic_solve =
        (((output.iterations + 1) % params.interval) == 0);
    if (periodic_solve || !initial_solve) {
      // get ucb matrices
      std::array<int, 9 * 9> p1_ucb_matrix;
      std::array<int, 9 * 9> p2_ucb_matrix;
      constexpr int discretize_factor = 256;
      const float log_T = std::log(output.iterations);
      for (auto i = 0; i < output.p1.k; ++i) {
        for (auto j = 0; j < output.p2.k; ++j) {
          float p1_entry = 0;
          float p2_entry = 1;
          if (output.visit_matrix[i][j] < params.minimum) {
            return std::pair<uint8_t, uint8_t>{i, j};
          }
          if (output.visit_matrix[i][j] > 0) {
            p1_entry = output.value_matrix[i][j] / output.visit_matrix[i][j];
            p2_entry = p1_entry;
          }
          const float exploration =
              params.c * std::sqrt(2 * (2 * log_T + ucb_weight) /
                                   (output.visit_matrix[i][j] + 1));
          p1_entry += exploration;
          p2_entry -= exploration;
          p1_ucb_matrix[i * output.p2.k + j] = p1_entry * discretize_factor;
          p2_ucb_matrix[i * output.p2.k + j] = p2_entry * discretize_factor;
        }
      }

      // solve and sample
      std::array<float, 9 + 2> dummy;
      LRSNash::FastInput p1_solve_input{
          static_cast<int>(output.p1.k), static_cast<int>(output.p2.k),
          p1_ucb_matrix.data(), discretize_factor};
      LRSNash::FloatOneSumOutput p1_solve_output{p1_nash.data(), dummy.data(),
                                                 0};
      LRSNash::solve_fast(&p1_solve_input, &p1_solve_output);
      LRSNash::FastInput p2_solve_input{
          static_cast<int>(output.p1.k), static_cast<int>(output.p2.k),
          p2_ucb_matrix.data(), discretize_factor};
      LRSNash::FloatOneSumOutput p2_solve_output{dummy.data(), p2_nash.data(),
                                                 0};
      LRSNash::solve_fast(&p2_solve_input, &p2_solve_output);

      initial_solve = true;
    }

    float p = device.uniform();
    for (auto i = 0; i < output.p1.k; ++i) {
      p -= p1_nash[i];
      if (p <= 0) {
        p1_index = i;
        break;
      }
    }
    p = device.uniform();
    for (auto i = 0; i < output.p2.k; ++i) {
      p -= p2_nash[i];
      if (p <= 0) {
        p2_index = i;
        break;
      }
    }

    return std::pair<uint8_t, uint8_t>{p1_index, p2_index};
  }

  // pkmn_gen1_battle_options_set with constexpr logic
  void battle_options_set(pkmn_gen1_battle &battle, size_t depth) {
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

  // use battle seed to quickly compute a clamped damage roll
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

  void set_turn_limit(auto &battle) const noexcept {
    reinterpret_cast<uint16_t *>(battle.bytes +
                                 PKMN::Layout::Offsets::Battle::turn)[0] = 1000;
  }

  inline bool turn_limit(const auto &battle) const noexcept {
    return reinterpret_cast<const uint16_t *>(
               battle.bytes + PKMN::Layout::Offsets::Battle::turn)[0] >= 1000;
  }

  const auto &durations() const noexcept {
    return *pkmn_gen1_battle_options_chance_durations(&options);
  }

  void process_output(Output &output, size_t beta_n = 0) noexcept {
    // prepare output, solve empirical root matrix if enabled
    // output.empirical_value = output.total_value / output.iterations;
    double total_value = 0;

    output.p1.empirical = {};
    output.p2.empirical = {};

    constexpr int discretize_factor = 256;
    std::array<int, 9 * 9> solve_matrix;
    for (int i = 0; i < output.p1.k; ++i) {
      for (int j = 0; j < output.p2.k; ++j) {
        total_value += output.value_matrix[i][j];
        auto n = output.visit_matrix[i][j];
        output.p1.empirical[i] += n;
        output.p2.empirical[j] += n;
        n += !n;
        solve_matrix[output.p2.k * i + j] =
            output.value_matrix[i][j] / n * discretize_factor;
      }
    }

    output.empirical_value = total_value / output.iterations;
    LRSNash::FastInput solve_input{static_cast<int>(output.p1.k),
                                   static_cast<int>(output.p2.k),
                                   solve_matrix.data(), discretize_factor};
    // LRSNash convention: 2 extra entries needed for output denom, nash value
    std::array<float, 9 + 2> nash1{}, nash2{};
    LRSNash::FloatOneSumOutput solve_output{nash1.data(), nash2.data(), 0};
    LRSNash::solve_fast(&solve_input, &solve_output);

    for (int i = 0; i < output.p1.k; ++i) {
      output.p1.empirical[i] /= (float)output.iterations;
      output.p1.nash[i] = nash1[i];
    }
    for (int j = 0; j < output.p2.k; ++j) {
      output.p2.empirical[j] /= (float)output.iterations;
      output.p2.nash[j] = nash2[j];
    }
    output.nash_value = solve_output.value;

    output.p1.beta = {};
    output.p2.beta = {};

    // std::mt19937 rd{std::random_device{}()};
    // for (auto k = 0; k < beta_n; ++k) {
    //   for (int i = 0; i < output.p1.k; ++i) {
    //     for (int j = 0; j < output.p2.k; ++j) {
    //       auto n = output.visit_matrix[i][j];
    //       double v = output.value_matrix[i][j];
    //       if (n == 0) {
    //         n = 1;
    //         v = .5;
    //       }
    //       auto w = beta_sample(v, n, rd);
    //       solve_matrix[output.p2.k * i + j] = w * discretize_factor;
    //     }
    //   }
    //   LRSNash::FastInput solve_input{static_cast<int>(output.p1.k),
    //                                  static_cast<int>(output.p2.k),
    //                                  solve_matrix.data(), discretize_factor};
    //   std::array<float, 9 + 2> nash1{}, nash2{};
    //   LRSNash::FloatOneSumOutput solve_output{nash1.data(), nash2.data(), 0};
    //   LRSNash::solve_fast(&solve_input, &solve_output);
    //   for (auto i = 0; i < output.p1.k; ++i) {
    //     output.p1.beta[i] += nash1[i] / beta_n;
    //   }
    //   for (auto i = 0; i < output.p2.k; ++i) {
    //     output.p2.beta[i] += nash2[i] / beta_n;
    //   }
    // }
  }

  template <NN::Activation activation, bool rewrite_hp, typename T,
            typename... Caches>
  void write_battle_embedding(T *embedding, const PKMN::Battle &battle,
                              NN::Battle::NetworkBase &eval,
                              Caches &...caches) {
    const auto &d = PKMN::view(durations());
    if constexpr (sizeof...(Caches) == 2) {
      auto &p1_cache = std::get<0>(std::tie(caches...));
      auto &p2_cache = std::get<1>(std::tie(caches...));
      embedding = NN::Battle::write_side_embedding<T, activation, rewrite_hp>(
          embedding, battle.sides[0], d.get(0), eval, p1_cache);
      embedding = NN::Battle::write_side_embedding<T, activation, rewrite_hp>(
          embedding, battle.sides[1], d.get(1), eval, p2_cache);
    } else {
      embedding = NN::Battle::write_side_embedding<T, activation, rewrite_hp>(
          embedding, battle.sides[0], d.get(0), eval);
      embedding = NN::Battle::write_side_embedding<T, activation, rewrite_hp>(
          embedding, battle.sides[1], d.get(1), eval);
    }
  }
};

} // namespace MCTS