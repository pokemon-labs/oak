#include <util/search.h>

#include <search/mcts.h>
#include <util/file-lock.h>
#include <util/strings.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

// budget
// #define NO_ITERATION
// #define NO_DURATION
#define NO_FLAG

// evals
// #define NO_MONTE_CARLO
// #define NO_POKE_ENGINE
// #define NO_NETWORK

// bandits
// #define NO_UCB
#define NO_UCB1
// #define NO_PUCB
// #define NO_EXP3
// #define NO_PEXP3
// #define NO_REGRET_MATCHING
// matrix ucb
// #define NO_MATRIX_UCB

// heap
// #define NO_NODE
#define NO_TABLE

namespace RuntimeSearch {

MCTS::Output run(mt19937 &device, const pkmn_gen1_battle &battle,
                 const pkmn_gen1_chance_durations &durations,
                 const Search::Budget &budget, const Search::Bandit &params,
                 Search::Heap &heap, Search::Eval &eval, MCTS::Output output,
                 MCTS::RuntimeOptions options, Search::SideCache *p1_cache_ptr,
                 Search::SideCache *p2_cache_ptr) {

  MCTS::Input input{battle, durations, PKMN::result(battle)};

  const auto parse_eval = [&](const auto dur, const auto &params, auto &heap) {
    MCTS::Search s{};
    if (false) {
    }
#ifndef NO_NETWORK
    else if (auto *ptr = std::get_if<std::shared_ptr<NN::Battle::NetworkBase>>(
                 &eval.data)) {
      auto network_search = [&](auto &net) {
        using output_type = std::remove_cvref_t<decltype(net)>::T;
        constexpr auto activation = std::remove_cvref_t<decltype(net)>::act;

        const bool has_caches = p1_cache_ptr && p2_cache_ptr;
        if (has_caches) {
          if constexpr (std::is_same_v<output_type, float>) {
            auto &p1_cache =
                std::get<Search::SideCache::Cache<float>>(p1_cache_ptr->data);
            auto &p2_cache =
                std::get<Search::SideCache::Cache<float>>(p2_cache_ptr->data);
            output = s.run(device, dur, params, heap, net, input, output,
                           options, p1_cache, p2_cache);
          } else {
            auto &p1_cache =
                std::get<Search::SideCache::Cache<uint8_t>>(p1_cache_ptr->data);
            auto &p2_cache =
                std::get<Search::SideCache::Cache<uint8_t>>(p2_cache_ptr->data);
            output = s.run(device, dur, params, heap, net, input, output,
                           options, p1_cache, p2_cache);
          }
        } else {
          output =
              s.run(device, dur, params, heap, net, input, output, options);
          // throw std::runtime_error{"Network search must use caches."};
        }
      };
      auto net = *ptr; // shared ptr to base
      NN::Battle::visit_network(net, network_search);
      return output;
    }
#endif
    else {
      if constexpr (MCTS::is_contextual_bandit<decltype(MCTS::get_bandit_params(
                        params))>) {
        throw std::runtime_error{"Contextual bandits must use network eval"};
        return output;
      } else {
        if (false) {
        }
#ifndef NO_MONTE_CARLO
        else if (auto *ptr = std::get_if<MCTS::MonteCarlo>(&eval.data)) {
          return s.run(device, dur, params, heap, *ptr, input, output, options);
        }
#endif
#ifndef NO_POKE_ENGINE
        else if (auto *ptr = std::get_if<PokeEngine::Eval>(&eval.data)) {
          return s.run(device, dur, params, heap, *ptr, input, output, options);
        }
#endif
        else {
          throw std::runtime_error{"Invalid Eval"};
          return output;
        }
      }
    }
  };

  const auto parse_heap = [&](auto dur, auto params) {
    constexpr bool is_matrix_ucb = []() {
      if constexpr (requires { params.bandit; }) {
        return true;
      } else {
        return false;
      }
    }();
    auto bandit = [is_matrix_ucb](auto params) {
      if constexpr (is_matrix_ucb) {
        return params.bandit;
      } else {
        return params;
      }
    }(params);
    using Bandit = std::remove_cvref_t<decltype(bandit)>;
    using Node = MCTS::Node<Bandit>;
    using MatrixUCBNode = MCTS::MatrixUCBNode<Bandit>;
    using Table = MCTS::Table<Bandit>;
    auto *data = &heap.data;
    if (false) {
      // TODO yolo'd the constexpr below and don't understand it
    } else if constexpr (is_matrix_ucb) {
      auto *ptr = std::get_if<Search::Heap::MatrixUCBNodeVariant>(data);
      if (!ptr) {
        throw std::runtime_error{"MatrixUCB parse error"};
      }
      auto &mucb_node_variant = *ptr;
      if (std::holds_alternative<std::monostate>(mucb_node_variant)) {
        mucb_node_variant = MatrixUCBNode{};
      }
      if (std::holds_alternative<MatrixUCBNode>(mucb_node_variant)) {
        return parse_eval(dur, params,
                          std::get<MatrixUCBNode>(mucb_node_variant));
      } else {
        throw std::runtime_error{
            "MatrixUCBNode exists but does not match bandit"};
        return output;
      }
    }
#ifndef NO_NODE
    else if (auto *ptr = std::get_if<Search::Heap::NodeVariant>(data)) {
      auto &node_variant = *ptr;
      if (std::holds_alternative<std::monostate>(node_variant)) {
        node_variant = Node{};
      }
      if (std::holds_alternative<Node>(node_variant)) {
        return parse_eval(dur, params, std::get<Node>(node_variant));
      } else {
        throw std::runtime_error{"Node exists but does not match bandit"};
        return output;
      }
#endif
#ifndef NO_TABLE
    } else if (auto *ptr = std::get_if<Search::Heap::TableVariant>(data)) {
      auto &table_variant = *ptr;
      if (std::holds_alternative<std::monostate>(table_variant)) {
        table_variant = Table{};
      }
      if (std::holds_alternative<Table>(table_variant)) {
        return parse_eval(dur, params, std::get<Table>(table_variant));
      } else {
        throw std::runtime_error{"Node exists but does not match bandit"};
        return output;
      }
#endif
      // } else if () {

    } else {
      throw std::runtime_error{"Invalid heap"};
      return output;
    }
  };

  const auto parse_matrix_ucb_params = [&](auto dur, auto bandit) {
    const auto *matrix_ucb_params =
        dynamic_cast<const Search::MatrixUCB *>(&params);
    if (false) {
    }
#ifndef NO_MATRIX_UCB
    else if (matrix_ucb_params) {
      auto matrix_ucb =
          MCTS::MatrixUCBParams<std::remove_cvref_t<decltype(bandit)>>{};
      matrix_ucb.bandit = bandit;
      matrix_ucb.c = matrix_ucb_params->c;
      matrix_ucb.interval = matrix_ucb_params->interval;
      matrix_ucb.grow = matrix_ucb_params->grow;
      matrix_ucb.discretize_factor = matrix_ucb_params->discretize_factor;
      return parse_heap(dur, matrix_ucb);
    }
#endif
    else {
      return parse_heap(dur, bandit);
    }
  };

  const auto parse_bandit = [&](auto dur) {
    const auto *data = &params.data;
    if (false) {
#ifndef NO_EXP3
    } else if (auto *ptr = std::get_if<MCTS::Exp3>(data)) {
      return parse_matrix_ucb_params(dur, *ptr);
#endif
#ifndef NO_PEXP3
    } else if (auto *ptr = std::get_if<MCTS::PExp3>(data)) {
      return parse_matrix_ucb_params(dur, *ptr);
#endif
#ifndef NO_UCB
    } else if (auto *ptr = std::get_if<MCTS::UCB>(data)) {
      return parse_matrix_ucb_params(dur, *ptr);
#endif
#ifndef NO_PUCB
    } else if (auto *ptr = std::get_if<MCTS::PUCB>(data)) {
      return parse_matrix_ucb_params(dur, *ptr);
#endif
#ifndef NO_UCB1
    } else if (auto *ptr = std::get_if<MCTS::UCB1>(data)) {
      return parse_matrix_ucb_params(dur, *ptr);
#endif
#ifndef NO_REGRET_MATCHING
    } else if (auto *ptr = std::get_if<MCTS::RegretMatching>(data)) {
      return parse_matrix_ucb_params(dur, *ptr);
#endif
    } else {
      throw std::runtime_error{"Invalid bandit"};
      return output;
    }
  };

  const auto parse_budget_and_search = [&]() {
    if (false) {
    }
#ifndef NO_FLAG
    else if (auto *ptr = std::get_if<bool *>(&budget.data)) {
      return parse_bandit(*ptr);
    }
#endif
#ifndef NO_ITERATION
    else if (auto *ptr = std::get_if<size_t>(&budget.data)) {
      return parse_bandit(*ptr);
    }
#endif
#ifndef NO_DURATION
    else if (auto *ptr = std::get_if<std::chrono::milliseconds>(&budget.data)) {
      return parse_bandit(*ptr);
    }
#endif
    else {
      throw std::runtime_error("Invalid search budget.");
      return output;
    }
  };

  return parse_budget_and_search();
}

} // namespace RuntimeSearch
