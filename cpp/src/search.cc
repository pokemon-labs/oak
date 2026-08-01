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
// #define NO_FLAG

// evals
// #define NO_MONTE_CARLO
// #define NO_POKE_ENGINE
// #define NO_NETWORK

// bandits
// #define NO_UCB
// #define NO_UCB1
// #define NO_PUCB
// #define NO_EXP3
// #define NO_PEXP3
// matrix ucb
// #define NO_MATRIX_UCB

// heap
// #define NO_NODE
// #define NO_TABLE

namespace RuntimeSearch {

MCTS::Output run(mt19937 &device, const Py::PKMN::Battle &battle,
                 const Py::PKMN::Durations &durations,
                 const Py::Search::Budget &budget,
                 const Py::Search::BanditParams &params, Py::Search::Heap &heap,
                 Py::Search::Eval &eval) {

  MCTS::Input input {
    battle.raw, durations.raw, PKMN::result(battle.raw, durations.raw);
  };

  const auto parse_eval = [&](const auto dur, const auto &params, auto &heap) {
    MCTS::Search s{};
    if (false) {
    }
#ifndef NO_MONTE_CARLO
    else if (auto *ptr = std::get_if<MCTS::MonteCarlo>(&eval.data); ptr) {
      return s.run(device, dur, params, heap, *ptr, input, output);
    }
#endif
#ifndef NO_POKE_ENGINE
    else if (auto *ptr = std::get_if<PokeEngine::Eval>(&eval.data); ptr) {
      return s.run(device, dur, params, heap, *ptr, input, output);
    }
#endif
#ifndef NO_NETWORK
    else if (auto *ptr = std::get_if<std::shared_ptr<NN::Battle::NetworkBase>>(
                 &eval.data)) {
      auto *network = ptr->get();
      // if (!agent.network_ptr) {
      //   agent.initialize_network(input.battle);
      // }
      // if (auto network =
      //         dynamic_cast<NN::Battle::Network *>(agent.network_ptr.get())) {
      //   return s.run(device, dur, params, heap, *network, input, output);
      // } else if (auto network = dynamic_cast<NN::Battle::NetworkClamped *>(
      //                agent.network_ptr.get())) {
      //   return s.run(device, dur, params, heap, *network, input, output);
      // } else if (auto network = dynamic_cast<NN::Battle::NetworkScaled *>(
      //                agent.network_ptr.get())) {
      //   return s.run(device, dur, params, heap, *network, input, output);
      // } else {
      //   const auto [id, hd, vd, pd] = agent.network_ptr->shape();
      //   auto q_network_ptr = NN::Battle::visit_quantized_network(
      //       id, hd, vd, pd,
      //       [&](auto &net) {
      //         output = s.run(device, dur, params, heap, net, input, output);
      //       },
      //       std::move(agent.network_ptr));
      //   if (q_network_ptr) {
      //     agent.network_ptr = std::move(q_network_ptr);
      //   }
      //   return output;
      // }
      return output;
    }
#endif
    else {
      throw std::runtime_error{"Invalid Eval"};
      return output;
    }
  };

  const auto parse_params = [&](auto dur, auto bandit, auto &heap) {
    auto *matrix_ucb_params = dynamic_cast<Py::Search::MatrixUCB>(&params);
    using T = std::remove_cvref_t<decltype(bandit)>::Params;
    auto bandit_params = std::get<T>(params.data);

    if (false) {
    }
#ifndef NO_MATRIX_UCB
    else if (matrix_ucb_params) {
      auto eee = MCTS::MatrixUCBParams<T>{};
      eee.bandit_params = bandit_params;
      eee.delay = matrix_ucb_params->delay;
      eee.interval = matrix_ucb_params->interval;
      eee.minimum = matrix_ucb_params->minimum;
      eee.c = matrix_ucb_params->C;
      return parse_eval(dur, eee, heap);
    }
#endif
    else {
      return parse_eval(dur, bandit_params, heap)
    }
  };

  const auto parse_heap = [&](auto dur, auto bandit) {
    using Node = MCTS::Node<Joint<std::remove_cvref_t<decltype(bandit)>>>;
    using Table = MCTS::Table<Joint<std::remove_cvref_t<decltype(bandit)>>>;
    if (false) {
    }
#ifndef NO_NODE
    else if (auto *ptr =
                 std::get_if<Py::Search::Heap::NodeVariant>(&heap.data)) {
      auto &node_variant = *ptr;
      if (std::holds_alternative<std::monostate>(node_variant)) {
        node_variant = Node{};
      }
      if (std::holds_alternative<Node>(node_variant)) {
        return parse_params(dur, bandit, std::get<Node>(node_variant));
      } else {
        throw std::runtime_error{"Node exists but does not match bandit"};
        return output;
      }
#endif
#ifndef NO_TABLE
    } else if (auto *ptr =
                   std::get_if<Py::Search::Heap::TableVariant>(&heap.data)) {
      auto &table_variant = *ptr;
      if (std::holds_alternative<std::monostate>(table_variant)) {
        table_variant = Table{};
      }
      if (std::holds_alternative<Table>(table_variant)) {
        return parse_params(dur, bandit, std::get<Table>(table_variant));
      } else {
        throw std::runtime_error{"Node exists but does not match bandit"};
        return output;
      }
#endif
    } else {
      throw std::runtime_error{"Invalid heap"};
      return output;
    }
  };

  const auto parse_bandit = [&](auto dur) {
    if (false) {
#ifndef NO_EXP3
    } else if (auto *ptr = std::get_if<Exp3::Bandit::Params>(&params.data);
               ptr) {
      return parse_heap(dur, Exp3::Bandit{});
#endif
#ifndef NO_PEXP3
    } else if (auto *ptr = std::get_if<PExp3::Bandit::Params>(&params.data);
               ptr) {
      return parse_heap(dur, PExp3::Bandit{});
#endif
#ifndef NO_UCB
    } else if (auto *ptr = std::get_if<UCB::Bandit::Params>(&params.data);
               ptr) {
      return parse_heap(dur, UCB::Bandit{});
#endif
#ifndef NO_PUCB
    } else if (auto *ptr = std::get_if<PUCB::Bandit::Params>(&params.data);
               ptr) {
      return parse_heap(dur, PUCB::Bandit{});
#endif
#ifndef NO_UCB1
    } else if (auto *ptr = std::get_if<UCB::Bandit::Params>(&params.data);
               ptr) {
      return parse_heap(dur, UCB1::Bandit{});
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
    else if (auto *ptr = std::get_if<bool *>(&budget.data); ptr) {
      return parse_bandit(*ptr);
    }
#endif
#ifndef NO_ITERATION
    else if (auto *ptr = std::get_if<size_t>(&budget.data); ptr) {
      return parse_bandit(*ptr);
    }
#endif
#ifndef NO_DURATION
    else if (auto *ptr = std::get_if<std::chrono::milliseconds>(&budget.data);
             ptr) {
      return parse_bandit(*ptr);
    }
#endif
    else {
      throw std::runtime_error("Invalid search duration specification: " +
                               agent.budget);
      return output;
    }
  };

  return parse_budget_and_search();
}

} // namespace RuntimeSearch

// budget
#undef NO_ITERATION
#undef NO_DURATION
#undef NO_FLAG

// evals
#undef NO_MONTE_CARLO
#undef NO_POKE_ENGINE
#undef NO_NETWORK

// bandits
#undef NO_UCB
#undef NO_UCB1
#undef NO_PUCB
#undef NO_EXP3
#undef NO_PEXP3
#undef NO_MATRIX_UCB

// heap
#undef NO_NODE
#undef NO_TABLE