#pragma once

#include <nn/battle/network.h>
#include <search/bandit/exp3-ix.h>
#include <search/bandit/exp3.h>
#include <search/bandit/pexp3.h>
#include <search/bandit/pucb.h>
#include <search/bandit/ucb.h>
#include <search/bandit/ucb1.h>
#include <search/mcts.h>
#include <util/random.h>

#include <memory>
#include <variant>

namespace RuntimeSearch {

struct Heap {

  template <typename... T>
  using BanditVariantT =
      std::variant<std::monostate, MCTS::Node<T>..., MCTS::Table<T>...>;

  using BanditVariant =
      BanditVariantT<Exp3::JointBandit, Exp3IX::JointBandit, PExp3::JointBandit,
                     UCB::JointBandit, PUCB::JointBandit, UCB1::JointBandit>;

  BanditVariant data;

  bool update(uint8_t i, uint8_t j, const MCTS::Obs &obs);
  std::string type() const noexcept;
  bool empty() const noexcept;
};

struct AgentParams {
  std::string budget;
  std::string bandit;
  std::string eval;
  std::string matrix_ucb;
  bool discrete;
  bool table;

  constexpr bool operator==(const AgentParams &) const = default;
};

struct Agent : AgentParams {

  std::unique_ptr<NN::Battle::NetworkBase> network_ptr{};

  Agent(const AgentParams &params) : AgentParams{params}, network_ptr{} {}
  Agent() = default;

  constexpr bool operator==(const Agent &other) const {
    return static_cast<AgentParams>(*this) == static_cast<AgentParams>(other);
  }

  bool is_monte_carlo() const {
    return eval.empty() || eval == "mc" || eval == "montecarlo" ||
           eval == "monte-carlo";
  }
  bool is_foul_play() const { return eval == "fp"; }
  bool is_network() const { return !is_monte_carlo() && !is_foul_play(); }

  void initialize_network(const pkmn_gen1_battle &b);
};

MCTS::Output run(mt19937 &device, const MCTS::Input &input, Heap &heap_variant,
                 Agent &agent, MCTS::Output output = {}, bool *const flag = {});

} // namespace RuntimeSearch