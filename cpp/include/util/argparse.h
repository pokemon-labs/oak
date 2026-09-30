#pragma once

#include "argparse/argparse.hpp"

namespace Argparse {
template <typename T> using Identity = T;

struct TeamBuildingArgs : public argparse::Args {
  double &team_modify_prob =
      kwarg("team-modify-prob", "Probability the base team (from sample teams "
                                "or teams file) is modified")
          .set_default(0);
  double &pokemon_delete_prob =
      kwarg("pokemon-delete-prob",
            "Probability a set (species + moveset) is omitted")
          .set_default(0);
  double &move_delete_prob =
      kwarg("move-delete-prob", "Probability a move is omitted").set_default(0);
  std::string &build_network_path =
      kwarg("build-network-path", "").set_default("");
  int &max_pokemon = kwarg("max-pokemon", "Max team size").set_default(6);
};

#define MAKE_AGENT_ARGS(NAME, BASE, WRAPPER, A, B)                             \
  struct NAME : public BASE {                                                  \
    WRAPPER<std::string> &A##budget =                                          \
        kwarg(B "budget", "Search budget, e.g. 1024/100ms/8s");                \
                                                                               \
    WRAPPER<std::string> &A##bandit =                                          \
        kwarg(B "bandit", "Bandit algorithm and parameters");                  \
                                                                               \
    WRAPPER<std::string> &A##eval =                                            \
        kwarg(B "eval", "Eval mc/fp/<network-path>");                          \
                                                                               \
    std::optional<std::string> &A##matrix_ucb = kwarg(                         \
        B "matrix-ucb",                                                        \
        "MatrixUCB C-INTERVAL-GROW(-DISCRETIZE). The UCB Matrix is solved "    \
        "iff "                                                                 \
        "(total_visits + 1) \% INTERVAL == 0, otherwise the network "          \
        "priors/uniform-strategy/cached-Nash are used. INTERVAL=1 in "         \
        "the paper but that's untenably slow. GROW will 'expand' all extant "  \
        "joint bandit nodes oorresponding to a choice pair in MatrixUCB "      \
        "nodes, but only when the parent node's root visits + 1 == GROW. "     \
        "This means GROW=0 will never expand nodes and GROW=1 "                \
        "will immediately expand. DISCRETIZE is the quantization factor when " \
        "coverting float matrices to int for LRSNash. Error (exploitability) " \
        "is bounded by 2/DISCRETIZE");                                         \
                                                                               \
    std::optional<std::string> &A##options = kwarg(                            \
        B "options",                                                           \
        "Runtime search options DEPTH-ROLLOUT-TEMP. DEPTH is the max depth "   \
        "the search tree will grow. The default value of 0 means unbounded. "  \
        "ROLLOUT is how many updates a battle is updated at leaf node "        \
        "evaluation. A value of 0 means always rollout until terminal, and a " \
        "value of N>0 means rollout for N-1 many turns (default=1). TEMP is "  \
        "the temperature of the network policy during rollout.");              \
                                                                               \
    bool &A##quantize = flag(B "quantize", "Use quantized main subnet");       \
  };

#define MAKE_AGENT_POLICY_ARGS(NAME, BASE, WRAPPER, A, B)                      \
  struct NAME : public BASE {                                                  \
    WRAPPER<std::string> &A##policy_mode =                                     \
        kwarg(B "policy-mode", "Policy mode");                                 \
    std::optional<double> &A##policy_temp =                                    \
        kwarg(B "policy-temp", "P-norm just before clipping/sampling")         \
            .set_default(1.0);                                                 \
    std::optional<double> &A##policy_min =                                     \
        kwarg(B "policy-min", "Probs below this will be zerod")                \
            .set_default(0);                                                   \
  };

#define MAKE_AGENT_ADJUDICATE_ARGS(NAME, BASE, WRAPPER, A, B)                  \
  struct NAME : public BASE {                                                  \
    WRAPPER<double> &A##forfeit_value =                                        \
        kwarg(B "forfeit-value",                                               \
              "Adjuticate the battle when the magnitude of the inverse "       \
              "sigmoid of the value estimate is larger than this number for "  \
              "--n-forfeit "                                                   \
              "of the latest searches.")                                       \
            .set_default(0);                                                   \
    WRAPPER<size_t> &A##forfeit_n =                                            \
        kwarg(B "forfeit-n", "Min consectutive turns to adjudicate")           \
            .set_default(1);                                                   \
  }; // namespace Argparse

MAKE_AGENT_ARGS(AgentArgs, TeamBuildingArgs, Identity, , "")
MAKE_AGENT_ARGS(AgentArgsOptional, TeamBuildingArgs, std::optional, , "")
using BenchmarkArgs = AgentArgsOptional;

MAKE_AGENT_POLICY_ARGS(AgentPolicyArgs, AgentArgs, Identity, , "")
MAKE_AGENT_POLICY_ARGS(FastAgentPolicyArgs, AgentPolicyArgs, std::optional,
                       fast_, "fast-")
MAKE_AGENT_ARGS(FastAgentArgs, FastAgentPolicyArgs, std::optional, fast_,
                "fast-")
MAKE_AGENT_ARGS(T1AgentArgs, FastAgentArgs, std::optional, t1_, "t1-")
MAKE_AGENT_ADJUDICATE_ARGS(GenerateArgs, T1AgentArgs, std::optional, , "")
// using GenerateArgs = T1AgentArgs;

MAKE_AGENT_POLICY_ARGS(AgentOptionalPolicyArgs, AgentArgsOptional,
                       std::optional, , "")
MAKE_AGENT_POLICY_ARGS(P1PolicyArgs, AgentOptionalPolicyArgs, std::optional,
                       p1_, "p1-")
MAKE_AGENT_POLICY_ARGS(P2PolicyArgs, P1PolicyArgs, std::optional, p2_, "p2-")
MAKE_AGENT_ARGS(P1AgentArgs, P2PolicyArgs, std::optional, p1_, "p1-")
MAKE_AGENT_ARGS(P2AgentArgs, P1AgentArgs, std::optional, p2_, "p2-")
MAKE_AGENT_ADJUDICATE_ARGS(VsArgs, P2AgentArgs, std::optional, , "")
// using VsArgs = P2AgentArgs;

} // namespace Argparse

using Argparse::BenchmarkArgs;
using Argparse::GenerateArgs;
using Argparse::VsArgs;
