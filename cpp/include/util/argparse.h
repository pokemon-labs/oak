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
    WRAPPER<std::string> &A##matrix_ucb =                                      \
        kwarg(B "matrix-ucb", "MatrixUCB start/interval/minimum/c")            \
            .set_default("");                                                  \
                                                                               \
    WRAPPER<std::string> &A##eval =                                            \
        kwarg(B "eval", "Eval mc/fp/<network-path>");                          \
                                                                               \
    bool &A##use_discrete =                                                    \
        flag(B "use-discrete", "Use quantized main subnet");                   \
                                                                               \
    bool &A##use_table =                                                       \
        flag(B "use-table", "Use a transposition table instead of a tree");    \
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
              "Forfeit when value lies outside this range for n-many turns")   \
            .set_default("0.0");                                               \
    WRAPPER<size_t> &A##forfeit_n =                                            \
        kwarg(B "forfeit-n", "Min consectutive turns to forfeit")              \
            .set_default(1);                                                   \
  }; // namespace Argparse

MAKE_AGENT_ARGS(AgentArgs, TeamBuildingArgs, Identity, , "")
MAKE_AGENT_ARGS(AgentArgsOptional, TeamBuildingArgs, std::optional, , "")
using BenchmarkArgs = AgentArgsOptional;

MAKE_AGENT_POLICY_ARGS(AgentPolicyOptionalArgs, AgentArgsOptional,
                       std::optional, , "")
using ChallArgs = AgentPolicyOptionalArgs;

MAKE_AGENT_POLICY_ARGS(AgentPolicyArgs, AgentArgs, Identity, , "")
MAKE_AGENT_POLICY_ARGS(FastAgentPolicyArgs, AgentPolicyArgs, std::optional,
                       fast_, "fast-")
MAKE_AGENT_ARGS(FastAgentArgs, FastAgentPolicyArgs, std::optional, fast_,
                "fast-")
MAKE_AGENT_ARGS(T1AgentArgs, FastAgentArgs, std::optional, t1_, "t1-")
using GenerateArgs = T1AgentArgs;

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
using Argparse::ChallArgs;
using Argparse::GenerateArgs;
using Argparse::VsArgs;
