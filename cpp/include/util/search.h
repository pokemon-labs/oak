#pragma once

#include <nn/battle/network.h>
#include <search/data.h>
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

MCTS::Output run(mt19937 &device, const pkmn_gen1_battle &battle,
                 const pkmn_gen1_chance_durations &durations,
                 const Search::Budget &budget,
                 const Search::BanditParams &params, Search::Heap &heap,
                 Search::Eval &eval, MCTS::Output output = {},
                 Search::SideCache *p1_cache = nullptr,
                 Search::SideCache *p2_cache = nullptr);
} // namespace RuntimeSearch