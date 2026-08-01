#pragma once

#include <nn/battle/network.h>
#include <py/libpkmn/data.h>
#include <py/search/data.h>
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

MCTS::Output run(mt19937 &device, const Py::PKMN::Battle &battle,
                 const Py::PKMN::Durations &durations,
                 const Py::Search::Budget &budget,
                 const Py::Search::BanditParams &params, Py::Search::Heap &heap,
                 Py::Search::Eval &eval);

MCTS::Output run(mt19937 &device, const Py::PKMN::Battle &battle,
                 const Py::PKMN::Durations &durations,
                 const Py::Search::Budget &budget,
                 const Py::Search::BanditParams &params, Py::Search::Heap &heap,
                 Py::Search::Eval &eval, Py::Search::SideCache &p1_cache,
                 Py::Search::SideCache &p2_cache);

} // namespace RuntimeSearch