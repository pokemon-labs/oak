// Python binding for src/env.hpp's SelfPlay (birch's birch_env.cpp on
// libpkmn's Gen 1). Plain pybind11 + numpy: every array is new, so
// torch.from_numpy wraps it without a copy. Built by CMakeLists.txt into
// ataraxos/_ext/.
//
// SelfPlay(battles, seed=1, teams=[], max_turns=0, shards=64, threads=8,
//          state=False):
// `teams` is a list of Showdown export files, pooled (none: truly random
// teams); `max_turns` ends a battle as turn max_turns + 1 starts, a tie
// with end_capped set (0: Showdown's 1000); `state` adds each decision's
// bookkeeping state (src/state_tokens.hpp).
//
// `reset()` and `step(picks)` return the next decisions as a dict:
//   battle u32 [D], game u32 [D], player u8 [D] (0 P1, 1 P2)
//   tokens i16 [T, COLUMNS], offsets i64 [D + 1]: decision i's tokens are
//     tokens[offsets[i]:offsets[i + 1]], its view since it last chose
//   request i16 [D, REQUEST_FIELDS]
//   state i16 [D, STATE_FIELDS], with `state` only
//   choices u8 [D, MAX_CHOICES] (libpkmn's choice bytes: type in bits 0-1,
//     data above; 0 past n), n u8 [D], at least 2
//   end_battle u32 [E], end_game u32 [E], end_result i8 [E] (P1's: 1 win,
//     -1 loss, 0 tie): the games that ended in this call; end_capped u8 [E],
//     1 for a tie by max_turns
//   end_team i32 [E, 2], end_prob f64 [E, 2]: each side's pool team and
//     the probability it was drawn with (-1 and 0 without a pool)
// `picks` is one index per decision, into its choices.
//
// `set_team_weights(w, role=-1)` sets the distribution new games draw teams
// from (f64 [pool], normalized; uniform until set), role 0's, 1's or both's
// (src/env.hpp: per seat, or per seat swapped in the odd battles after
// `set_alternate(True)`); `team_probs` (role 0's; `team_probs_of(role)`),
// `team_counts` (draws per team, both sides, since construction) and
// `team_names` are properties.
//
// `game_data()` gives libpkmn's tables for heuristic agents
// (ataraxos/eval/agents.py), as a dict of arrays indexed by the enums:
// move_bp, move_type, move_accuracy [MOVES] (None's row 0); species_types
// [SPECIES, 2], species_stats [SPECIES, 5] (HP, Atk, Def, Spe, Spc);
// type_chart [TYPES, TYPES] (effectiveness x 10); and the constant
// ARG_SWITCH.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>

#include "env.hpp"

namespace py = pybind11;
using ataraxos::SelfPlay;

// The pool's team names, kept beside the env.
struct Env : SelfPlay {
  using SelfPlay::SelfPlay;
  std::vector<std::string> names;
};

namespace {

template <typename T>
py::array_t<T> array(std::vector<py::ssize_t> shape) {
  return py::array_t<T>(shape);
}

py::dict observe(const SelfPlay& env) {
  const auto& shards = env.shards();
  size_t D = 0, T = 0, E = 0;
  std::vector<size_t> d0(shards.size()), t0(shards.size()), e0(shards.size());
  for (size_t s = 0; s < shards.size(); s++) {
    d0[s] = D, t0[s] = T, e0[s] = E;
    D += shards[s].decisions.size();
    T += shards[s].tokens.size();
    E += shards[s].ends.size();
  }
  auto battle = array<uint32_t>({py::ssize_t(D)});
  auto game = array<uint32_t>({py::ssize_t(D)});
  auto player = array<uint8_t>({py::ssize_t(D)});
  auto tokens = array<int16_t>({py::ssize_t(T), py::ssize_t(ataraxos::COLUMNS)});
  auto offsets = array<int64_t>({py::ssize_t(D + 1)});
  auto request = array<int16_t>({py::ssize_t(D), py::ssize_t(ataraxos::REQUEST_FIELDS)});
  auto state = array<int16_t>({py::ssize_t(env.state() ? D : 0), py::ssize_t(ataraxos::STATE_FIELDS)});
  auto choices = array<uint8_t>({py::ssize_t(D), py::ssize_t(ataraxos::MAX_CHOICES)});
  auto n = array<uint8_t>({py::ssize_t(D)});
  auto end_battle = array<uint32_t>({py::ssize_t(E)});
  auto end_game = array<uint32_t>({py::ssize_t(E)});
  auto end_result = array<int8_t>({py::ssize_t(E)});
  auto end_capped = array<uint8_t>({py::ssize_t(E)});
  auto end_team = array<int32_t>({py::ssize_t(E), 2});
  auto end_prob = array<double>({py::ssize_t(E), 2});

  uint32_t* pb = battle.mutable_data();
  uint32_t* pg = game.mutable_data();
  uint8_t* pp = player.mutable_data();
  int16_t* pt = tokens.mutable_data();
  int64_t* po = offsets.mutable_data();
  int16_t* pr = request.mutable_data();
  int16_t* ps = state.mutable_data();
  uint8_t* pc = choices.mutable_data();
  uint8_t* pn = n.mutable_data();
  uint32_t* eb = end_battle.mutable_data();
  uint32_t* eg = end_game.mutable_data();
  int8_t* er = end_result.mutable_data();
  uint8_t* ec = end_capped.mutable_data();
  int32_t* et = end_team.mutable_data();
  double* ep = end_prob.mutable_data();
  po[D] = int64_t(T);
  {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic) num_threads(env.threads())
    for (size_t s = 0; s < shards.size(); s++) {
      const ataraxos::Shard& sh = shards[s];
      int64_t at = int64_t(t0[s]);
      for (size_t k = 0; k < sh.decisions.size(); k++) {
        const ataraxos::Decision& x = sh.decisions[k];
        const size_t i = d0[s] + k;
        pb[i] = x.battle;
        pg[i] = x.game;
        pp[i] = uint8_t(x.player);
        po[i] = at;
        at += sh.lengths[k];
        pn[i] = x.n;
        for (size_t c = 0; c < ataraxos::MAX_CHOICES; c++) {
          pc[i * ataraxos::MAX_CHOICES + c] = c < x.n ? x.options[c] : 0;
        }
      }
      std::memcpy(pt + t0[s] * ataraxos::COLUMNS, sh.tokens.data(), sh.tokens.size() * sizeof(ataraxos::Token));
      std::memcpy(pr + d0[s] * ataraxos::REQUEST_FIELDS, sh.requests.data(), sh.requests.size() * 2);
      if (env.state()) std::memcpy(ps + d0[s] * ataraxos::STATE_FIELDS, sh.states.data(), sh.states.size() * 2);
      for (size_t k = 0; k < sh.ends.size(); k++) {
        const ataraxos::End& e = sh.ends[k];
        eb[e0[s] + k] = e.battle;
        eg[e0[s] + k] = e.game;
        er[e0[s] + k] = int8_t(e.outcome);
        ec[e0[s] + k] = e.capped;
        for (size_t side = 0; side < 2; side++) {
          et[(e0[s] + k) * 2 + side] = e.teams.index[side];
          ep[(e0[s] + k) * 2 + side] = e.teams.prob[side];
        }
      }
    }
  }
  py::dict out;
  out["battle"] = battle;
  out["game"] = game;
  out["player"] = player;
  out["tokens"] = tokens;
  out["offsets"] = offsets;
  out["request"] = request;
  if (env.state()) out["state"] = state;
  out["choices"] = choices;
  out["n"] = n;
  out["end_battle"] = end_battle;
  out["end_game"] = end_game;
  out["end_result"] = end_result;
  out["end_capped"] = end_capped;
  out["end_team"] = end_team;
  out["end_prob"] = end_prob;
  return out;
}

py::dict game_data() {
  namespace D = PKMN::Data;
  constexpr size_t MOVES = ataraxos::MOVE_COUNT, SPECIES = ataraxos::SPECIES_COUNT, TYPES = 15;
  auto move_bp = array<int16_t>({py::ssize_t(MOVES)});
  auto move_type = array<int16_t>({py::ssize_t(MOVES)});
  auto move_accuracy = array<int16_t>({py::ssize_t(MOVES)});
  for (uint16_t m = 0; m < MOVES; m++) {
    const bool none = m == 0;
    // Struggle is past MOVE_DATA's table.
    const bool table = !none && m < MOVES - 1;
    move_bp.mutable_data()[m] = table ? D::move_data(m).bp : none ? 0 : 50;
    move_type.mutable_data()[m] = table ? int16_t(D::move_data(m).type) : 0;
    // MOVE_DATA's accuracy is out of 255.
    move_accuracy.mutable_data()[m] = table ? int16_t(D::move_data(m).accuracy * 100 / 255) : 100;
  }
  auto types = array<int16_t>({py::ssize_t(SPECIES), 2});
  auto stats = array<int16_t>({py::ssize_t(SPECIES), 5});
  std::fill(types.mutable_data(), types.mutable_data() + types.size(), 0);
  std::fill(stats.mutable_data(), stats.mutable_data() + stats.size(), 0);
  for (uint16_t s = 1; s < SPECIES; s++) {
    const auto d = D::get_species_data(D::Species(s));
    const auto& b = d.base_stats;
    const int16_t row[5] = {b.hp, b.atk, b.def, b.spe, b.spc};
    for (size_t i = 0; i < 2; i++) types.mutable_data()[s * 2 + i] = int16_t(d.types[i]);
    for (size_t i = 0; i < 5; i++) stats.mutable_data()[s * 5 + i] = row[i];
  }
  auto chart = array<int16_t>({py::ssize_t(TYPES), py::ssize_t(TYPES)});
  for (uint8_t a = 0; a < TYPES; a++) {
    for (uint8_t d = 0; d < TYPES; d++) {
      chart.mutable_data()[a * TYPES + d] = int16_t(int(D::TYPE_CHART[a][d]) * 10 / 2);
    }
  }
  py::dict out;
  out["move_bp"] = move_bp;
  out["move_type"] = move_type;
  out["move_accuracy"] = move_accuracy;
  out["species_types"] = types;
  out["species_stats"] = stats;
  out["type_chart"] = chart;
  out["ARG_SWITCH"] = int(ataraxos::ArgType::Switch);
  return out;
}

} // namespace

PYBIND11_MODULE(ataraxos_env, m) {
  m.doc() = "Lockstep Gen 1 self-play over libpkmn (src/env.hpp).";
  m.attr("COLUMNS") = int(ataraxos::COLUMNS);
  m.attr("REQUEST_FIELDS") = int(ataraxos::REQUEST_FIELDS);
  m.attr("STATE_FIELDS") = int(ataraxos::STATE_FIELDS);
  m.attr("MAX_CHOICES") = int(ataraxos::MAX_CHOICES);
  m.def("game_data", &game_data);

  py::class_<Env>(m, "SelfPlay")
      .def(py::init([](size_t battles, uint64_t seed, std::vector<std::string> teams,
                       uint16_t max_turns, size_t shards, int threads, bool state) {
             size_t errors = 0;
             std::vector<std::string> names;
             std::vector<ataraxos::Team> pool = ataraxos::load_teams(teams, errors, &names);
             if (!teams.empty() && pool.empty()) throw std::invalid_argument("no team parses");
             if (errors) py::print("ataraxos_env:", errors, "teams don't parse, skipped");
             Env* env = new Env(battles, seed, shards, threads, std::move(pool), max_turns, state);
             env->names = std::move(names);
             return env;
           }),
           py::arg("battles"), py::arg("seed") = 1, py::arg("teams") = std::vector<std::string>{},
           py::arg("max_turns") = 0, py::arg("shards") = 64, py::arg("threads") = 8,
           py::arg("state") = false)
      .def("reset",
           [](Env& env) {
             {
               py::gil_scoped_release release;
               env.reset();
             }
             return observe(env);
           })
      .def("step",
           [](Env& env, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> picks) {
             if (picks.ndim() != 1 || size_t(picks.size()) != env.decisions()) {
               throw std::invalid_argument("step: one pick per decision");
             }
             const uint8_t* p = picks.data();
             size_t i = 0;
             for (const ataraxos::Shard& sh : env.shards()) {
               for (const ataraxos::Decision& x : sh.decisions) {
                 if (p[i++] >= x.n) throw std::invalid_argument("step: a pick past its choices");
               }
             }
             {
               py::gil_scoped_release release;
               env.step({p, size_t(picks.size())});
             }
             return observe(env);
           },
           py::arg("picks"))
      .def("set_team_weights",
           [](Env& env, py::array_t<double, py::array::c_style | py::array::forcecast> w, int role) {
             if (w.ndim() != 1) throw std::invalid_argument("set_team_weights: one weight per team");
             env.set_team_weights(std::vector<double>(w.data(), w.data() + w.size()), role);
           },
           py::arg("w"), py::arg("role") = -1)
      .def("set_alternate", &SelfPlay::set_alternate, py::arg("on"))
      .def("team_probs_of",
           [](const Env& env, int role) {
             if (role < 0 || role > 1) throw std::invalid_argument("role is 0 or 1");
             const auto& p = env.team_probs(role);
             return py::array_t<double>(py::ssize_t(p.size()), p.data());
           },
           py::arg("role"))
      .def_property_readonly("team_probs",
                             [](const Env& env) {
                               const auto& p = env.team_probs();
                               return py::array_t<double>(py::ssize_t(p.size()), p.data());
                             })
      .def_property_readonly("team_counts",
                             [](const Env& env) {
                               auto out = array<int64_t>({py::ssize_t(env.pool_size())});
                               int64_t* o = out.mutable_data();
                               std::fill(o, o + env.pool_size(), 0);
                               for (const ataraxos::Shard& sh : env.shards()) {
                                 for (size_t i = 0; i < sh.team_counts.size(); i++) o[i] += sh.team_counts[i];
                               }
                               return out;
                             })
      .def_property_readonly("team_names", [](const Env& env) { return env.names; })
      .def_property_readonly("battles", &SelfPlay::size)
      .def_property_readonly("decisions", &SelfPlay::decisions)
      .def_property_readonly("stats", [](const Env& env) {
        uint64_t forced = 0, games = 0, capped = 0;
        for (const ataraxos::Shard& sh : env.shards()) {
          forced += sh.forced, games += sh.games, capped += sh.capped;
        }
        py::dict d;
        d["forced"] = forced; // one-choice requests answered without the net
        d["errors"] = 0;      // picks Showdown would reject: none in Gen 1
        d["games"] = games;
        d["capped"] = capped; // games ended by max_turns
        return d;
      });
}
