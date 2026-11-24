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

#include "client.hpp"
#include "env.hpp"
#include "pi4.hpp"
#include "pi_env.hpp"
#include "rollouts.hpp"

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

template <class Source> py::dict observe(const Source& env) {
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

// Rollouts' decisions: observe's dict (`battle` the rollout) with leaf u8
// [D], 1 where only the value is read, and root u32 [D].
py::dict observe_rollouts(const ataraxos::Rollouts& r) {
  py::dict out = observe(r);
  size_t D = 0;
  for (const auto& sh : r.shards()) D += sh.decisions.size();
  auto leaf = array<uint8_t>({py::ssize_t(D)});
  auto root = array<uint32_t>({py::ssize_t(D)});
  size_t i = 0;
  for (size_t s = 0; s < r.shards().size(); s++) {
    for (size_t k = 0; k < r.shards()[s].decisions.size(); k++, i++) {
      leaf.mutable_data()[i] = r.leaves()[s][k];
      root.mutable_data()[i] = r.rollouts()[r.shards()[s].decisions[k].battle].root;
    }
  }
  out["leaf"] = leaf;
  out["root"] = root;
  return out;
}

// Prefill summaries of worlds `ws` (World's layout): tokens i16 [T,
// COLUMNS], offsets i64 [S + 1] over summaries, request i16 [S,
// REQUEST_FIELDS], summaries i64 [W + 1] (world w's are summaries[w]:
// summaries[w + 1]), t0 i64 [W], root u32 [W], source u8 [W] (0 a team, 1
// sets, 2 random).
py::dict layout_worlds(const std::vector<const ataraxos::World*>& ws) {
  size_t T = 0, S = 0;
  for (const auto* w : ws) T += w->tokens.size(), S += w->ends.size();
  auto tokens = array<int16_t>({py::ssize_t(T), py::ssize_t(ataraxos::COLUMNS)});
  auto offsets = array<int64_t>({py::ssize_t(S + 1)});
  auto request = array<int16_t>({py::ssize_t(S), py::ssize_t(ataraxos::REQUEST_FIELDS)});
  auto summaries = array<int64_t>({py::ssize_t(ws.size() + 1)});
  auto t0 = array<int64_t>({py::ssize_t(ws.size())});
  auto root = array<uint32_t>({py::ssize_t(ws.size())});
  auto source = array<uint8_t>({py::ssize_t(ws.size())});
  size_t t = 0, s = 0;
  for (size_t i = 0; i < ws.size(); i++) {
    const auto& w = *ws[i];
    std::memcpy(tokens.mutable_data() + t * ataraxos::COLUMNS, w.tokens.data(), w.tokens.size() * sizeof(ataraxos::Token));
    summaries.mutable_data()[i] = int64_t(s);
    std::memcpy(request.mutable_data() + s * ataraxos::REQUEST_FIELDS, w.requests.data(), w.requests.size() * 2);
    for (const uint32_t e : w.ends) offsets.mutable_data()[++s] = int64_t(t + e);
    t += w.tokens.size();
    t0.mutable_data()[i] = w.t0;
    root.mutable_data()[i] = w.root;
    source.mutable_data()[i] = uint8_t(w.source);
  }
  offsets.mutable_data()[0] = 0;
  summaries.mutable_data()[ws.size()] = int64_t(s);
  py::dict out;
  out["tokens"] = tokens;
  out["offsets"] = offsets;
  out["request"] = request;
  out["summaries"] = summaries;
  out["t0"] = t0;
  out["root"] = root;
  out["source"] = source;
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
    move_bp.mutable_data()[m] = none ? 0 : D::move_data(m).bp;
    move_type.mutable_data()[m] = none ? 0 : int16_t(D::move_data(m).type);
    // MOVE_DATA's accuracy is out of 255.
    move_accuracy.mutable_data()[m] = none ? 0 : int16_t(D::move_data(m).accuracy * 100 / 255);
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
      .def("set_keep_after", &SelfPlay::set_keep_after, py::arg("on"))
      .def("set_light", &SelfPlay::set_light, py::arg("on"))
      .def_property_readonly("stats",
                             [](const Env& env) {
                               uint64_t forced = 0, games = 0, capped = 0;
                               for (const ataraxos::Shard& sh : env.shards()) {
                                 forced += sh.forced;
                                 games += sh.games;
                                 capped += sh.capped;
                               }
                               py::dict out;
                               out["forced"] = forced;
                               out["games"] = games;
                               out["capped"] = capped;
                               return out;
                             })
      // pi4's input (src/pi4.hpp, docs/pi4.md) for the last gather. With
      // `rows` (training) one row per battle with a decision, each player's
      // choices filled where it decides; else one row per decision, only the
      // deciding player's (evals, an agent per seat). state i16 [R, 2,
      // PI4_FIELDS] (from P1's side, then P2's), request i16 [R, 2,
      // REQUEST_FIELDS] (each player's current request), choices u8 [R, 2,
      // 9] and n u8 [R, 2] (n 1 where a side doesn't decide in the row),
      // battle and game u32 [R], player u8 [R] (the first deciding player),
      // dec i64 [R, 2] (the decision each side's pick answers, -1 none).
      // Needs `state` (light or not).
      .def("pi4_obs",
           [](Env& env, bool rows) {
             if (!env.tracking()) throw std::logic_error("pi4_obs needs state=True");
             const auto& shards = env.shards();
             const size_t S = shards.size();
             std::vector<size_t> r0(S), dbase(S);
             size_t R = 0, D = 0;
             for (size_t s = 0; s < S; s++) {
               r0[s] = R;
               dbase[s] = D;
               uint32_t last = UINT32_MAX;
               for (const ataraxos::Decision& x : shards[s].decisions) {
                 if (!rows || x.battle != last) R++;
                 last = x.battle;
               }
               D += shards[s].decisions.size();
             }
             constexpr size_t C = ataraxos::MAX_CHOICES, F = ataraxos::PI4_FIELDS, Q = ataraxos::REQUEST_FIELDS;
             auto state = array<int16_t>({py::ssize_t(R), 2, py::ssize_t(F)});
             auto request = array<int16_t>({py::ssize_t(R), 2, py::ssize_t(Q)});
             auto choices = array<uint8_t>({py::ssize_t(R), 2, py::ssize_t(C)});
             auto n = array<uint8_t>({py::ssize_t(R), 2});
             auto battle = array<uint32_t>({py::ssize_t(R)});
             auto game = array<uint32_t>({py::ssize_t(R)});
             auto player = array<uint8_t>({py::ssize_t(R)});
             auto dec = array<int64_t>({py::ssize_t(R), 2});
             int16_t* ps = state.mutable_data();
             int16_t* pq = request.mutable_data();
             uint8_t* pc = choices.mutable_data();
             uint8_t* pn = n.mutable_data();
             uint32_t* pb = battle.mutable_data();
             uint32_t* pg = game.mutable_data();
             uint8_t* pp = player.mutable_data();
             int64_t* pd = dec.mutable_data();
             std::fill(pc, pc + R * 2 * C, uint8_t(0));
             std::fill(pn, pn + R * 2, uint8_t(1));
             std::fill(pd, pd + R * 2, int64_t(-1));
             {
               py::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic) num_threads(env.threads())
               for (size_t s = 0; s < S; s++) {
                 const ataraxos::Shard& sh = shards[s];
                 size_t r = r0[s];
                 uint32_t last = UINT32_MAX;
                 for (size_t k = 0; k < sh.decisions.size(); k++) {
                   const ataraxos::Decision& x = sh.decisions[k];
                   if (k > 0 && (!rows || x.battle != last)) r++;
                   if (k == 0 || !rows || x.battle != last) {
                     const ataraxos::BattleData& d = env.battle(x.battle);
                     pb[r] = x.battle;
                     pg[r] = x.game;
                     pp[r] = x.player;
                     ataraxos::encode_pi4(d, ps + r * 2 * F);
                     for (uint8_t q = 0; q < 2; q++) ataraxos::encode(d.request[q], pq + (r * 2 + q) * Q);
                   }
                   last = x.battle;
                   for (size_t c = 0; c < x.n; c++) pc[(r * 2 + x.player) * C + c] = x.options[c];
                   pn[r * 2 + x.player] = x.n;
                   pd[r * 2 + x.player] = int64_t(dbase[s] + k);
                 }
               }
             }
             py::dict out;
             out["state"] = state;
             out["request"] = request;
             out["choices"] = choices;
             out["n"] = n;
             out["battle"] = battle;
             out["game"] = game;
             out["player"] = player;
             out["dec"] = dec;
             return out;
           },
           py::arg("rows") = true)
      // The last gather's decisions as the perfect-info net's input
      // (src/pi_env.hpp's encode on the battle and its chance durations), for
      // a pi net against a Showdown-view net (ataraxos/eval/pi_h2h.py): mon,
      // hp, active as ataraxos_pi's; policy i16 [D, 2, 9] and n u8 [D, 2]
      // with only the deciding player's side filled (its choices in this
      // env's order; the other side n 1).
      .def("pi_obs",
           [](Env& env) {
             const auto D = py::ssize_t(env.decisions()), C = py::ssize_t(ataraxos::MAX_CHOICES);
             auto mon = array<int16_t>({D, 2, 6, py::ssize_t(ataraxos::pi::MON_FIELDS)});
             auto hp = array<float>({D, 2, 6});
             auto active = array<float>({D, 2, py::ssize_t(ataraxos::pi::ACTIVE_DIM)});
             auto policy = array<int16_t>({D, 2, C});
             auto n = array<uint8_t>({D, 2});
             std::fill(policy.mutable_data(), policy.mutable_data() + policy.size(), int16_t(0));
             std::fill(n.mutable_data(), n.mutable_data() + n.size(), uint8_t(1));
             ataraxos::pi::Features f;
             size_t i = 0;
             for (const ataraxos::Shard& sh : env.shards()) {
               for (const ataraxos::Decision& x : sh.decisions) {
                 auto& d = const_cast<ataraxos::BattleData&>(env.battle(x.battle));
                 const PKMN::Battle& b = d.state();
                 ataraxos::pi::encode(b, PKMN::view(*pkmn_gen1_battle_options_chance_durations(&d.options)), f);
                 std::copy(f.mon.begin(), f.mon.end(), mon.mutable_data() + i * f.mon.size());
                 std::copy(f.hp.begin(), f.hp.end(), hp.mutable_data() + i * f.hp.size());
                 std::copy(f.active.begin(), f.active.end(), active.mutable_data() + i * f.active.size());
                 const std::span<const pkmn_choice> legal{d.legal[x.player].data(), d.n_legal[x.player]};
                 int16_t* pol = policy.mutable_data() + (i * 2 + x.player) * C;
                 for (size_t k = 0; k < x.n; k++) {
                   const pkmn_choice c = ataraxos::engine_choice(x.options[k], legal);
                   pol[k] = int16_t(Encode::Battle::Policy::get_index(b.sides[x.player], c));
                 }
                 n.mutable_data()[i * 2 + x.player] = x.n;
                 i++;
               }
             }
             py::dict out;
             out["mon"] = mon;
             out["hp"] = hp;
             out["active"] = active;
             out["policy"] = policy;
             out["n"] = n;
             return out;
           })
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

  // src/rollouts.hpp. Rollouts(teams, keep, shards=64, threads=8,
  // state=True, ratio=0.1): the sampler's pool (Showdown export files) and
  // its rank-decay prior; `keep` the foe's prefill summaries kept.
  py::class_<ataraxos::Rollouts>(m, "Rollouts")
      .def(py::init([](std::vector<std::string> teams, size_t keep, size_t shards, int threads, bool state,
                       double ratio) {
             size_t errors = 0;
             std::vector<ataraxos::Team> pool = ataraxos::load_teams(teams, errors);
             if (pool.empty()) throw std::invalid_argument("no team parses");
             auto* r = new ataraxos::Rollouts(std::move(pool), keep, shards, threads, state);
             r->pool() = ataraxos::Pool(r->pool().teams, ratio);
             return r;
           }),
           py::arg("teams"), py::arg("keep"), py::arg("shards") = 64, py::arg("threads") = 8,
           py::arg("state") = true, py::arg("ratio") = 0.1)
      .def("clear", &ataraxos::Rollouts::clear)
      .def("set_prior",
           [](ataraxos::Rollouts& r, py::array_t<double, py::array::c_style | py::array::forcecast> w) {
             if (size_t(w.size()) != r.pool().teams.size()) throw std::invalid_argument("set_prior: one weight per team");
             r.pool().set_prior({w.data(), size_t(w.size())});
           },
           py::arg("w"))
      // a search at a SelfPlay battle's decision (the env with state and
      // set_keep_after(True)); returns the root. `truth`: every world has
      // the foe's true sets (tests)
      .def("add_selfplay",
           [](ataraxos::Rollouts& r, const Env& env, size_t b, uint8_t p, uint32_t worlds, uint32_t reps,
              uint16_t depth, uint64_t seed, bool truth) {
             const ataraxos::BattleData& d = env.battle(b);
             if (!d.track || !d.keep_after) throw std::invalid_argument("add_selfplay: the env needs state and keep_after");
             if (!d.needs_choice(p)) throw std::invalid_argument("add_selfplay: no decision there");
             py::gil_scoped_release release;
             const ataraxos::Guess g = ataraxos::true_guess(d.bk[p], d.state().sides[1 - p]);
             return r.add(p, d.bk[p], d.history[p], d.after[p], d.request[p], {d.option[p].data(), d.n_option[p]},
                          d.binds, worlds, reps, depth, seed, truth ? &g : nullptr);
           },
           py::arg("env"), py::arg("battle"), py::arg("player"), py::arg("worlds"), py::arg("reps"),
           py::arg("depth"), py::arg("seed"), py::arg("truth") = false)
      // a search at the ladder Client's request now (ataraxos_client's
      // Client, which keeps no Binds: Binds{}); returns the root
      .def("add_client",
           [](ataraxos::Rollouts& r, const ataraxos::Client& c, uint32_t worlds, uint32_t reps, uint16_t depth,
              uint64_t seed) {
             if (c.req.kind == ataraxos::RequestData::Wait || c.n == 0) throw std::invalid_argument("add_client: no decision");
             py::gil_scoped_release release;
             return r.add(c.viewer(), c.bookkeeping(), c.history(), c.after(), c.req, {c.options.data(), c.n},
                          ataraxos::Binds{}, worlds, reps, depth, seed);
           },
           py::arg("client"), py::arg("worlds"), py::arg("reps"), py::arg("depth"), py::arg("seed"))
      // each world's foe prefill (layout_worlds) and each rollout's world
      // u32 [N], action u8 [N], differs u8 [N]
      .def("layout",
           [](const ataraxos::Rollouts& r) {
             std::vector<const ataraxos::World*> ws;
             for (const auto& w : r.worlds()) ws.push_back(&w);
             py::dict out = layout_worlds(ws);
             const auto& rs = r.rollouts();
             auto world = array<uint32_t>({py::ssize_t(rs.size())});
             auto action = array<uint8_t>({py::ssize_t(rs.size())});
             auto differs = array<uint8_t>({py::ssize_t(rs.size())});
             for (size_t i = 0; i < rs.size(); i++) {
               world.mutable_data()[i] = rs[i].world;
               action.mutable_data()[i] = rs[i].action;
               differs.mutable_data()[i] = rs[i].differs;
             }
             out["world"] = world;
             out["action"] = action;
             out["differs"] = differs;
             return out;
           })
      // a joint root at a SelfPlay battle's decision (add_selfplay's env),
      // `a` of our types and `b` of the foe's; returns the joint root.
      // `truth`: the foe's types all take its true sets (tests)
      .def("add_joint_selfplay",
           [](ataraxos::Rollouts& r, const Env& env, size_t b, uint8_t p, uint8_t types_a, uint8_t types_b,
              uint16_t depth, uint64_t seed, bool truth) {
             const ataraxos::BattleData& d = env.battle(b);
             if (!d.track || !d.keep_after) throw std::invalid_argument("add_joint_selfplay: the env needs state and keep_after");
             if (!d.needs_choice(p)) throw std::invalid_argument("add_joint_selfplay: no decision there");
             if (!types_a || !types_b) throw std::invalid_argument("add_joint_selfplay: a type a side at least");
             py::gil_scoped_release release;
             const ataraxos::Guess g = ataraxos::true_guess(d.bk[p], d.state().sides[1 - p]);
             return r.add_joint(p, d.bk[p], d.history[p], d.after[p], d.request[p],
                                {d.option[p].data(), d.n_option[p]}, d.binds, types_a, types_b, depth, seed,
                                truth ? &g : nullptr);
           },
           py::arg("env"), py::arg("battle"), py::arg("player"), py::arg("a"), py::arg("b"), py::arg("depth"),
           py::arg("seed"), py::arg("truth") = false)
      // a joint root at the ladder Client's request now (Binds{})
      .def("add_joint_client",
           [](ataraxos::Rollouts& r, const ataraxos::Client& c, uint8_t types_a, uint8_t types_b, uint16_t depth,
              uint64_t seed) {
             if (c.req.kind == ataraxos::RequestData::Wait || c.n == 0) throw std::invalid_argument("add_joint_client: no decision");
             if (!types_a || !types_b) throw std::invalid_argument("add_joint_client: a type a side at least");
             py::gil_scoped_release release;
             return r.add_joint(c.viewer(), c.bookkeeping(), c.history(), c.after(), c.req, {c.options.data(), c.n},
                                ataraxos::Binds{}, types_a, types_b, depth, seed);
           },
           py::arg("client"), py::arg("a"), py::arg("b"), py::arg("depth"), py::arg("seed"))
      .def("clear_rollouts", &ataraxos::Rollouts::clear_rollouts)
      // this round's joint rollouts: cells i64 [K, 6], each (root, a, b, i,
      // j, count)
      .def("schedule",
           [](ataraxos::Rollouts& r, py::array_t<int64_t, py::array::c_style | py::array::forcecast> cells) {
             if (cells.ndim() != 2 || cells.shape(1) != 6) throw std::invalid_argument("schedule: cells [K, 6]");
             const int64_t* c = cells.data();
             for (py::ssize_t k = 0; k < cells.shape(0); k++, c += 6) {
               const auto& js = r.joints();
               if (c[0] < 0 || size_t(c[0]) >= js.size()) throw std::invalid_argument("schedule: no such root");
               const auto& J = js[size_t(c[0])];
               if (c[1] < 0 || c[1] >= J.a || c[2] < 0 || c[2] >= J.b || c[5] < 0) throw std::invalid_argument("schedule: bad cell");
               if (c[3] < 0 || c[3] >= r.types()[J.first + c[1]].n || c[4] < 0 || c[4] >= r.types()[J.first + J.a + c[2]].n) {
                 throw std::invalid_argument("schedule: bad choice");
               }
               r.schedule(size_t(c[0]), uint8_t(c[1]), uint8_t(c[2]), uint8_t(c[3]), uint8_t(c[4]), uint32_t(c[5]));
             }
           },
           py::arg("cells"))
      // the joint roots' types, in order (root r's are first[r]:first[r] +
      // a[r] ours, A0 first, then b[r] the foe's): layout()'s prefill arrays
      // over types (each type's summaries before its root decision; none
      // for A0) and per type side u8, real u8, n u8, choices u8 [Y,
      // MAX_CHOICES], decides u8, source u8; the root decisions, one row a
      // type (run where decides): dec_tokens i16, dec_offsets i64 [Y + 1],
      // dec_request i16 [Y, REQUEST_FIELDS], dec_state i16 [Y,
      // STATE_FIELDS] (with state); per joint root first u32, a u8, b u8,
      // player u8
      .def("joint_layout",
           [](const ataraxos::Rollouts& r) {
             const auto& ts = r.types();
             const size_t Y = ts.size();
             std::vector<const ataraxos::World*> ws;
             size_t T = 0;
             for (const auto& t : ts) ws.push_back(&t.prefill), T += t.chunk.size();
             py::dict out = layout_worlds(ws);
             auto side = array<uint8_t>({py::ssize_t(Y)});
             auto real = array<uint8_t>({py::ssize_t(Y)});
             auto n = array<uint8_t>({py::ssize_t(Y)});
             auto decides = array<uint8_t>({py::ssize_t(Y)});
             auto choices = array<uint8_t>({py::ssize_t(Y), py::ssize_t(ataraxos::MAX_CHOICES)});
             auto tokens = array<int16_t>({py::ssize_t(T), py::ssize_t(ataraxos::COLUMNS)});
             auto offsets = array<int64_t>({py::ssize_t(Y + 1)});
             auto request = array<int16_t>({py::ssize_t(Y), py::ssize_t(ataraxos::REQUEST_FIELDS)});
             auto state = array<int16_t>({py::ssize_t(r.state() ? Y : 0), py::ssize_t(ataraxos::STATE_FIELDS)});
             size_t at = 0;
             for (size_t y = 0; y < Y; y++) {
               const auto& t = ts[y];
               side.mutable_data()[y] = t.side;
               real.mutable_data()[y] = t.real;
               n.mutable_data()[y] = t.n;
               decides.mutable_data()[y] = t.decides;
               for (size_t c = 0; c < ataraxos::MAX_CHOICES; c++) {
                 choices.mutable_data()[y * ataraxos::MAX_CHOICES + c] = c < t.n ? t.options[c] : 0;
               }
               offsets.mutable_data()[y] = int64_t(at);
               std::memcpy(tokens.mutable_data() + at * ataraxos::COLUMNS, t.chunk.data(), t.chunk.size() * sizeof(ataraxos::Token));
               at += t.chunk.size();
               ataraxos::encode(t.request, request.mutable_data() + y * ataraxos::REQUEST_FIELDS);
               if (r.state()) ataraxos::encode_state(t.bk, state.mutable_data() + y * ataraxos::STATE_FIELDS);
             }
             offsets.mutable_data()[Y] = int64_t(at);
             const auto& js = r.joints();
             auto first = array<uint32_t>({py::ssize_t(js.size())});
             auto ja = array<uint8_t>({py::ssize_t(js.size())});
             auto jb = array<uint8_t>({py::ssize_t(js.size())});
             auto player = array<uint8_t>({py::ssize_t(js.size())});
             for (size_t i = 0; i < js.size(); i++) {
               first.mutable_data()[i] = js[i].first;
               ja.mutable_data()[i] = js[i].a;
               jb.mutable_data()[i] = js[i].b;
               player.mutable_data()[i] = js[i].player;
             }
             out["side"] = side;
             out["real"] = real;
             out["n"] = n;
             out["choices"] = choices;
             out["decides"] = decides;
             out["dec_tokens"] = tokens;
             out["dec_offsets"] = offsets;
             out["dec_request"] = request;
             if (r.state()) out["dec_state"] = state;
             out["first"] = first;
             out["a"] = ja;
             out["b"] = jb;
             out["player"] = player;
             return out;
           })
      // each rollout's root u32, a u8, b u8, action u8, foe_action u8,
      // value f32 (ours), differs u8, done u8 (joint ones after a round)
      .def("joint_values",
           [](const ataraxos::Rollouts& r) {
             const auto& rs = r.rollouts();
             const py::ssize_t N = py::ssize_t(rs.size());
             auto root = array<uint32_t>({N});
             auto a = array<uint8_t>({N});
             auto b = array<uint8_t>({N});
             auto action = array<uint8_t>({N});
             auto foe_action = array<uint8_t>({N});
             auto value = array<float>({N});
             auto differs = array<uint8_t>({N});
             auto done = array<uint8_t>({N});
             for (size_t i = 0; i < rs.size(); i++) {
               const auto& x = rs[i];
               root.mutable_data()[i] = x.root;
               a.mutable_data()[i] = x.a;
               b.mutable_data()[i] = x.b;
               action.mutable_data()[i] = x.action;
               foe_action.mutable_data()[i] = x.foe_action;
               value.mutable_data()[i] = x.value;
               differs.mutable_data()[i] = x.differs;
               done.mutable_data()[i] = x.done;
             }
             py::dict out;
             out["root"] = root;
             out["a"] = a;
             out["b"] = b;
             out["action"] = action;
             out["foe_action"] = foe_action;
             out["value"] = value;
             out["differs"] = differs;
             out["done"] = done;
             return out;
           })
      .def("start",
           [](ataraxos::Rollouts& r, uint64_t seed) {
             {
               py::gil_scoped_release release;
               r.start(seed);
               r.release();
             }
             return observe_rollouts(r);
           },
           py::arg("seed"))
      .def("step",
           [](ataraxos::Rollouts& r, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> picks,
              py::array_t<float, py::array::c_style | py::array::forcecast> values) {
             size_t D = 0;
             for (const auto& sh : r.shards()) D += sh.decisions.size();
             if (size_t(picks.size()) != D || size_t(values.size()) != D) {
               throw std::invalid_argument("step: one pick and value per decision");
             }
             {
               py::gil_scoped_release release;
               r.step({picks.data(), D}, {values.data(), D});
             }
             return observe_rollouts(r);
           },
           py::arg("picks"), py::arg("values"))
      .def_property_readonly("done", &ataraxos::Rollouts::done)
      .def_property_readonly("rollouts", [](const ataraxos::Rollouts& r) { return r.rollouts().size(); })
      .def("q",
           [](const ataraxos::Rollouts& r, size_t root) {
             const std::vector<double> q = r.q(root);
             return py::array_t<double>(py::ssize_t(q.size()), q.data());
           },
           py::arg("root"));
}
