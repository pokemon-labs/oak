// Python binding for src/pi_env.hpp's perfect-info SelfPlay. Linked to oak's
// libpkmn_chance build (no log), as oak's search is, so it is its own module
// beside ataraxos_env (whose libpkmn has the log).
//
// SelfPlay(battles, seed=1, teams=[], max_turns=0, shards=64, threads=8)
// `reset()` and `step(picks)` return the next rows as a dict (R rows, one
// per battle):
//   battle u32 [R], game u32 [R]
//   n u8 [R, 2]: each side's choices (1 for a forced side, 0 never: a
//     battle with neither side choosing isn't a row)
//   choices u8 [R, 2, 9] (libpkmn's bytes), policy i16 [R, 2, 9] (oak's
//     flat Policy index, 0 past n)
//   mon i16 [R, 2, 6, 6], hp f32 [R, 2, 6], active f32 [R, 2, ACTIVE_DIM]
//     (src/pi_env.hpp's layout)
//   end_battle u32 [E], end_game u32 [E], end_result i8 [E] (P1's),
//   end_capped u8 [E], end_team i32 [E, 2], end_prob f64 [E, 2]
// `picks` is u8 [R, 2], each side's index into its choices.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>

#include "pi_env.hpp"

namespace py = pybind11;
namespace pi = ataraxos::pi;

struct PiEnv : pi::SelfPlay {
  using pi::SelfPlay::SelfPlay;
  std::vector<std::string> names;
};

namespace {

template <typename T> py::array_t<T> array(std::vector<py::ssize_t> shape) { return py::array_t<T>(shape); }

py::dict observe(const PiEnv& env) {
  const auto& shards = env.shards();
  size_t R = 0, E = 0;
  std::vector<size_t> r0(shards.size()), e0(shards.size());
  for (size_t s = 0; s < shards.size(); s++) {
    r0[s] = R, e0[s] = E;
    R += shards[s].rows.size();
    E += shards[s].ends.size();
  }
  const auto r = py::ssize_t(R), e = py::ssize_t(E), C = py::ssize_t(pi::MAX_CHOICES);
  auto battle = array<uint32_t>({r});
  auto game = array<uint32_t>({r});
  auto n = array<uint8_t>({r, 2});
  auto choices = array<uint8_t>({r, 2, C});
  auto policy = array<int16_t>({r, 2, C});
  auto mon = array<int16_t>({r, 2, 6, py::ssize_t(pi::MON_FIELDS)});
  auto hp = array<float>({r, 2, 6});
  auto active = array<float>({r, 2, py::ssize_t(pi::ACTIVE_DIM)});
  auto end_battle = array<uint32_t>({e});
  auto end_game = array<uint32_t>({e});
  auto end_result = array<int8_t>({e});
  auto end_capped = array<uint8_t>({e});
  auto end_team = array<int32_t>({e, 2});
  auto end_prob = array<double>({e, 2});
  uint32_t* pb = battle.mutable_data();
  uint32_t* pg = game.mutable_data();
  uint8_t* pn = n.mutable_data();
  uint8_t* pc = choices.mutable_data();
  int16_t* pp = policy.mutable_data();
  int16_t* pm = mon.mutable_data();
  float* ph = hp.mutable_data();
  float* pa = active.mutable_data();
  uint32_t* eb = end_battle.mutable_data();
  uint32_t* eg = end_game.mutable_data();
  int8_t* er = end_result.mutable_data();
  uint8_t* ec = end_capped.mutable_data();
  int32_t* et = end_team.mutable_data();
  double* ep = end_prob.mutable_data();
  {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic) num_threads(env.threads())
    for (size_t s = 0; s < shards.size(); s++) {
      const pi::Shard& sh = shards[s];
      for (size_t k = 0; k < sh.rows.size(); k++) {
        const pi::Row& x = sh.rows[k];
        const size_t i = r0[s] + k;
        pb[i] = x.battle;
        pg[i] = x.game;
        for (size_t side = 0; side < 2; side++) {
          pn[i * 2 + side] = x.n[side];
          for (size_t c = 0; c < pi::MAX_CHOICES; c++) {
            pc[(i * 2 + side) * pi::MAX_CHOICES + c] = x.choices[side][c];
            pp[(i * 2 + side) * pi::MAX_CHOICES + c] = x.policy[side][c];
          }
        }
        const pi::Features& f = x.features;
        std::memcpy(pm + i * f.mon.size(), f.mon.data(), sizeof(f.mon));
        std::memcpy(ph + i * f.hp.size(), f.hp.data(), sizeof(f.hp));
        std::memcpy(pa + i * f.active.size(), f.active.data(), sizeof(f.active));
      }
      for (size_t k = 0; k < sh.ends.size(); k++) {
        const pi::End& x = sh.ends[k];
        const size_t j = e0[s] + k;
        eb[j] = x.battle;
        eg[j] = x.game;
        er[j] = x.outcome;
        ec[j] = x.capped;
        for (size_t side = 0; side < 2; side++) {
          et[j * 2 + side] = x.teams.index[side];
          ep[j * 2 + side] = x.teams.prob[side];
        }
      }
    }
  }
  py::dict out;
  out["battle"] = battle;
  out["game"] = game;
  out["n"] = n;
  out["choices"] = choices;
  out["policy"] = policy;
  out["mon"] = mon;
  out["hp"] = hp;
  out["active"] = active;
  out["end_battle"] = end_battle;
  out["end_game"] = end_game;
  out["end_result"] = end_result;
  out["end_capped"] = end_capped;
  out["end_team"] = end_team;
  out["end_prob"] = end_prob;
  return out;
}

} // namespace

PYBIND11_MODULE(ataraxos_pi, m) {
  m.doc() = "Perfect-info lockstep Gen 1 self-play over libpkmn (src/pi_env.hpp).";
  m.attr("MAX_CHOICES") = int(pi::MAX_CHOICES);
  m.attr("MON_FIELDS") = int(pi::MON_FIELDS);
  m.attr("ACTIVE_DIM") = int(pi::ACTIVE_DIM);
  m.attr("POLICY_DIM") = int(pi::POLICY_DIM);
  m.attr("STATUS_DIM") = int(pi::STATUS_DIM);
  m.attr("MOVE_DIM") = int(pi::MOVE_DIM);
  m.attr("SPECIES_DIM") = 152; // 0 masked, 1-151

  py::class_<PiEnv>(m, "SelfPlay", py::module_local())
      .def(py::init([](size_t battles, uint64_t seed, std::vector<std::string> teams, uint16_t max_turns,
                       size_t shards, int threads) {
             size_t errors = 0;
             std::vector<std::string> names;
             std::vector<ataraxos::Team> pool = ataraxos::load_teams(teams, errors, &names);
             if (!teams.empty() && pool.empty()) throw std::invalid_argument("no team parses");
             if (errors) py::print("ataraxos_pi:", errors, "teams don't parse, skipped");
             PiEnv* env = new PiEnv(battles, seed, shards, threads, std::move(pool), max_turns);
             env->names = std::move(names);
             return env;
           }),
           py::arg("battles"), py::arg("seed") = 1, py::arg("teams") = std::vector<std::string>{},
           py::arg("max_turns") = 0, py::arg("shards") = 64, py::arg("threads") = 8)
      .def("reset",
           [](PiEnv& env) {
             {
               py::gil_scoped_release release;
               env.reset();
             }
             return observe(env);
           })
      .def("step",
           [](PiEnv& env, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> picks) {
             if (picks.ndim() != 2 || picks.shape(1) != 2 || size_t(picks.shape(0)) != env.rows())
               throw std::invalid_argument("picks is [R, 2]");
             std::span<const std::array<uint8_t, 2>> p(reinterpret_cast<const std::array<uint8_t, 2>*>(picks.data()),
                                                        size_t(picks.shape(0)));
             for (size_t i = 0, k = 0; i < env.shards().size(); i++) {
               for (const pi::Row& r : env.shards()[i].rows) {
                 for (size_t s = 0; s < 2; s++) {
                   if (p[k][s] >= std::max<uint8_t>(r.n[s], 1)) throw std::out_of_range("pick past the choices");
                 }
                 k++;
               }
             }
             {
               py::gil_scoped_release release;
               env.step(p);
             }
             return observe(env);
           })
      .def("set_team_weights", &PiEnv::set_team_weights)
      .def_property_readonly("team_probs", &PiEnv::team_probs)
      .def_property_readonly("team_names", [](const PiEnv& env) { return env.names; })
      .def_property_readonly("team_counts",
                             [](const PiEnv& env) {
                               std::vector<uint64_t> c(env.pool_size(), 0);
                               for (const auto& sh : env.shards()) {
                                 for (size_t i = 0; i < c.size(); i++) c[i] += sh.team_counts[i];
                               }
                               return c;
                             })
      .def_property_readonly("stats", [](const PiEnv& env) {
        uint64_t forced = 0, games = 0, capped = 0;
        for (const auto& sh : env.shards()) forced += sh.forced, games += sh.games, capped += sh.capped;
        py::dict d;
        d["forced"] = forced;
        d["games"] = games;
        d["capped"] = capped;
        return d;
      });
}
