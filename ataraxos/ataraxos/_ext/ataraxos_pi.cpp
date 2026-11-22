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
//
// `rollouts(battle u32 [K], picks u8 [K, 2], seeds u64 [K])` (search in
// training, depth 1; call between steps, battle ids of the current rows):
// mon, hp, active of each leaf (zeros where done), done u8 [K], result i8
// [K] (P1's, where done). `mmd(Q f32 [G, 9, 9], m1, m2 f32 [G, 9], n1, n2 u8
// [G], lam, eta, iters, tol, threads)`: (l1, l2 f32 [G, 9], iterations i32
// [G]), src/pi_env.hpp's mmd_batch. `rollouts_into(battle, picks, seeds,
// mon, hp, active, done, result)`: rollouts' output written into the given
// C-contiguous arrays (at least K rows; e.g. pinned torch buffers' numpy
// views), so the leaves go straight to the copy to the GPU.
// `solve(cells u8 [S, B, 2], wdl f32 [S * B, 3], n1, n2, m1, m2, shrink, lam,
// eta, iters, tol, threads)`: (l1, l2 [S, 9], swdl [S, 3], its [S]), the
// cell means, MMD and the W/D/L under the posteriors (solve_batch).
// `plan(prior f32 [S, 2, 9], n1, n2, budget, floor, seed, threads)`: the
// rollouts' cells u8 [S, budget, 2] (plan_cells).

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

template <typename T> using carray = py::array_t<T, py::array::c_style | py::array::forcecast>;

size_t check_rollouts(const PiEnv& env, const carray<uint32_t>& battle, const carray<uint8_t>& picks,
                      const carray<uint64_t>& seeds) {
  const size_t K = size_t(battle.size());
  if (battle.ndim() != 1 || picks.ndim() != 2 || picks.shape(1) != 2 || size_t(picks.shape(0)) != K ||
      seeds.ndim() != 1 || size_t(seeds.size()) != K)
    throw std::invalid_argument("battle [K], picks [K, 2], seeds [K]");
  const uint32_t* b = battle.data();
  const auto* p = reinterpret_cast<const std::array<uint8_t, 2>*>(picks.data());
  size_t battles = 0;
  for (const auto& sh : env.shards()) battles = std::max(battles, sh.end);
  for (size_t k = 0; k < K; k++) {
    if (b[k] >= battles) throw std::out_of_range("battle id");
    const pi::Battle& d = env.battle(b[k]);
    for (size_t s = 0; s < 2; s++) {
      if (p[k][s] >= std::max<uint8_t>(d.n[s], 1)) throw std::out_of_range("pick past the choices");
    }
  }
  return K;
}

void run_rollouts(const PiEnv& env, const carray<uint32_t>& battle, const carray<uint8_t>& picks,
                  const carray<uint64_t>& seeds, int16_t* mon, float* hp, float* active, uint8_t* done,
                  int8_t* result) {
  const size_t K = size_t(battle.size());
  const auto* p = reinterpret_cast<const std::array<uint8_t, 2>*>(picks.data());
  py::gil_scoped_release release;
  env.rollouts({battle.data(), K}, {p, K}, {seeds.data(), K}, mon, hp, active, done, result);
}

PYBIND11_MODULE(ataraxos_pi, m) {
  m.doc() = "Perfect-info lockstep Gen 1 self-play over libpkmn (src/pi_env.hpp).";
  m.attr("MAX_CHOICES") = int(pi::MAX_CHOICES);
  m.attr("MON_FIELDS") = int(pi::MON_FIELDS);
  m.attr("ACTIVE_DIM") = int(pi::ACTIVE_DIM);
  m.attr("POLICY_DIM") = int(pi::POLICY_DIM);
  m.attr("STATUS_DIM") = int(pi::STATUS_DIM);
  m.attr("MOVE_DIM") = int(pi::MOVE_DIM);
  m.attr("SPECIES_DIM") = 152; // 0 masked, 1-151

  m.def(
      "mmd",
      [](carray<float> Q, carray<float> m1, carray<float> m2, carray<uint8_t> n1, carray<uint8_t> n2, float lam,
         float eta, int iters, float tol, int threads) {
        const auto C = py::ssize_t(pi::MAX_CHOICES);
        if (Q.ndim() != 3 || Q.shape(1) != C || Q.shape(2) != C) throw std::invalid_argument("Q is [G, 9, 9]");
        const py::ssize_t G = Q.shape(0);
        for (const auto* a : {&m1, &m2})
          if (a->ndim() != 2 || a->shape(0) != G || a->shape(1) != C) throw std::invalid_argument("m is [G, 9]");
        if (n1.size() != G || n2.size() != G) throw std::invalid_argument("n is [G]");
        auto l1 = array<float>({G, C});
        auto l2 = array<float>({G, C});
        auto its = array<int32_t>({G});
        {
          py::gil_scoped_release release;
          pi::mmd_batch(size_t(G), Q.data(), m1.data(), m2.data(), n1.data(), n2.data(), lam, eta, iters, tol,
                        l1.mutable_data(), l2.mutable_data(), its.mutable_data(), threads);
        }
        return py::make_tuple(l1, l2, its);
      },
      py::arg("Q"), py::arg("m1"), py::arg("m2"), py::arg("n1"), py::arg("n2"), py::arg("lam"), py::arg("eta") = 1.0f,
      py::arg("iters") = 2000, py::arg("tol") = 1e-4f, py::arg("threads") = 8);

  m.def(
      "solve",
      [](carray<uint8_t> cells, carray<float> wdl, carray<uint8_t> n1, carray<uint8_t> n2, carray<float> m1,
         carray<float> m2, float shrink, float lam, float eta, int iters, float tol, int threads) {
        const auto C = py::ssize_t(pi::MAX_CHOICES);
        if (cells.ndim() != 3 || cells.shape(2) != 2) throw std::invalid_argument("cells is [S, B, 2]");
        const py::ssize_t S = cells.shape(0), B = cells.shape(1);
        if (wdl.size() != S * B * 3) throw std::invalid_argument("wdl is [S * B, 3]");
        if (n1.size() != S || n2.size() != S || m1.size() != S * C || m2.size() != S * C)
          throw std::invalid_argument("n is [S], m is [S, 9]");
        auto l1 = array<float>({S, C});
        auto l2 = array<float>({S, C});
        auto swdl = array<float>({S, 3});
        auto its = array<int32_t>({S});
        {
          py::gil_scoped_release release;
          pi::solve_batch(size_t(S), size_t(B), cells.data(), wdl.data(), n1.data(), n2.data(), m1.data(), m2.data(),
                          shrink, lam, eta, iters, tol, l1.mutable_data(), l2.mutable_data(), swdl.mutable_data(),
                          its.mutable_data(), threads);
        }
        return py::make_tuple(l1, l2, swdl, its);
      },
      py::arg("cells"), py::arg("wdl"), py::arg("n1"), py::arg("n2"), py::arg("m1"), py::arg("m2"), py::arg("shrink"),
      py::arg("lam"), py::arg("eta") = 1.0f, py::arg("iters") = 2000, py::arg("tol") = 1e-4f, py::arg("threads") = 8);
  m.def(
      "plan",
      [](carray<float> prior, carray<uint8_t> n1, carray<uint8_t> n2, size_t B, float floor, uint64_t seed,
         int threads) {
        const auto C = py::ssize_t(pi::MAX_CHOICES);
        if (prior.ndim() != 3 || prior.shape(1) != 2 || prior.shape(2) != C)
          throw std::invalid_argument("prior is [S, 2, 9]");
        const py::ssize_t S = prior.shape(0);
        if (n1.size() != S || n2.size() != S) throw std::invalid_argument("n is [S]");
        auto cells = array<uint8_t>({S, py::ssize_t(B), py::ssize_t(2)});
        {
          py::gil_scoped_release release;
          pi::plan_cells(size_t(S), B, prior.data(), n1.data(), n2.data(), floor, seed, cells.mutable_data(),
                         threads);
        }
        return cells;
      },
      py::arg("prior"), py::arg("n1"), py::arg("n2"), py::arg("budget"), py::arg("floor"), py::arg("seed"),
      py::arg("threads") = 8);

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
      .def("rollouts",
           [](const PiEnv& env, carray<uint32_t> battle, carray<uint8_t> picks, carray<uint64_t> seeds) {
             const size_t K = check_rollouts(env, battle, picks, seeds);
             const auto k = py::ssize_t(K);
             auto mon = array<int16_t>({k, 2, 6, py::ssize_t(pi::MON_FIELDS)});
             auto hp = array<float>({k, 2, 6});
             auto active = array<float>({k, 2, py::ssize_t(pi::ACTIVE_DIM)});
             auto done = array<uint8_t>({k});
             auto result = array<int8_t>({k});
             run_rollouts(env, battle, picks, seeds, mon.mutable_data(), hp.mutable_data(), active.mutable_data(),
                          done.mutable_data(), result.mutable_data());
             py::dict out;
             out["mon"] = mon;
             out["hp"] = hp;
             out["active"] = active;
             out["done"] = done;
             out["result"] = result;
             return out;
           },
           py::arg("battle"), py::arg("picks"), py::arg("seeds"))
      .def("rollouts_into",
           [](const PiEnv& env, carray<uint32_t> battle, carray<uint8_t> picks, carray<uint64_t> seeds,
              py::array_t<int16_t> mon, py::array_t<float> hp, py::array_t<float> active, py::array_t<uint8_t> done,
              py::array_t<int8_t> result) {
             const size_t K = check_rollouts(env, battle, picks, seeds);
             auto big = [&](const py::array& a, size_t per, const char* name) {
               if (!(a.flags() & py::array::c_style) || !a.writeable() || size_t(a.size()) < K * per)
                 throw std::invalid_argument(std::string(name) + ": a writeable C-contiguous array of K rows");
             };
             big(mon, 2 * 6 * pi::MON_FIELDS, "mon");
             big(hp, 2 * 6, "hp");
             big(active, 2 * pi::ACTIVE_DIM, "active");
             big(done, 1, "done");
             big(result, 1, "result");
             run_rollouts(env, battle, picks, seeds, mon.mutable_data(), hp.mutable_data(), active.mutable_data(),
                          done.mutable_data(), result.mutable_data());
           },
           py::arg("battle"), py::arg("picks"), py::arg("seeds"), py::arg("mon"), py::arg("hp"), py::arg("active"),
           py::arg("done"), py::arg("result"))
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
