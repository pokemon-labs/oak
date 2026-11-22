// Perfect-information lockstep self-play for the perfect-info trainer
// (ataraxos/pi/), whose net goes unmodified into oak's IS-MCTS. Bound by
// ataraxos/_ext/ataraxos_pi.cpp, which links oak's libpkmn_chance build (no
// log), as oak's search does.
//
// One row per battle step: the battle, the durations, both players' libpkmn
// choices. A step where neither player has two choices is played here and
// never reaches the net. Each row's input is oak's Encode::Battle features
// (battle + durations, hidden counters left out), one token per Pokémon:
//
//   mon i16 [2][6][MON_FIELDS]: by side and order position (0 the active):
//     species (0: empty or fainted, masked), status (oak's 15-way index,
//     sleep turns from the durations), then 4 moves as id x PP bucket
//     ((id - 1) * 6 + bucket; -1 for none, Struggle or no PP). The active's
//     moves are side.active.moves (Mimic, Transform), as oak's.
//   hp f32 [2][6]: HP / max HP on oak's 50 buckets, the value oak's search
//     cache embeds (hp = max * bucket / 50), 0 when masked.
//   active f32 [2][ACTIVE_DIM]: oak's dense Active features (stats, types,
//     volatiles, durations), 0 when the active is fainted.
//   policy i16 [2][MAX_CHOICES]: oak's flat Policy index of each choice.
//
// Teams: a pool drawn by weights (Exp3 sets them, as ataraxos's env), or
// truly random teams with no pool. A battle past max_turns ends as a tie,
// `capped`. Shards with their own RNG, so the output depends on the seed and
// the shard count, not the thread count.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <encode/battle/battle.h>
#include <encode/battle/key.h>
#include <encode/battle/policy.h>
#include <libpkmn/data.h>
#include <libpkmn/pkmn.h>
#include <search/durations.h>

#include "teams.hpp"

namespace ataraxos::pi {

inline constexpr size_t MAX_CHOICES = 9;
inline constexpr size_t MON_FIELDS = 6; // species, status, 4 moves
inline constexpr size_t ACTIVE_DIM = Encode::Battle::Active::n_dim;
inline constexpr size_t POLICY_DIM = Encode::Battle::Policy::n_dim;
inline constexpr size_t STATUS_DIM = Encode::Battle::Status::n_dim;
inline constexpr size_t MOVE_DIM = Encode::Battle::Moves::n_dim; // id x PP bucket
inline constexpr uint16_t MAX_TURNS = 1000;

struct Features {
  std::array<int16_t, 2 * 6 * MON_FIELDS> mon;
  std::array<float, 2 * 6> hp;
  std::array<float, 2 * ACTIVE_DIM> active;
};

inline int16_t move_index(PKMN::MoveSlot m) {
  using PKMN::Data::Move;
  if (m.id == Move::None || m.id == Move::Struggle || !m.pp) return -1;
  return int16_t((uint16_t(m.id) - 1) * Encode::Battle::Moves::n_pp + Encode::Battle::ceil_log2_u8(m.pp) - 1);
}

// The HP feature oak's search cache embeds: hp on 50 buckets (cache.h's
// precompute) over max HP.
inline float bucket_hp(const PKMN::Pokemon& p) {
  const uint8_t bucket = Encode::Battle::Key::hp48(p.hp, p.stats.hp);
  return float(uint16_t(uint32_t(p.stats.hp) * bucket / 50)) / float(p.stats.hp);
}

inline void encode(const PKMN::Battle& battle, const PKMN::Durations& durations, Features& f) {
  f.mon.fill(0);
  f.hp.fill(0);
  f.active.fill(0);
  for (size_t s = 0; s < 2; s++) {
    const PKMN::Side& side = battle.sides[s];
    const PKMN::Duration& d = durations.get(s);
    for (size_t k = 0; k < 6; k++) {
      int16_t* m = &f.mon[(s * 6 + k) * MON_FIELDS];
      for (size_t j = 2; j < MON_FIELDS; j++) m[j] = -1;
      const uint8_t id = side.order[k];
      if (id == 0) continue;
      const PKMN::Pokemon& p = side.pokemon[id - 1];
      if (p.hp == 0) continue;
      m[0] = int16_t(p.species);
      m[1] = int16_t(Encode::Battle::Status::get_status_index(p.status, d.sleep(k)));
      const auto& moves = k == 0 ? side.active.moves : p.moves;
      for (size_t j = 0; j < 4; j++) m[2 + j] = move_index(moves[j]);
      f.hp[s * 6 + k] = bucket_hp(p);
      if (k == 0) Encode::Battle::Active::write(side.active, d, &f.active[s * ACTIVE_DIM]);
    }
  }
}

struct Row {
  uint32_t battle;
  uint32_t game;
  std::array<uint8_t, 2> n;
  std::array<std::array<pkmn_choice, MAX_CHOICES>, 2> choices;
  std::array<std::array<int16_t, MAX_CHOICES>, 2> policy;
  Features features;
};

struct Teams {
  std::array<int32_t, 2> index{-1, -1};
  std::array<double, 2> prob{0, 0};
};

struct End {
  uint32_t battle;
  uint32_t game;
  int8_t outcome; // P1's: 1 win, 0 tie, -1 loss
  Teams teams;
  bool capped = false;
};

struct Battle {
  pkmn_gen1_battle battle;
  pkmn_gen1_battle_options options;
  pkmn_result result;
  std::array<std::array<pkmn_choice, MAX_CHOICES>, 2> choices;
  std::array<uint8_t, 2> n;

  const PKMN::Battle& state() const { return PKMN::view(battle); }
  const PKMN::Durations& durations() const {
    return PKMN::view(*pkmn_gen1_battle_options_chance_durations(&options));
  }

  void reset(const pkmn_gen1_battle& b) {
    battle = b;
    options = {};
    pkmn_gen1_chance_options chance{};
    pkmn_gen1_battle_options_set(&options, nullptr, &chance, nullptr);
    update(0, 0);
  }
  void update(pkmn_choice c1, pkmn_choice c2) {
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    if (pkmn_result_type(result) == PKMN_RESULT_ERROR) throw std::logic_error("libpkmn error");
    if (over()) {
      n = {0, 0};
      return;
    }
    n[0] = uint8_t(pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices[0].data(),
                                            choices[0].size()));
    n[1] = uint8_t(pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices[1].data(),
                                            choices[1].size()));
  }
  bool over() const { return pkmn_result_type(result) != PKMN_RESULT_NONE; }
};

struct Shard {
  size_t begin = 0, end = 0;
  std::mt19937_64 rng;
  std::vector<Row> rows;
  std::vector<End> ends; // since the last step began
  std::vector<uint64_t> team_counts;
  uint64_t forced = 0, games = 0, capped = 0;
};

class SelfPlay {
 public:
  SelfPlay(size_t battles, uint64_t seed, size_t shards, int threads, std::vector<Team> teams, uint16_t max_turns)
      : battles_(battles), games_(battles, 0), teams_of_(battles), shards_(std::min(shards, battles)),
        threads_(threads), teams_(std::move(teams)), max_turns_(max_turns ? max_turns : MAX_TURNS) {
    if (battles == 0 || shards == 0) throw std::invalid_argument("battles and shards must be positive");
    if (PKMN_GEN1_MAX_CHOICES != MAX_CHOICES) throw std::logic_error("PKMN_GEN1_MAX_CHOICES isn't 9");
    for (const Team& t : teams_) {
      if (t.empty() || t.size() > 6) throw std::invalid_argument("a team has 1-6 Pokémon");
    }
    set_team_weights(std::vector<double>(teams_.size(), 1.0));
    const size_t k = shards_.size();
    for (size_t s = 0; s < k; s++) {
      Shard& sh = shards_[s];
      sh.begin = battles * s / k;
      sh.end = battles * (s + 1) / k;
      std::seed_seq seq{seed, uint64_t(s)};
      sh.rng.seed(seq);
      sh.team_counts.assign(teams_.size(), 0);
    }
  }

  // New games draw both teams from w (normalized). Call between steps.
  void set_team_weights(const std::vector<double>& w) {
    if (w.size() != teams_.size()) throw std::invalid_argument("one weight per team");
    double total = 0;
    for (double x : w) {
      if (!(x >= 0)) throw std::invalid_argument("weights must be nonnegative");
      total += x;
    }
    if (!teams_.empty() && !(total > 0)) throw std::invalid_argument("weights sum to 0");
    prob_.resize(w.size());
    cdf_.resize(w.size());
    double acc = 0;
    for (size_t i = 0; i < w.size(); i++) {
      prob_[i] = w[i] / total;
      acc += prob_[i];
      cdf_[i] = acc;
    }
  }
  const std::vector<double>& team_probs() const { return prob_; }
  size_t pool_size() const { return teams_.size(); }

  void reset() {
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
    for (size_t s = 0; s < shards_.size(); s++) {
      Shard& sh = shards_[s];
      sh.ends.clear();
      for (size_t b = sh.begin; b < sh.end; b++) {
        games_[b] = 0;
        start(sh, b);
      }
      gather(sh);
    }
  }

  // picks[i] = {P1's, P2's} index into row i's choices (0 for a side with
  // one choice).
  void step(std::span<const std::array<uint8_t, 2>> picks) {
    if (picks.size() != rows()) throw std::invalid_argument("one pick pair per row");
    std::vector<size_t> at(shards_.size());
    for (size_t s = 0, n = 0; s < shards_.size(); n += shards_[s].rows.size(), s++) at[s] = n;
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
    for (size_t s = 0; s < shards_.size(); s++) {
      Shard& sh = shards_[s];
      sh.ends.clear();
      for (size_t k = 0; k < sh.rows.size(); k++) {
        const Row& r = sh.rows[k];
        const auto pick = picks[at[s] + k];
        assert(pick[0] < std::max<uint8_t>(r.n[0], 1) && pick[1] < std::max<uint8_t>(r.n[1], 1));
        advance(sh, r.battle, r.choices[0][pick[0]], r.choices[1][pick[1]]);
      }
      gather(sh);
    }
  }

  // Depth-1 rollouts for search in training (ataraxos/pi/search.py): for each
  // k, a copy of battle[k] (as its current row) with a fresh RNG seed and
  // its hidden counters drawn from the durations (oak's MCTS root
  // iteration), plays picks[k] = {P1's, P2's} index into the row's choices,
  // then the forced steps, up to the next state where the game is over or a
  // side has 2+ choices: the leaf. done[k] marks a finished game (result
  // P1's +1/0/-1; a turn past max_turns is a tie), else features[k] is the
  // leaf for the value head, written to mon / hp / active at row k (the
  // layout of a row's features; zeros where done).
  void rollouts(std::span<const uint32_t> battle, std::span<const std::array<uint8_t, 2>> picks,
                std::span<const uint64_t> seeds, int16_t* mon, float* hp, float* active, uint8_t* done,
                int8_t* result) const {
    constexpr size_t NM = 2 * 6 * MON_FIELDS, NH = 2 * 6, NA = 2 * ACTIVE_DIM;
    const size_t K = battle.size();
#pragma omp parallel for schedule(static) num_threads(threads_)
    for (size_t k = 0; k < K; k++) {
      Battle c = battles_[battle[k]];
      PKMN::view(c.battle).rng = seeds[k];
      pkmn_gen1_chance_options chance{};
      chance.durations = *pkmn_gen1_battle_options_chance_durations(&c.options);
      MCTS::randomize_hidden_variables(c.battle, chance.durations);
      pkmn_gen1_battle_options_set(&c.options, nullptr, &chance, nullptr);
      const auto p = picks[k];
      c.update(c.choices[0][p[0]], c.choices[1][p[1]]);
      Features f;
      for (;;) {
        if (c.over() || c.state().turn > max_turns_) {
          const auto r = c.over() ? pkmn_result_type(c.result) : PKMN_RESULT_TIE;
          done[k] = 1;
          result[k] = r == PKMN_RESULT_WIN ? 1 : r == PKMN_RESULT_LOSE ? -1 : 0;
          std::fill_n(mon + k * NM, NM, int16_t(0));
          std::fill_n(hp + k * NH, NH, 0.f);
          std::fill_n(active + k * NA, NA, 0.f);
          break;
        }
        if (c.n[0] > 1 || c.n[1] > 1) {
          done[k] = 0;
          result[k] = 0;
          encode(c.state(), c.durations(), f);
          std::copy(f.mon.begin(), f.mon.end(), mon + k * NM);
          std::copy(f.hp.begin(), f.hp.end(), hp + k * NH);
          std::copy(f.active.begin(), f.active.end(), active + k * NA);
          break;
        }
        c.update(c.choices[0][0], c.choices[1][0]);
      }
    }
  }

  size_t rows() const {
    size_t n = 0;
    for (const Shard& sh : shards_) n += sh.rows.size();
    return n;
  }
  const std::vector<Shard>& shards() const { return shards_; }
  int threads() const { return threads_; }
  const Battle& battle(size_t b) const { return battles_[b]; }

 private:
  std::vector<Battle> battles_;
  std::vector<uint32_t> games_;
  std::vector<Teams> teams_of_;
  std::vector<Shard> shards_;
  int threads_;
  std::vector<Team> teams_;
  uint16_t max_turns_;
  std::vector<double> prob_, cdf_;

  size_t draw(Shard& sh) {
    const double u = std::uniform_real_distribution<double>(0, cdf_.back())(sh.rng);
    size_t i = size_t(std::upper_bound(cdf_.begin(), cdf_.end(), u) - cdf_.begin());
    if (i == cdf_.size()) {
      for (i--; prob_[i] == 0; i--) {}
    }
    sh.team_counts[i]++;
    return i;
  }

  void start(Shard& sh, size_t b) {
    Teams& t = teams_of_[b];
    if (teams_.empty()) {
      t = Teams{};
      const Team t1 = random_team(sh.rng);
      const Team t2 = random_team(sh.rng);
      battles_[b].reset(init_battle(sh.rng(), t1, t2));
    } else {
      for (size_t s = 0; s < 2; s++) {
        const size_t i = draw(sh);
        t.index[s] = int32_t(i);
        t.prob[s] = prob_[i];
      }
      battles_[b].reset(init_battle(sh.rng(), teams_[t.index[0]], teams_[t.index[1]]));
    }
    settle(sh, b);
  }

  void end(Shard& sh, size_t b, int8_t outcome, bool capped) {
    sh.ends.push_back({uint32_t(b), games_[b], outcome, teams_of_[b], capped});
    sh.games++;
    games_[b]++;
    start(sh, b);
  }

  // Ends a finished or capped game (starting the next), until the battle
  // waits on a real choice. Forced steps (neither side has two choices)
  // are played here.
  void settle(Shard& sh, size_t b) {
    Battle& d = battles_[b];
    for (;;) {
      if (d.over()) {
        const auto r = pkmn_result_type(d.result);
        return end(sh, b, r == PKMN_RESULT_WIN ? 1 : r == PKMN_RESULT_LOSE ? -1 : 0, false);
      }
      if (d.state().turn > max_turns_) {
        sh.capped++;
        return end(sh, b, 0, true);
      }
      if (d.n[0] > 1 || d.n[1] > 1) return;
      sh.forced++;
      d.update(d.choices[0][0], d.choices[1][0]);
    }
  }

  void advance(Shard& sh, size_t b, pkmn_choice c1, pkmn_choice c2) {
    battles_[b].update(c1, c2);
    settle(sh, b);
  }

  void gather(Shard& sh) {
    sh.rows.clear();
    for (size_t b = sh.begin; b < sh.end; b++) {
      const Battle& d = battles_[b];
      Row& r = sh.rows.emplace_back();
      r.battle = uint32_t(b);
      r.game = games_[b];
      r.n = d.n;
      for (size_t s = 0; s < 2; s++) {
        r.choices[s].fill(0);
        r.policy[s].fill(0);
        for (size_t i = 0; i < d.n[s]; i++) {
          r.choices[s][i] = d.choices[s][i];
          r.policy[s][i] = int16_t(Encode::Battle::Policy::get_index(d.state().sides[s], d.choices[s][i]));
        }
      }
      encode(d.state(), d.durations(), r.features);
    }
  }
};

// Two-player MMD to the fixed point for one root, search_np.mmd with one
// type a side and extragradient steps: Q [MAX_CHOICES]^2 (P1's, row-major),
// magnets' log-probs m1, m2 (the net's prior), a, b legal choices (a side
// with one has the policy [1]), lam the regularization, eta the step, at
// most `iters` iterations or until no probability moves more than `tol`.
// Writes the log-policies l1, l2 (-1e9 past a, b); returns the iterations.
inline int mmd_root(const double* q, const float* m1, const float* m2, size_t a, size_t b, double lam, double eta,
                    int iters, double tol, float* l1, float* l2) {
  constexpr size_t C = MAX_CHOICES;
  constexpr double NEG = -1e9;
  std::array<double, C> M1{}, M2{}, x1{}, x2{}, h1{}, h2{}, y1{}, y2{}, p1{}, p2{};
  auto norm = [](std::array<double, C>& l, size_t n) {
    double mx = NEG;
    for (size_t i = 0; i < n; i++) mx = std::max(mx, l[i]);
    double s = 0;
    for (size_t i = 0; i < n; i++) s += std::exp(l[i] - mx);
    const double z = mx + std::log(s);
    for (size_t i = 0; i < n; i++) l[i] -= z;
    for (size_t i = n; i < C; i++) l[i] = NEG;
  };
  for (size_t i = 0; i < C; i++) M1[i] = m1[i], M2[i] = m2[i];
  norm(M1, a);
  norm(M2, b);
  x1 = M1, x2 = M2;
  // one MMD step from (x1, x2) with q read at the policies (at1, at2)
  auto step = [&](const std::array<double, C>& at1, const std::array<double, C>& at2, std::array<double, C>& o1,
                  std::array<double, C>& o2) {
    std::array<double, C> e1{}, e2{};
    for (size_t i = 0; i < a; i++) e1[i] = std::exp(at1[i]);
    for (size_t j = 0; j < b; j++) e2[j] = std::exp(at2[j]);
    for (size_t i = 0; i < a; i++) {
      double v = 0;
      for (size_t j = 0; j < b; j++) v += q[i * C + j] * e2[j];
      o1[i] = (x1[i] + eta * v + eta * lam * M1[i]) / (1 + eta * lam);
    }
    for (size_t j = 0; j < b; j++) {
      double v = 0;
      for (size_t i = 0; i < a; i++) v -= q[i * C + j] * e1[i];
      o2[j] = (x2[j] + eta * v + eta * lam * M2[j]) / (1 + eta * lam);
    }
    norm(o1, a);
    norm(o2, b);
  };
  int it = 0;
  if (a > 1 || b > 1) {
    for (it = 1; it <= iters; it++) {
      for (size_t i = 0; i < a; i++) p1[i] = std::exp(x1[i]);
      for (size_t j = 0; j < b; j++) p2[j] = std::exp(x2[j]);
      step(x1, x2, h1, h2);
      step(h1, h2, y1, y2);
      double d = 0;
      for (size_t i = 0; i < a; i++) d = std::max(d, std::abs(std::exp(y1[i]) - p1[i]));
      for (size_t j = 0; j < b; j++) d = std::max(d, std::abs(std::exp(y2[j]) - p2[j]));
      x1 = y1, x2 = y2;
      if (d < tol) break;
    }
    it = std::min(it, iters);
  }
  for (size_t i = 0; i < C; i++) l1[i] = float(x1[i]), l2[i] = float(x2[i]);
  return it;
}

// mmd_root over G roots: Q [G][C][C], m1, m2, l1, l2 [G][C], n1, n2, its [G].
inline void mmd_batch(size_t G, const float* Q, const float* m1, const float* m2, const uint8_t* n1,
                      const uint8_t* n2, float lam, float eta, int iters, float tol, float* l1, float* l2,
                      int32_t* its, int threads) {
  constexpr size_t C = MAX_CHOICES;
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
  for (size_t g = 0; g < G; g++) {
    std::array<double, C * C> q;
    for (size_t k = 0; k < C * C; k++) q[k] = Q[g * C * C + k];
    its[g] = mmd_root(q.data(), m1 + g * C, m2 + g * C, std::max<uint8_t>(n1[g], 1), std::max<uint8_t>(n2[g], 1),
                      lam, eta, iters, tol, l1 + g * C, l2 + g * C);
  }
}

// search_np.cell_means for one root and one value channel: sums / counts
// [C][C] over the legal a x b cells; an empty cell reads the mean of its
// row's and column's means (one if the other is empty, else the root's mean,
// else 0), every cell shrunk toward that by `shrink` pseudo-counts; 0 past
// a, b.
inline void cell_means_root(const double* total, const double* count, size_t a, size_t b, double shrink,
                            double* out) {
  constexpr size_t C = MAX_CHOICES;
  std::array<double, C> rs{}, rc{}, cs{}, cc{};
  double bs = 0, bc = 0;
  for (size_t i = 0; i < a; i++) {
    for (size_t j = 0; j < b; j++) {
      const double t = total[i * C + j], n = count[i * C + j];
      rs[i] += t, rc[i] += n, cs[j] += t, cc[j] += n, bs += t, bc += n;
    }
  }
  const double block = bc > 0 ? bs / bc : 0.0;
  for (size_t i = 0; i < C; i++) {
    for (size_t j = 0; j < C; j++) {
      if (i >= a || j >= b) {
        out[i * C + j] = 0;
        continue;
      }
      const bool hr = rc[i] > 0, hc = cc[j] > 0;
      const double prior = hr && hc ? (rs[i] / rc[i] + cs[j] / cc[j]) / 2
                           : hr     ? rs[i] / rc[i]
                           : hc     ? cs[j] / cc[j]
                                    : block;
      const double t = total[i * C + j], n = count[i * C + j];
      out[i * C + j] = shrink > 0 ? (t + shrink * prior) / (n + shrink) : n > 0 ? t / n : prior;
    }
  }
}

// The whole solve for S roots of B rollouts each (search in training's
// finish, --search_solve cpp): cells [S][B][2] (P1's, P2's choice),
// wdl [S][B][3] the leaves' P1 (loss, draw, win), n1, n2, m1, m2 as
// mmd_batch; cell means of Q = win - loss and of each W/D/L channel, MMD,
// then the W/D/L under the posteriors. Writes l1, l2 [S][C], swdl [S][3],
// its [S].
inline void solve_batch(size_t S, size_t B, const uint8_t* cells, const float* wdl, const uint8_t* n1,
                        const uint8_t* n2, const float* m1, const float* m2, float shrink, float lam, float eta,
                        int iters, float tol, float* l1, float* l2, float* swdl, int32_t* its, int threads) {
  constexpr size_t C = MAX_CHOICES;
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
  for (size_t s = 0; s < S; s++) {
    std::array<double, C * C> count{}, tq{}, Q{};
    std::array<std::array<double, C * C>, 3> tw{}, W{};
    for (size_t k = 0; k < B; k++) {
      const uint8_t* c = cells + (s * B + k) * 2;
      const float* w = wdl + (s * B + k) * 3;
      const size_t cell = size_t(c[0]) * C + c[1];
      count[cell] += 1;
      tq[cell] += double(w[2]) - double(w[0]);
      for (size_t h = 0; h < 3; h++) tw[h][cell] += w[h];
    }
    const size_t a = std::max<uint8_t>(n1[s], 1), b = std::max<uint8_t>(n2[s], 1);
    cell_means_root(tq.data(), count.data(), a, b, shrink, Q.data());
    for (size_t h = 0; h < 3; h++) cell_means_root(tw[h].data(), count.data(), a, b, shrink, W[h].data());
    its[s] = mmd_root(Q.data(), m1 + s * C, m2 + s * C, a, b, lam, eta, iters, tol, l1 + s * C, l2 + s * C);
    std::array<double, C> p1{}, p2{};
    double z1 = 0, z2 = 0;
    for (size_t i = 0; i < a; i++) z1 += p1[i] = std::exp(double(l1[s * C + i]));
    for (size_t j = 0; j < b; j++) z2 += p2[j] = std::exp(double(l2[s * C + j]));
    for (size_t h = 0; h < 3; h++) {
      double v = 0;
      for (size_t i = 0; i < a; i++)
        for (size_t j = 0; j < b; j++) v += p1[i] * p2[j] * W[h][i * C + j];
      swdl[s * 3 + h] = float(v / (z1 * z2));
    }
  }
}

// The rollouts' cells for S roots (search in training's begin, --search_plan
// cpp): B pairs a root, each with probability `floor` uniform over the legal
// a x b pairs, else P1's and P2's choices drawn from their priors (probs
// [S][2][C], 0 past n). seed and the root index seed each root's RNG.
inline void plan_cells(size_t S, size_t B, const float* prior, const uint8_t* n1, const uint8_t* n2, float floor,
                       uint64_t seed, uint8_t* cells, int threads) {
  constexpr size_t C = MAX_CHOICES;
#pragma omp parallel for schedule(static) num_threads(threads)
  for (size_t s = 0; s < S; s++) {
    std::mt19937_64 rng(seed ^ (0x9E3779B97F4A7C15ull * (s + 1)));
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const size_t n[2] = {std::max<uint8_t>(n1[s], 1), std::max<uint8_t>(n2[s], 1)};
    std::array<std::array<double, C>, 2> cdf{};
    for (size_t side = 0; side < 2; side++) {
      double acc = 0;
      for (size_t i = 0; i < n[side]; i++) cdf[side][i] = acc += double(prior[(s * 2 + side) * C + i]);
    }
    for (size_t k = 0; k < B; k++) {
      uint8_t* c = cells + (s * B + k) * 2;
      const bool uni = u(rng) < floor;
      for (size_t side = 0; side < 2; side++) {
        const size_t m = n[side];
        size_t i;
        if (uni) {
          i = std::min(size_t(u(rng) * double(m)), m - 1);
        } else {
          const double x = u(rng) * cdf[side][m - 1];
          i = 0;
          while (i + 1 < m && x > cdf[side][i]) i++;
        }
        c[side] = uint8_t(i);
      }
    }
  }
}

} // namespace ataraxos::pi
