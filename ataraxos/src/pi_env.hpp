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

} // namespace ataraxos::pi
