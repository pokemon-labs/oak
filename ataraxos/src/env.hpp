// Lockstep self-play over many battles, for the Python trainer
// (ataraxos/_ext/ataraxos_env.cpp binds it). birch's src/env.hpp on
// libpkmn's Gen 1.
//
// `step` takes one pick per decision of the last gather (an index into that
// decision's choices), answers them, and gathers the next decisions: every
// player with a request to answer and at least two choices. A request with
// one choice is answered here and never reaches the network; its update's
// tokens join the player's next decision. A finished battle is reported as
// an End and restarts on two new teams: two drawn from the pool,
// independently from the team weights (`set_team_weights`, uniform until
// set; ataraxos/rl/teams.py's Exp3 sets them), or with no pool two truly
// random ones (teams.hpp). Each End carries its teams and the probability
// each was drawn with, for Exp3's importance weights; `team_counts` counts
// the draws. `schedule` queues games with given teams and probabilities,
// which new games take before drawing; `add_teams` appends teams to the
// pool, with weight 0 until set. With `set_idle(true)` a battle whose shard
// has nothing queued waits, asking nothing, instead of drawing a game, and
// starts the next game queued to its shard at a later gather. `play` queues
// games that bring their own two teams, outside the pool; their End reports
// the caller's tag as both teams' index, with probability 0.
// Each seat can have its own team distribution (`set_team_weights(w,
// role)`): role 0 draws P1's team and role 1 P2's, or, with
// `set_alternate(true)`, the other way round in the odd battles, so an
// agent that plays P1 in the even battles and P2 in the odd ones always
// draws from role 0's.
// With `max_turns`, a battle still going as turn max_turns + 1 starts ends
// as a tie with `capped` set, as Showdown's turn limit is (1000 turns); the
// trainer's `at_cap` decides what it counts as (ataraxos/rl/buffer.py).
//
// With `state` (off by default), each decision also carries its player's
// bookkeeping state (state_tokens.hpp), STATE_FIELDS int16s. `set_light`
// (2026-11-24, pi4's training, which reads src/pi4.hpp's input through the
// binding's pi4_obs) leaves out each decision's tokens and state; the
// Bookkeepings still track.
//
// The battles are split into fixed shards, each with its own RNG and run on
// any thread, so the output depends on the seed and the shard count, not on
// the thread count.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <deque>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "battle_data.hpp"
#include "state_tokens.hpp"
#include "teams.hpp"

namespace ataraxos {

// Showdown's turn limit.
inline constexpr uint16_t MAX_TURNS = 1000;

struct Decision {
  uint32_t battle;
  uint32_t game; // the battle's game count when it was asked
  uint8_t player; // 0 P1, 1 P2
  uint8_t n;      // choices, at least 2
  std::array<pkmn_choice, MAX_CHOICES> options;
};

// A game's teams: pool indices and the probability each was drawn with (-1
// and 0 without a pool).
struct Teams {
  std::array<int32_t, 2> index{-1, -1};
  std::array<double, 2> prob{0, 0};
};

enum class Outcome : int8_t { Lose = -1, Tie = 0, Win = 1 }; // P1's

struct End {
  uint32_t battle;
  uint32_t game;
  Outcome outcome;
  Teams teams;
  bool capped = false; // a Tie by max_turns
};

// A scheduled game: pool teams, or its own (`play`).
struct Queued {
  Teams teams;
  std::shared_ptr<const std::array<Team, 2>> own;
};

struct Shard {
  size_t begin = 0, end = 0;
  std::mt19937_64 rng;
  std::vector<Decision> decisions;
  std::vector<Token> tokens;      // each decision's chunk, in order
  std::vector<uint32_t> lengths;  // tokens per decision
  std::vector<int16_t> requests;  // REQUEST_FIELDS per decision
  std::vector<int16_t> states;    // STATE_FIELDS per decision, with `state`
  std::vector<End> ends;          // since the last step began
  std::vector<uint64_t> team_counts; // draws per pool team, both sides
  std::deque<Queued> queue;          // scheduled games, taken before drawing
  uint64_t forced = 0, games = 0, capped = 0;
};

class SelfPlay {
 public:
  // An empty `teams` plays truly random teams; `max_turns` 0 is Showdown's
  // limit.
  SelfPlay(size_t battles, uint64_t seed, size_t shards, int threads, std::vector<Team> teams,
           uint16_t max_turns, bool state = false)
      : battles_(new BattleData[battles]), size_(battles), games_(battles, 0), teams_of_(battles),
        shards_(std::min(shards, battles)), threads_(threads), teams_(std::move(teams)),
        max_turns_(max_turns ? max_turns : MAX_TURNS), state_(state) {
    assert(battles > 0 && shards > 0);
    if (PKMN_GEN1_MAX_CHOICES != MAX_CHOICES) throw std::logic_error("PKMN_GEN1_MAX_CHOICES isn't 9");
    for (const Team& t : teams_) {
      if (t.empty() || t.size() > 6) throw std::invalid_argument("a team has 1-6 Pokémon");
    }
    for (size_t b = 0; b < battles; b++) battles_[b].track = state;
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

  // The distribution new games draw teams from, one weight per pool team
  // (normalized here): `role` 0 or 1's (see the top), -1 both. Call between
  // steps.
  void set_team_weights(const std::vector<double>& w, int role = -1) {
    if (w.size() != teams_.size()) throw std::invalid_argument("one weight per team");
    if (role < -1 || role > 1) throw std::invalid_argument("role is -1, 0 or 1");
    double total = 0;
    for (double x : w) {
      if (!(x >= 0)) throw std::invalid_argument("weights must be nonnegative");
      total += x;
    }
    if (!teams_.empty() && !(total > 0)) throw std::invalid_argument("weights sum to 0");
    for (int r = 0; r < 2; r++) {
      if (role != -1 && role != r) continue;
      prob_[r].resize(w.size());
      cdf_[r].resize(w.size());
      double acc = 0;
      for (size_t i = 0; i < w.size(); i++) {
        prob_[r][i] = w[i] / total;
        acc += prob_[r][i];
        cdf_[r][i] = acc;
      }
    }
  }
  void set_alternate(bool on) { alternate_ = on; }
  // Appends teams to the pool, with weight 0; returns the first's index.
  // Call between steps.
  size_t add_teams(std::vector<Team> teams) {
    const size_t first = teams_.size();
    for (Team& t : teams) teams_.push_back(std::move(t));
    for (int r = 0; r < 2; r++) {
      const double last = cdf_[r].empty() ? 0 : cdf_[r].back();
      prob_[r].resize(teams_.size(), 0);
      cdf_[r].resize(teams_.size(), last);
    }
    for (Shard& sh : shards_) sh.team_counts.resize(teams_.size(), 0);
    return first;
  }

  // Queues games: each new game takes the shortest queue's first, if any,
  // before drawing. `prob` is what its End reports. Call between steps.
  void schedule(std::span<const Teams> games) {
    for (const Teams& g : games) {
      for (size_t s = 0; s < 2; s++) {
        if (g.index[s] < 0 || size_t(g.index[s]) >= teams_.size()) throw std::invalid_argument("no such team");
      }
      shortest_queue().push_back({g, nullptr});
    }
  }
  // Queues games with their own teams (each 1-6 Pokémon), outside the pool;
  // game i's End reports {tags[i], tags[i]} as its teams, probability 0. Call
  // between steps.
  void play(std::vector<std::array<Team, 2>> games, std::span<const int32_t> tags) {
    if (tags.size() != games.size()) throw std::invalid_argument("one tag per game");
    for (size_t i = 0; i < games.size(); i++) {
      Teams t;
      t.index = {tags[i], tags[i]};
      shortest_queue().push_back({t, std::make_shared<const std::array<Team, 2>>(std::move(games[i]))});
    }
  }
  void set_idle(bool on) { idle_when_empty_ = on; }
  // Records each battle's omniscient log from its next game on (BattleData::trace).
  void set_trace(bool on) {
    for (size_t b = 0; b < size_; b++) battles_[b].trace = on;
  }
  const BattleData& battle_data(size_t b) const { return battles_[b]; }
  size_t idle() const {
    size_t n = 0;
    for (uint8_t x : idle_) n += x;
    return n;
  }
  size_t scheduled() const {
    size_t n = 0;
    for (const Shard& sh : shards_) n += sh.queue.size();
    return n;
  }

  const std::vector<double>& team_probs(int role = 0) const { return prob_[role]; }
  size_t pool_size() const { return teams_.size(); }

  // Starts every battle and gathers the first decisions.
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

  // `picks[i]` answers decision i of the last gather, in shard order.
  void step(std::span<const uint8_t> picks) {
    assert(picks.size() == decisions());
    std::vector<size_t> at(shards_.size());
    for (size_t s = 0; s < shards_.size(); s++) at[s] = first(s);
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
    for (size_t s = 0; s < shards_.size(); s++) {
      Shard& sh = shards_[s];
      sh.ends.clear();
      const uint8_t* pick = picks.data() + at[s];
      for (size_t k = 0; k < sh.decisions.size(); k++) {
        const Decision& d = sh.decisions[k];
        assert(pick[k] < d.n);
        // A battle's decisions are both players' of one request, so neither
        // can end the game before the other is answered.
        assert(games_[d.battle] == d.game);
        battles_[d.battle].asked[d.player] = false;
        answer(sh, d.battle, d.player, d.options[pick[k]]);
      }
      gather(sh);
    }
  }

  size_t decisions() const {
    size_t n = 0;
    for (const Shard& sh : shards_) n += sh.decisions.size();
    return n;
  }
  // The index of shard `s`'s first decision.
  size_t first(size_t s) const {
    size_t n = 0;
    for (size_t t = 0; t < s; t++) n += shards_[t].decisions.size();
    return n;
  }
  const std::vector<Shard>& shards() const { return shards_; }
  size_t size() const { return size_; }
  int threads() const { return threads_; }
  // Whether each decision carries its state (state_tokens.hpp).
  bool state() const { return state_ && !light_; }
  bool light() const { return light_; }
  // Whether the battles' Bookkeepings track (`state`, light or not).
  bool tracking() const { return state_; }
  void set_light(bool on) { light_ = on; }
  const BattleData& battle(size_t b) const { return battles_[b]; }
  // Keep each player's Bookkeeping after every update (with `state`), for
  // searching (rollouts.hpp). Takes effect as games start.
  void set_keep_after(bool on) {
    for (size_t b = 0; b < size_; b++) battles_[b].keep_after = on;
  }

 private:
  std::unique_ptr<BattleData[]> battles_;
  size_t size_;
  std::vector<uint32_t> games_;
  std::vector<Teams> teams_of_; // each battle's current game's
  std::vector<Shard> shards_;
  int threads_;
  std::vector<Team> teams_;
  uint16_t max_turns_;
  bool state_;
  bool light_ = false;
  std::array<std::vector<double>, 2> prob_, cdf_; // per role
  bool alternate_ = false;
  bool idle_when_empty_ = false;
  std::vector<uint8_t> idle_ = std::vector<uint8_t>(size_, 0); // waiting for a scheduled game

  std::deque<Queued>& shortest_queue() {
    size_t best = 0;
    for (size_t s = 1; s < shards_.size(); s++) {
      if (shards_[s].queue.size() < shards_[best].queue.size()) best = s;
    }
    return shards_[best].queue;
  }

  size_t draw(Shard& sh, int role) {
    const std::vector<double>& prob = prob_[role];
    const std::vector<double>& cdf = cdf_[role];
    const double u = std::uniform_real_distribution<double>(0, cdf.back())(sh.rng);
    // cdf[i - 1] <= u < cdf[i], so prob[i] > 0; u == cdf.back() (a
    // rounding the distribution allows) takes the last team with weight.
    size_t i = size_t(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
    if (i == cdf.size()) {
      for (i--; prob[i] == 0; i--) {}
    }
    sh.team_counts[i]++;
    return i;
  }

  void start(Shard& sh, size_t b) {
    Teams& t = teams_of_[b];
    idle_[b] = 0;
    if (!sh.queue.empty()) {
      const Queued q = std::move(sh.queue.front());
      sh.queue.pop_front();
      t = q.teams;
      const Team& t1 = q.own ? (*q.own)[0] : teams_[t.index[0]];
      const Team& t2 = q.own ? (*q.own)[1] : teams_[t.index[1]];
      battles_[b].reset(init_battle(sh.rng(), t1, t2));
      return;
    }
    if (idle_when_empty_) {
      idle_[b] = 1;
      return;
    }
    if (teams_.empty()) {
      t = Teams{};
      const Team t1 = random_team(sh.rng);
      const Team t2 = random_team(sh.rng);
      battles_[b].reset(init_battle(sh.rng(), t1, t2));
      return;
    }
    for (size_t s = 0; s < 2; s++) {
      const int role = int(s ^ (alternate_ ? b & 1 : 0));
      const size_t i = draw(sh, role);
      t.index[s] = int32_t(i);
      t.prob[s] = prob_[role][i];
    }
    battles_[b].reset(init_battle(sh.rng(), teams_[t.index[0]], teams_[t.index[1]]));
  }

  void end(Shard& sh, size_t b, Outcome outcome, bool capped = false) {
    sh.ends.push_back({uint32_t(b), games_[b], outcome, teams_of_[b], capped});
    sh.games++;
    games_[b]++;
    start(sh, b);
  }

  static Outcome outcome(pkmn_result r) {
    switch (pkmn_result_type(r)) {
    case PKMN_RESULT_WIN: return Outcome::Win;
    case PKMN_RESULT_LOSE: return Outcome::Lose;
    default: return Outcome::Tie;
    }
  }

  void answer(Shard& sh, size_t b, uint8_t p, pkmn_choice c) {
    BattleData& d = battles_[b];
    switch (d.choose(p, c)) {
    case BattleData::Held: break;
    case BattleData::Advanced:
      if (d.turn() > max_turns_) {
        sh.capped++;
        end(sh, b, Outcome::Tie, true);
      }
      break;
    case BattleData::Ended:
      assert(pkmn_result_type(d.result) != PKMN_RESULT_ERROR);
      end(sh, b, outcome(d.result));
      break;
    }
  }

  // Answers every one-choice request, then asks every other one.
  void gather(Shard& sh) {
    sh.decisions.clear();
    sh.tokens.clear();
    sh.lengths.clear();
    sh.requests.clear();
    sh.states.clear();
    for (size_t b = sh.begin; b < sh.end; b++) {
      if (idle_[b]) {
        if (sh.queue.empty()) continue;
        start(sh, b);
      }
      BattleData& d = battles_[b];
      for (bool again = true; again && !idle_[b];) { // a forced answer can end the game
        again = false;
        for (uint8_t p = 0; p < 2; p++) {
          if (d.asked[p] || !d.needs_choice(p)) continue;
          Decision x{uint32_t(b), games_[b], p, d.n_option[p], {}};
          std::copy_n(d.option[p].begin(), x.n, x.options.begin());
          assert(x.n >= 1);
          if (x.n == 1) {
            sh.forced++;
            answer(sh, b, p, x.options[0]);
            again = true;
            break;
          }
          if (light_) {
            sh.lengths.push_back(0);
          } else {
            sh.tokens.insert(sh.tokens.end(), d.chunk[p].begin(), d.chunk[p].end());
            sh.lengths.push_back(uint32_t(d.chunk[p].size()));
          }
          d.chunk[p].clear();
          const size_t at = sh.requests.size();
          sh.requests.resize(at + REQUEST_FIELDS);
          encode(d.request[p], &sh.requests[at]);
          if (state()) {
            const size_t st = sh.states.size();
            sh.states.resize(st + STATE_FIELDS);
            encode_state(d.bk[p], &sh.states[st]);
          }
          d.asked[p] = true;
          sh.decisions.push_back(x);
        }
      }
    }
  }
};

} // namespace ataraxos
