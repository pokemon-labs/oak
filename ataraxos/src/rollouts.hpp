// Ataraxos's test-time search, its rollouts (the user's design, 2026-11-01):
// at a root decision of ours, `worlds` history samples (history_sampler.hpp),
// and in each world `reps` rollouts per legal choice of ours (Pokémon is
// stochastic, and the foe's choices are sampled too), worlds * reps * n in
// all. A rollout plays our choice first (forced: the root's net already ran
// on that summary), then both sides' nets choose (their picks come from
// Python, as SelfPlay's), the foe's from its rebuilt history. After `depth`
// turns, our next decision is the leaf: the net's value there is the
// rollout's (`step`'s `values`); a game that ends first is its outcome.
// `q(root)` is each choice's mean over its rollouts, our side's (+1 a win).
//
// Gathering is SelfPlay's: a request with one choice is answered here and
// its tokens join the player's next decision, except at the leaf, which is
// always asked (its value is wanted). Each decision's `battle` is its
// rollout's index. The foe's history for the net's cache is
// `World::prefill`: its summaries before its first rollout decision, the
// one-choice ones joined to the next as in training, the last `keep`;
// `t0` is the first one's position in the foe's game.
//
// The rollouts are split into fixed shards, each with its own RNG, run on
// any thread.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "env.hpp"
#include "history_sampler.hpp"

namespace ataraxos {

struct Root {
  uint8_t player;
  uint8_t n; // our choices
  std::array<pkmn_choice, MAX_CHOICES> options;
  uint32_t first_world, worlds;
  uint32_t first_rollout, rollouts;
};

struct World {
  uint32_t root;
  Guess::Source source;
  pkmn_choice_kind foe_kind;
  // the foe's prefill summaries: summary i is tokens [ends[i - 1], ends[i])
  std::vector<Token> tokens;
  std::vector<uint32_t> ends;
  std::vector<int16_t> requests; // REQUEST_FIELDS per summary
  uint32_t t0 = 0;
};

struct Rollout {
  uint32_t root, world;
  uint8_t action;
  uint16_t until; // the leaf's turn
  bool done = false;
  bool differs = false; // the world's choices for us aren't the root's
  float value = 0;
};

class Rollouts {
 public:
  // `keep`: the foe's prefill summaries kept (the net's window - 1).
  Rollouts(std::vector<Team> pool, size_t keep, size_t shards = 64, int threads = 8, bool state = true)
      : pool_(std::move(pool)), keep_(keep), shards_(shards), threads_(threads), state_(state) {}

  Pool& pool() { return pool_; }

  void clear() {
    roots_.clear();
    worlds_.clear();
    rollouts_.clear();
    nows_.clear();
    release();
  }

  // A search at player p's decision now. `updates` and `after` are p's view
  // and Bookkeeping after each update (the sampler's), `request` and
  // `options` p's request now and its Showdown choices, `binds` the binds
  // as p's request saw them. Returns the root's index.
  size_t add(uint8_t p, const Bookkeeping& now, std::span<const std::vector<Obs>> updates,
             std::span<const Bookkeeping> after, const RequestData& request, std::span<const pkmn_choice> options,
             const Binds& binds, uint32_t worlds, uint32_t reps, uint16_t depth, uint64_t seed) {
    assert(options.size() >= 1 && options.size() <= MAX_CHOICES);
    Root r{p, uint8_t(options.size()), {}, uint32_t(worlds_.size()), worlds, uint32_t(rollouts_.size()),
           worlds * reps * uint32_t(options.size())};
    std::copy(options.begin(), options.end(), r.options.begin());
    const size_t root = roots_.size();
    roots_.push_back(r);
    nows_.push_back(now);
    std::mt19937_64 rng(seed);
    Sampler sampler(pool_, now);
    std::vector<Guess> guesses(worlds);
    std::vector<uint64_t> seeds(worlds);
    for (uint32_t w = 0; w < worlds; w++) {
      guesses[w] = sampler.guess(rng);
      seeds[w] = rng();
    }
    std::vector<Sample> samples(worlds);
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
    for (uint32_t w = 0; w < worlds; w++) samples[w] = sampler.sample(guesses[w], updates, after, seeds[w]);

    const uint8_t foe = 1 - p;
    const pkmn_choice_kind ours = request.kind == RequestData::Move_    ? PKMN_CHOICE_MOVE
                                  : request.kind == RequestData::Switch ? PKMN_CHOICE_SWITCH
                                                                        : PKMN_CHOICE_PASS;
    const uint16_t turn = now.battle.turn;
    for (uint32_t w = 0; w < worlds; w++) {
      Sample& s = samples[w];
      World& x = worlds_.emplace_back();
      x.root = uint32_t(root);
      x.source = s.guess.source;
      const PKMN::Battle& b = PKMN::view(s.battle);
      x.foe_kind = b.sides[foe].stored().hp == 0 ? PKMN_CHOICE_SWITCH
                   : b.sides[p].stored().hp == 0 ? PKMN_CHOICE_PASS
                                                 : PKMN_CHOICE_MOVE;
      prefill(s, x, first_chunk_.emplace_back());
      samples_.push_back(std::move(s));
      for (uint8_t a = 0; a < r.n; a++) {
        for (uint32_t k = 0; k < reps; k++) rollouts_.push_back({uint32_t(root), uint32_t(worlds_.size() - 1), a, uint16_t(turn + depth)});
      }
      kinds_.push_back({p == 0 ? ours : x.foe_kind, p == 0 ? x.foe_kind : ours});
      binds_.push_back(binds);
    }
    return root;
  }

  // Resumes every rollout from its world, plays our forced choice and
  // gathers the first decisions.
  void start(uint64_t seed) {
    if (battles_size_ < rollouts_.size()) {
      battles_ = std::make_unique<BattleData[]>(rollouts_.size());
      battles_size_ = rollouts_.size();
    }
    const size_t n = std::max<size_t>(1, std::min(shards_, rollouts_.size()));
    shards_v_.assign(n, Shard{});
    leaf_.assign(n, {});
    for (size_t s = 0; s < n; s++) {
      shards_v_[s].begin = rollouts_.size() * s / n;
      shards_v_[s].end = rollouts_.size() * (s + 1) / n;
      shards_v_[s].rng.seed(seed + s);
    }
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
    for (size_t s = 0; s < n; s++) {
      Shard& sh = shards_v_[s];
      for (size_t i = sh.begin; i < sh.end; i++) {
        Rollout& r = rollouts_[i];
        const Root& root = roots_[r.root];
        const Sample& w = samples_[r.world];
        BattleData& d = battles_[i];
        d.track = state_;
        const Bookkeeping bks[2] = {root.player == 0 ? now_of(r) : w.foe, root.player == 0 ? w.foe : now_of(r)};
        auto durations = std::bit_cast<pkmn_gen1_chance_durations>(w.durations);
        // a fresh battle seed per rollout (the world's for the first)
        pkmn_gen1_battle b = w.battle;
        PKMN::view(b).rng = PKMN::view(w.battle).rng + i * 0x9E3779B97F4A7C15ull;
        d.resume(b, durations, kinds_[r.world], binds_[r.world], bks);
        d.chunk[1 - root.player] = first_chunk_[r.world];
        const uint8_t p = root.player;
        const pkmn_choice c = root.options[r.action];
        r.differs = d.n_option[p] != root.n || !std::equal(root.options.begin(), root.options.begin() + root.n, d.option[p].begin());
        assert(d.needs_choice(p));
        answer(r, d, p, c);
      }
      gather(s);
    }
  }

  // `picks[i]` answers decision i of the last gather; `values[i]` is the
  // decision's player's value (P(win) - P(loss)), read at the leaves.
  void step(std::span<const uint8_t> picks, std::span<const float> values) {
    std::vector<size_t> at(shards_v_.size());
    size_t total = 0;
    for (size_t s = 0; s < shards_v_.size(); s++) at[s] = total, total += shards_v_[s].decisions.size();
    assert(picks.size() == total && values.size() == total);
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
    for (size_t s = 0; s < shards_v_.size(); s++) {
      Shard& sh = shards_v_[s];
      for (size_t k = 0; k < sh.decisions.size(); k++) {
        const Decision& x = sh.decisions[k];
        Rollout& r = rollouts_[x.battle];
        BattleData& d = battles_[x.battle];
        d.asked[x.player] = false;
        if (leaf_[s][k]) {
          r.value = values[at[s] + k];
          r.done = true;
          continue;
        }
        assert(picks[at[s] + k] < x.n);
        answer(r, d, x.player, x.options[picks[at[s] + k]]);
      }
      gather(s);
    }
  }

  // No decision left: every rollout is done.
  bool done() const {
    for (const Shard& sh : shards_v_) {
      if (!sh.decisions.empty()) return false;
    }
    return true;
  }

  // Each of the root's choices' mean value over its rollouts.
  std::vector<double> q(size_t root) const {
    const Root& r = roots_[root];
    std::vector<double> sum(r.n, 0);
    std::vector<uint32_t> count(r.n, 0);
    for (uint32_t i = r.first_rollout; i < r.first_rollout + r.rollouts; i++) {
      const Rollout& x = rollouts_[i];
      assert(x.done);
      sum[x.action] += x.value;
      count[x.action]++;
    }
    for (uint8_t a = 0; a < r.n; a++) sum[a] /= std::max<uint32_t>(1, count[a]);
    return sum;
  }

  const std::vector<Root>& roots() const { return roots_; }
  const std::vector<World>& worlds() const { return worlds_; }
  const std::vector<Rollout>& rollouts() const { return rollouts_; }
  const std::vector<Shard>& shards() const { return shards_v_; }
  const std::vector<std::vector<uint8_t>>& leaves() const { return leaf_; }
  const BattleData& battle(size_t i) const { return battles_[i]; }
  int threads() const { return threads_; }
  bool state() const { return state_; }

  // Drops the samples, keeping roots, worlds and rollouts (call after
  // `start`, which reads them).
  void release() {
    samples_.clear();
    first_chunk_.clear();
    kinds_.clear();
    binds_.clear();
  }

 private:
  Pool pool_;
  size_t keep_, shards_;
  int threads_;
  bool state_;
  std::vector<Root> roots_;
  std::vector<World> worlds_;
  std::vector<Rollout> rollouts_;
  // per world, until `start`
  std::vector<Sample> samples_;
  std::vector<std::vector<Token>> first_chunk_;
  std::vector<std::array<pkmn_choice_kind, 2>> kinds_;
  std::vector<Binds> binds_;
  std::vector<Bookkeeping> nows_; // per root, until `clear`
  std::unique_ptr<BattleData[]> battles_;
  size_t battles_size_ = 0;
  std::vector<Shard> shards_v_;
  std::vector<std::vector<uint8_t>> leaf_;

  const Bookkeeping& now_of(const Rollout& r) const { return nows_[r.root]; }

  // The foe's summaries before its first rollout decision, one-choice ones
  // joined to the next, the last `keep_` (into x), and its first chunk.
  void prefill(const Sample& s, World& x, std::vector<Token>& first) const {
    const bool deciding = x.foe_kind != PKMN_CHOICE_PASS;
    const size_t m = s.ends.size() - (deciding && !s.ends.empty() ? 1 : 0);
    struct Merged {
      uint32_t begin, end; // tokens
      size_t request;
    };
    std::vector<Merged> merged;
    uint32_t begin = 0;
    for (size_t i = 0; i < m; i++) {
      std::array<pkmn_choice, MAX_CHOICES> o;
      if (showdown_choices(s.requests[i], o.data()) < 2) continue; // joins the next
      merged.push_back({begin, s.ends[i], i});
      begin = s.ends[i];
    }
    first.assign(s.tokens.begin() + begin, s.tokens.end());
    const size_t from = merged.size() > keep_ ? merged.size() - keep_ : 0;
    x.t0 = uint32_t(from);
    for (size_t i = from; i < merged.size(); i++) {
      x.tokens.insert(x.tokens.end(), s.tokens.begin() + merged[i].begin, s.tokens.begin() + merged[i].end);
      x.ends.push_back(uint32_t(x.tokens.size()));
      const size_t at = x.requests.size();
      x.requests.resize(at + REQUEST_FIELDS);
      encode(s.requests[merged[i].request], &x.requests[at]);
    }
  }

  void answer(Rollout& r, BattleData& d, uint8_t p, pkmn_choice c) {
    if (d.choose(p, c) == BattleData::Ended) {
      const uint8_t us = roots_[r.root].player;
      const pkmn_result_kind t = pkmn_result_type(d.result);
      r.value = t == PKMN_RESULT_WIN ? (us == 0 ? 1.f : -1.f) : t == PKMN_RESULT_LOSE ? (us == 0 ? -1.f : 1.f) : 0.f;
      r.done = true;
    }
  }

  // SelfPlay's gather, with the leaf.
  void gather(size_t s) {
    Shard& sh = shards_v_[s];
    std::vector<uint8_t>& leaf = leaf_[s];
    sh.decisions.clear();
    sh.tokens.clear();
    sh.lengths.clear();
    sh.requests.clear();
    sh.states.clear();
    leaf.clear();
    for (size_t i = sh.begin; i < sh.end; i++) {
      Rollout& r = rollouts_[i];
      BattleData& d = battles_[i];
      const uint8_t us = roots_[r.root].player;
      for (bool again = true; again && !r.done;) {
        again = false;
        for (uint8_t p = 0; p < 2; p++) {
          if (d.asked[p] || !d.needs_choice(p)) continue;
          const bool is_leaf = p == us && d.turn() >= r.until;
          Decision x{uint32_t(i), 0, p, d.n_option[p], {}};
          std::copy_n(d.option[p].begin(), x.n, x.options.begin());
          assert(x.n >= 1);
          if (x.n == 1 && !is_leaf) {
            answer(r, d, p, x.options[0]);
            again = true;
            break;
          }
          sh.tokens.insert(sh.tokens.end(), d.chunk[p].begin(), d.chunk[p].end());
          sh.lengths.push_back(uint32_t(d.chunk[p].size()));
          d.chunk[p].clear();
          const size_t at = sh.requests.size();
          sh.requests.resize(at + REQUEST_FIELDS);
          encode(d.request[p], &sh.requests[at]);
          if (state_) {
            const size_t st = sh.states.size();
            sh.states.resize(st + STATE_FIELDS);
            encode_state(d.bk[p], &sh.states[st]);
          }
          d.asked[p] = true;
          sh.decisions.push_back(x);
          leaf.push_back(is_leaf);
          // the leaf ends the rollout: the foe's decision with it isn't asked
          if (is_leaf) break;
        }
      }
    }
  }
};

} // namespace ataraxos
