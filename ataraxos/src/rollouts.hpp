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
//
// Joint roots (`add_joint`, CLAUDE.md "Joint search"): a Bayesian game over
// types, `a` of ours (A0 our real team, the rest guessed from what the foe
// has seen of us: Seen, Sampler::own_side) and `b` of the foe's (the
// sampler's guesses). Each type has its own history (ours as type A:
// Sampler::own; A0 is the real actor's cache) and, when it has a choice at
// the root, its root decision, which the caller runs on that type's cache
// for its policy. Rollouts come in rounds: `schedule` puts `count` in cell
// (A, B, i, j), world (A, B) is B's world with our side as A's (hidden
// counters and battle seed fresh per rollout), and both sides' root
// choices, i of A's and j of B's, are forced. The caller keeps each cell's
// values (`clear_rollouts` between rounds keeps the types).

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
  // joint: `root` a joint root, the cell's types and the foe's root choice
  bool joint = false;
  uint8_t a = 0, b = 0, foe_action = 0;
};

// One type of one side at a joint root.
struct JointType {
  uint32_t root;
  uint8_t side; // 0 ours, 1 the foe's
  bool real = false; // A0: our real team, the real actor's cache
  Guess guess;
  // its summaries before the root decision (none for A0)
  World prefill;
  // its choices at the root (1 when it waits)
  uint8_t n = 1;
  std::array<pkmn_choice, MAX_CHOICES> options{};
  // n >= 2: the root decision, run on its cache (not for A0): its tokens and
  // request; else its tokens join its first rollout decision (`first`)
  bool decides = false;
  std::vector<Token> chunk, first;
  RequestData request{};
  // the rollouts': its Bookkeeping (ours: our real one with our side as
  // this type's); ours, our side; the foe's, its world (hidden counters
  // as bookkeeping's), durations and request kind
  Bookkeeping bk;
  PKMN::Side ours{};
  PKMN::Battle world{};
  PKMN::Durations durations{};
  pkmn_choice_kind kind = PKMN_CHOICE_PASS;
};

struct JointRoot {
  uint8_t player;
  uint8_t a, b;
  uint32_t first; // types [first, first + a) ours, the next b the foe's
  pkmn_choice_kind kind; // ours
  Binds binds;
  uint16_t turn, depth;
};

inline pkmn_choice_kind choice_kind(const RequestData& r) {
  return r.kind == RequestData::Move_ ? PKMN_CHOICE_MOVE : r.kind == RequestData::Switch ? PKMN_CHOICE_SWITCH : PKMN_CHOICE_PASS;
}

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
    joints_.clear();
    types_.clear();
    release();
  }

  // The rollouts only (between joint rounds).
  void clear_rollouts() { rollouts_.clear(); }

  // A search at player p's decision now. `updates` and `after` are p's view
  // and Bookkeeping after each update (the sampler's), `request` and
  // `options` p's request now and its Showdown choices, `binds` the binds
  // as p's request saw them; `fixed`, a guess every world takes (tests).
  // Returns the root's index.
  size_t add(uint8_t p, const Bookkeeping& now, std::span<const std::vector<Obs>> updates,
             std::span<const Bookkeeping> after, const RequestData& request, std::span<const pkmn_choice> options,
             const Binds& binds, uint32_t worlds, uint32_t reps, uint16_t depth, uint64_t seed,
             const Guess* fixed = nullptr) {
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
      guesses[w] = fixed ? *fixed : sampler.guess(rng);
      seeds[w] = rng();
    }
    std::vector<Sample> samples(worlds);
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
    for (uint32_t w = 0; w < worlds; w++) samples[w] = sampler.sample(guesses[w], updates, after, seeds[w]);

    const uint8_t foe = 1 - p;
    const pkmn_choice_kind ours = choice_kind(request);
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
      prefill(s, s.ends.size() - (x.foe_kind != PKMN_CHOICE_PASS && !s.ends.empty() ? 1 : 0), x,
              first_chunk_.emplace_back());
      samples_.push_back(std::move(s));
      for (uint8_t a = 0; a < r.n; a++) {
        for (uint32_t k = 0; k < reps; k++) rollouts_.push_back({uint32_t(root), uint32_t(worlds_.size() - 1), a, uint16_t(turn + depth)});
      }
      kinds_.push_back({p == 0 ? ours : x.foe_kind, p == 0 ? x.foe_kind : ours});
      binds_.push_back(binds);
    }
    return root;
  }

  // A joint root at player p's decision now (add's arguments), with `a` of
  // our types and `b` of the foe's. `fixed`: the foe's types all take it
  // (tests). Returns the joint root's index.
  size_t add_joint(uint8_t p, const Bookkeeping& now, std::span<const std::vector<Obs>> updates,
                   std::span<const Bookkeeping> after, const RequestData& request,
                   std::span<const pkmn_choice> options, const Binds& binds, uint8_t a, uint8_t b, uint16_t depth,
                   uint64_t seed, const Guess* fixed = nullptr) {
    assert(a >= 1 && b >= 1 && options.size() >= 1 && options.size() <= MAX_CHOICES);
    const uint8_t foe = 1 - p;
    const size_t root = joints_.size();
    joints_.push_back({p, a, b, uint32_t(types_.size()), choice_kind(request), binds, now.battle.turn, depth});
    std::mt19937_64 rng(seed);
    Sampler foes(pool_, now);
    std::vector<Guess> gb(b);
    std::vector<uint64_t> seeds(b);
    for (uint8_t k = 0; k < b; k++) {
      gb[k] = fixed ? *fixed : foes.guess(rng);
      seeds[k] = rng();
    }
    std::vector<Sample> samples(b);
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
    for (uint8_t k = 0; k < b; k++) samples[k] = foes.sample(gb[k], updates, after, seeds[k], false);
    // our side as the foe has seen it is the same in every world
    Sampler ours(pool_, samples[0].foe);
    const Seen& seen = samples[0].seen;
    std::vector<Guess> ga(a);
    for (uint8_t k = 1; k < a; k++) ga[k] = ours.guess(rng);
    std::vector<History> histories(a);
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
    for (uint8_t k = 1; k < a; k++) histories[k] = ours.own(ga[k], seen, updates, after);

    const PKMN::Battle& w0 = PKMN::view(samples[0].battle);
    for (uint8_t k = 0; k < a; k++) {
      JointType& t = types_.emplace_back();
      t.root = uint32_t(root);
      t.side = 0;
      t.real = k == 0;
      t.guess = ga[k];
      t.ours = k == 0 ? now.battle.sides[p] : ours.own_side(now.battle.sides[p], seen, ga[k]);
      t.bk = now;
      t.bk.battle.sides[p] = t.ours;
      t.kind = joints_[root].kind;
      t.prefill.root = uint32_t(root);
      t.prefill.source = t.guess.source;
      t.prefill.foe_kind = t.kind;
      if (k == 0) {
        t.request = request;
        t.n = uint8_t(options.size());
        std::copy(options.begin(), options.end(), t.options.begin());
        continue;
      }
      PKMN::Battle w = w0;
      w.sides[p] = t.ours;
      t.request = request_data(w, p, t.kind, binds.of(w, p));
      t.n = showdown_choices(t.request, t.options.data());
      t.decides = t.n >= 2;
      const History& h = histories[k];
      prefill(h, h.ends.size() - (t.decides && !h.ends.empty() ? 1 : 0), t.prefill, t.decides ? t.chunk : t.first);
    }
    for (uint8_t k = 0; k < b; k++) {
      Sample& s = samples[k];
      JointType& t = types_.emplace_back();
      t.root = uint32_t(root);
      t.side = 1;
      t.guess = s.guess;
      t.world = PKMN::view(s.battle);
      t.durations = s.durations;
      t.bk = s.foe;
      const PKMN::Battle& w = t.world;
      t.kind = w.sides[foe].stored().hp == 0 ? PKMN_CHOICE_SWITCH
               : w.sides[p].stored().hp == 0 ? PKMN_CHOICE_PASS
                                             : PKMN_CHOICE_MOVE;
      t.prefill.root = uint32_t(root);
      t.prefill.source = t.guess.source;
      t.prefill.foe_kind = t.kind;
      if (t.kind != PKMN_CHOICE_PASS) {
        t.request = request_data(w, foe, t.kind, binds.of(w, foe));
        t.n = showdown_choices(t.request, t.options.data());
      } else {
        t.request.kind = RequestData::Wait;
      }
      t.decides = t.n >= 2;
      prefill(s, s.ends.size() - (t.kind != PKMN_CHOICE_PASS && !s.ends.empty() ? 1 : 0), t.prefill,
              t.decides ? t.chunk : t.first);
    }
    return root;
  }

  // `count` rollouts in cell (a, b, i, j) of joint root `root`: our type a's
  // choice i and the foe's type b's choice j.
  void schedule(size_t root, uint8_t a, uint8_t b, uint8_t i, uint8_t j, uint32_t count) {
    const JointRoot& J = joints_[root];
    assert(a < J.a && b < J.b && i < types_[J.first + a].n && j < types_[J.first + J.a + b].n);
    Rollout r{uint32_t(root), 0, i, uint16_t(J.turn + J.depth)};
    r.joint = true;
    r.a = a;
    r.b = b;
    r.foe_action = j;
    rollouts_.insert(rollouts_.end(), count, r);
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
        if (r.joint) {
          start_joint(i, seed);
          continue;
        }
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
  const std::vector<JointRoot>& joints() const { return joints_; }
  const std::vector<JointType>& types() const { return types_; }
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
  std::vector<JointRoot> joints_;
  std::vector<JointType> types_;
  std::unique_ptr<BattleData[]> battles_;
  size_t battles_size_ = 0;
  std::vector<Shard> shards_v_;
  std::vector<std::vector<uint8_t>> leaf_;

  const Bookkeeping& now_of(const Rollout& r) const { return nows_[r.root]; }

  uint8_t us_of(const Rollout& r) const { return r.joint ? joints_[r.root].player : roots_[r.root].player; }

  // Joint rollout i: world (A, B), both root choices forced.
  void start_joint(size_t i, uint64_t seed) {
    Rollout& r = rollouts_[i];
    const JointRoot& J = joints_[r.root];
    const JointType& A = types_[J.first + r.a];
    const JointType& B = types_[J.first + J.a + r.b];
    const uint8_t p = J.player, foe = 1 - p;
    BattleData& d = battles_[i];
    d.track = state_;
    PKMN::Battle w = B.world;
    w.sides[p] = A.ours;
    w.rng = B.world.rng + (seed * 0x100000001B3ull + i) * 0x9E3779B97F4A7C15ull;
    pkmn_gen1_battle b = std::bit_cast<pkmn_gen1_battle>(w);
    auto durations = std::bit_cast<pkmn_gen1_chance_durations>(B.durations);
    MCTS::randomize_hidden_variables(b, durations);
    const Bookkeeping bks[2] = {p == 0 ? A.bk : B.bk, p == 0 ? B.bk : A.bk};
    std::array<pkmn_choice_kind, 2> kinds;
    kinds[p] = J.kind;
    kinds[foe] = B.kind;
    d.resume(b, durations, kinds, J.binds, bks);
    d.chunk[p] = A.first;
    d.chunk[foe] = B.first;
    const auto same = [&](const JointType& t, uint8_t q) {
      if (!d.needs_choice(q)) return t.kind == PKMN_CHOICE_PASS;
      return d.n_option[q] == t.n && std::equal(t.options.begin(), t.options.begin() + t.n, d.option[q].begin());
    };
    r.differs = !same(A, p) || !same(B, foe);
    assert(d.needs_choice(p));
    answer(r, d, p, d.option[p][std::min<uint8_t>(r.action, d.n_option[p] - 1)]);
    if (!r.done && d.needs_choice(foe)) answer(r, d, foe, d.option[foe][std::min<uint8_t>(r.foe_action, d.n_option[foe] - 1)]);
  }

  // A player's summaries before its first rollout decision (the first `m`
  // of s), one-choice ones joined to the next, the last `keep_` (into x),
  // and the tokens after them (its first chunk).
  void prefill(const History& s, size_t m, World& x, std::vector<Token>& first) const {
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
      const uint8_t us = us_of(r);
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
      const uint8_t us = us_of(r);
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
