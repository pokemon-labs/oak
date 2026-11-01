// The history sampler (the user's design, 2026-11-01): a world for an
// Ataraxos rollout, made up from our Bookkeeping and a guess at the foe's
// team, with the foe's history for the net to read before it plays the
// foe's side. Nothing is replayed: the world is our bookkeeping with the
// foe's unrevealed species and moves filled in (every set maxes its DVs and
// stat experience, so a set is its species, level and moves).
//
// The guess (safari's TeamPredictor, ~/safari/src/teams.py), strongest
// first:
// - a pool team every revealed Pokémon fits (species, level, its revealed
//   moves among the set's; one set each, a maximum matching as birch's
//   determinizer), drawn by the pool's prior: its revealed Pokémon take
//   their sets, the unseen ones its leftover sets;
// - no team fits: safari's SetDict. Each pool set weighs the sum of
//   prior / |team| over the teams with it; a revealed Pokémon draws a set of
//   its species that fits, an unseen slot any set of a species not used yet;
// - no set of its species fits: its revealed moves and random moves from
//   the OU learnset (oak's fill_random_moveset). Only so a world can always
//   be built.
// The prior is rank decay over the pool's order (teams1500 is by peak
// ladder rating, best first: safari's first_to_last_ratio), or any weights
// (`set_prior`, e.g. Exp3's average over the run).
//
// The foe's history: our own view's bytes, update by update, read as the
// foe (tokenize with viewer = the foe, so every ident flips), not what
// Showdown would have sent it. HP is made consistent: the foe's own HP is
// exact in its lines, each Pokémon's a running estimate from the percent
// we saw (Bookkeeping's rule, bk::hp_from_percent), out of its max HP; ours
// becomes a percent of 100. The foe's request after each update is
// request_data on our bookkeeping after that update with the same guess
// filled in and the foe's HP set to the same estimates, so its requests,
// its lines and the world agree. A Wait isn't a decision: its lines join
// the next summary, as BattleData's chunk does.
//
// The numbering: our side by party position, the foe's by first appearance
// (our view's, and bookkeeping's storage), on both sides of the history and
// the world. A rollout's Views start as `identity_view` to keep it.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <format/ou/data.h>
#include <libpkmn/data.h>
#include <libpkmn/init.h>
#include <libpkmn/pkmn.h>
#include <search/durations.h>

#include "bookkeeping.hpp"
#include "obs.hpp"
#include "request_data.hpp"
#include "teams.hpp"
#include "tokens.hpp"
#include "view.hpp"

namespace ataraxos {

namespace sampler {

using D = PKMN::Data::Move;

inline bool has(const std::array<D, 4>& moves, D m) { return std::find(moves.begin(), moves.end(), m) != moves.end(); }

// Whether set `s` can be revealed Pokémon `p` (bookkeeping's stored one).
inline bool fits(const Set& s, const PKMN::Pokemon& p) {
  if (s.species != p.species || s.level != p.level) return false;
  for (const PKMN::MoveSlot& m : p.moves) {
    if (m.id != D::None && m.id != D::Struggle && !has(s.moves, m.id)) return false;
  }
  return true;
}

// Kuhn's augmenting path from revealed Pokémon i; `owner[j]` is the
// Pokémon set j is matched to, or -1 (birch's determinizer).
inline bool augment(uint8_t i, const std::array<uint8_t, 6>& fit, uint8_t& tried, std::array<int8_t, 6>& owner) {
  for (uint8_t j = 0; j < 6; ++j) {
    if (!(fit[i] >> j & 1) || (tried >> j & 1)) continue;
    tried |= uint8_t(1 << j);
    if (owner[j] < 0 || augment(uint8_t(owner[j]), fit, tried, owner)) {
      owner[j] = int8_t(i);
      return true;
    }
  }
  return false;
}

// `p`'s moves with the empty slots filled from set `s`, in its order.
inline std::array<D, 4> complete(const PKMN::Pokemon& p, const Set& s) {
  std::array<D, 4> out{};
  for (uint8_t i = 0; i < 4; i++) out[i] = p.moves[i].id;
  size_t next = 0;
  for (D& m : out) {
    if (m != D::None) continue;
    while (next < 4 && (s.moves[next] == D::None || has(out, s.moves[next]))) next++;
    if (next == 4) break;
    m = s.moves[next++];
  }
  return out;
}

// Random OU moves in `moves`' empty slots (oak's fill_random_moveset).
template <class RNG> void fill_random(PKMN::Data::Species species, std::array<D, 4>& moves, RNG& rng) {
  using OU = Format::OU;
  std::vector<D> rest;
  for (size_t i = 0; i < OU::move_pool_size(species); i++) {
    const D m = OU::move_pool(species)[i];
    if (!has(moves, m)) rest.push_back(m);
  }
  std::shuffle(rest.begin(), rest.end(), rng);
  for (D& m : moves) {
    if (m != D::None || rest.empty()) continue;
    m = rest.back();
    rest.pop_back();
  }
}

} // namespace sampler

// The pool and its prior.
struct Pool {
  std::vector<Team> teams;
  std::vector<double> prior; // sums to 1
  // Each distinct pool set and its weight, sum(prior / |team|) over the
  // teams with it (safari's SetDict).
  std::vector<Set> sets;
  std::vector<double> set_weight;

  // `ratio` is the last team's prior over the first's, geometric between
  // (safari's first_to_last_ratio; 1 is uniform).
  explicit Pool(std::vector<Team> t, double ratio = 0.1) : teams(std::move(t)) {
    assert(!teams.empty() && ratio > 0 && ratio <= 1);
    std::vector<double> w(teams.size());
    const double step = teams.size() > 1 ? std::log(ratio) / double(teams.size() - 1) : 0;
    for (size_t i = 0; i < w.size(); i++) w[i] = std::exp(step * double(i));
    set_prior(w);
  }

  // Any non-negative weights, one per team.
  void set_prior(std::span<const double> w) {
    assert(w.size() == teams.size());
    double total = 0;
    for (double x : w) total += x;
    assert(total > 0);
    prior.assign(w.begin(), w.end());
    for (double& x : prior) x /= total;
    sets.clear();
    set_weight.clear();
    for (size_t t = 0; t < teams.size(); t++) {
      for (const Set& s : teams[t]) {
        size_t j = 0;
        while (j < sets.size() && !(sets[j].species == s.species && sets[j].level == s.level && sets[j] == s)) j++;
        if (j == sets.size()) {
          sets.push_back(s);
          set_weight.push_back(0);
        }
        set_weight[j] += prior[t] / double(teams[t].size());
      }
    }
  }
};

// A guess at the foe's sets, by bookkeeping's numbering (the k-th Pokémon
// to appear in sets[k - 1]); past the team, species None.
struct Guess {
  enum Source : uint8_t { FromTeam, FromSets, Random };
  std::array<Set, 6> sets{};
  Source source = FromTeam; // the weakest step any Pokémon took
  uint32_t team = UINT32_MAX; // the pool team, when every set is one team's
};

// One world: the battle to roll out from, and the foe's history.
struct Sample {
  Guess guess;
  pkmn_gen1_battle battle;
  PKMN::Durations durations;
  // The foe's Bookkeeping now, for its state tokens: the world with our side
  // cut to what the foe has seen (our Pokémon that came in, the moves they
  // used) and our HP as its estimate from our percent.
  Bookkeeping foe;
  // The foe's summaries: summary i is tokens [ends[i - 1], ends[i]) (from 0)
  // and requests[i]. The last is the foe's decision now unless it waits;
  // tokens past the last end then join its next one.
  std::vector<Token> tokens;
  std::vector<uint32_t> ends;
  std::vector<RequestData> requests;
};

// A View that keeps the numbering as it is (every slot already its number).
inline View identity_view(uint8_t p) {
  View v{p};
  for (uint8_t s = 1; s <= 6; s++) v.number[s] = s;
  v.seen = 6;
  return v;
}

// The guesses at one decision: the pool teams that fit what bookkeeping has
// seen of the foe are found once. Holds `pool` and `now` by reference.
class Sampler {
 public:
  Sampler(const Pool& pool, const Bookkeeping& now) : pool_(pool), now_(now), foe_(uint8_t(1 - now.player)) {
    const PKMN::Side& side = now.battle.sides[foe_];
    teamsize_ = now.teamsize[foe_];
    for (uint8_t k = 0; k < 6; k++) {
      if (side.pokemon[k].species != PKMN::Data::Species::None) revealed_ = uint8_t(k + 1);
    }
    for (uint32_t t = 0; t < pool.teams.size(); t++) {
      const Team& team = pool.teams[t];
      const size_t n = std::min<size_t>(team.size(), 6);
      std::array<uint8_t, 6> fit{};
      for (uint8_t k = 0; k < revealed_; k++) {
        for (uint8_t j = 0; j < n; j++) fit[k] |= uint8_t(sampler::fits(team[j], side.pokemon[k]) << j);
      }
      std::array<int8_t, 6> owner{-1, -1, -1, -1, -1, -1};
      bool all = true;
      for (uint8_t k = 0; k < revealed_ && all; k++) {
        uint8_t tried = 0;
        all = fit[k] && sampler::augment(k, fit, tried, owner);
      }
      if (!all) continue;
      matches_.push_back({t, owner});
      cdf_.push_back((cdf_.empty() ? 0 : cdf_.back()) + pool.prior[t]);
    }
  }

  // The pool teams that fit.
  size_t matches() const { return matches_.size(); }
  bool matches(uint32_t team) const {
    return std::any_of(matches_.begin(), matches_.end(), [&](const Match& m) { return m.team == team; });
  }

  template <class RNG> Guess guess(RNG& rng) {
    Guess g;
    if (!matches_.empty() && cdf_.back() > 0) {
      const double u = std::uniform_real_distribution<double>(0, cdf_.back())(rng);
      const size_t i = std::min<size_t>(std::upper_bound(cdf_.begin(), cdf_.end(), u) - cdf_.begin(), cdf_.size() - 1);
      return from_team(matches_[i], rng);
    }
    g.source = Guess::FromSets;
    const PKMN::Side& side = now_.battle.sides[foe_];
    for (uint8_t k = 0; k < revealed_; k++) {
      const PKMN::Pokemon& p = side.pokemon[k];
      const int j = draw_set(rng, [&](const Set& s) { return sampler::fits(s, p); });
      if (j >= 0) {
        g.sets[k] = pool_.sets[size_t(j)];
        continue;
      }
      g.source = Guess::Random;
      Set& s = g.sets[k];
      s.species = p.species;
      s.level = p.level;
      for (uint8_t i = 0; i < 4; i++) s.moves[i] = p.moves[i].id == sampler::D::Struggle ? sampler::D::None : p.moves[i].id;
      sampler::fill_random(s.species, s.moves, rng);
    }
    fill_unseen(g, revealed_, rng);
    return g;
  }

  // The world and the foe's history for guess `g`. `updates` are our view's
  // updates and `after` our Bookkeeping after each (before the request that
  // follows it), from the game's start; `seed` the world's RNG.
  Sample sample(const Guess& g, std::span<const std::vector<Obs>> updates, std::span<const Bookkeeping> after,
                uint64_t seed) const {
    assert(updates.size() == after.size());
    const uint8_t us = now_.player, foe = foe_;
    Sample out;
    out.guess = g;

    PKMN::Battle world = now_.battle;
    fill(world, g);
    // max HP by ident slot, both sides
    std::array<std::array<uint16_t, 7>, 2> max{};
    for (uint8_t k = 1; k <= 6; k++) {
      max[us][k] = world.sides[us].pokemon[k - 1].stats.hp;
      max[foe][k] = world.sides[foe].pokemon[k - 1].stats.hp;
    }
    // the running estimates (0: not seen yet), by ident slot
    std::array<std::array<uint16_t, 7>, 2> est{};
    std::array<bool, 7> seen{};
    std::array<std::array<sampler::D, 4>, 7> used{};

    std::vector<uint8_t> bytes;
    for (size_t t = 0; t < updates.size(); t++) {
      bytes.clear();
      bool over = false;
      for (const Obs& o : updates[t]) {
        Obs x = o;
        rewrite(x, max, est);
        over |= x.type == ArgType::Win || x.type == ArgType::Tie;
        const uint8_t id = x.data[0];
        if ((x.type == ArgType::Switch || x.type == ArgType::Move) && id && ident_player(id) == us) {
          const uint8_t s = ident_slot(id);
          seen[s] = true;
          if (x.type == ArgType::Move) {
            const auto m = sampler::D(x.data[1]);
            auto& u = used[s];
            if (!sampler::has(u, m)) {
              for (auto& slot : u) {
                if (slot == sampler::D::None) {
                  slot = m;
                  break;
                }
              }
            }
          }
        }
        bytes.push_back(uint8_t(x.type));
        bytes.insert(bytes.end(), x.data, x.data + x.size());
      }
      bytes.push_back(0);
      tokenize(bytes, foe, out.tokens);
      if (over) break;

      PKMN::Battle b = after[t].battle;
      fill(b, g);
      set_hp(b.sides[foe], est[foe]);
      const pkmn_choice_kind kind = b.sides[foe].stored().hp == 0  ? PKMN_CHOICE_SWITCH
                                    : b.sides[us].stored().hp == 0 ? PKMN_CHOICE_PASS
                                                                   : PKMN_CHOICE_MOVE;
      if (kind == PKMN_CHOICE_PASS) continue;
      out.requests.push_back(request_data(b, foe, kind, Bind{}));
      out.ends.push_back(uint32_t(out.tokens.size()));
    }

    set_hp(world.sides[foe], est[foe]);
    out.durations = now_.durations;

    // The foe's bookkeeping: our side as it has seen it.
    out.foe = Bookkeeping(foe, now_.teamsize, world.sides[foe]);
    out.foe.battle = world;
    out.foe.durations = now_.durations;
    PKMN::Side& ours = out.foe.battle.sides[us];
    for (uint8_t s = 1; s <= 6; s++) {
      PKMN::Pokemon& p = ours.pokemon[s - 1];
      if (!seen[s]) {
        p = PKMN::Pokemon{};
        for (uint8_t& o : ours.order) o = o == s ? 0 : o;
        continue;
      }
      if (est[us][s] || p.hp == 0) p.hp = est[us][s];
      for (PKMN::MoveSlot& m : p.moves) {
        if (!sampler::has(used[s], m.id)) m = PKMN::MoveSlot{};
      }
    }
    if (ours.order[0] && !ours.active.volatiles.transform()) {
      for (PKMN::MoveSlot& m : ours.active.moves) {
        if (!sampler::has(used[ours.order[0]], m.id)) m = PKMN::MoveSlot{};
      }
    }

    world.rng = seed;
    out.battle = std::bit_cast<pkmn_gen1_battle>(world);
    auto durations = std::bit_cast<pkmn_gen1_chance_durations>(out.durations);
    MCTS::randomize_hidden_variables(out.battle, durations);
    return out;
  }

  // The foe's side of `b` (a bookkeeping's battle) with `g` filled in: the
  // revealed Pokémon's empty move slots from their sets (at full PP), the
  // unseen Pokémon from theirs (at full HP), in the order's empty places.
  void fill(PKMN::Battle& b, const Guess& g) const {
    PKMN::Side& side = b.sides[foe_];
    for (uint8_t k = 0; k < 6; k++) {
      PKMN::Pokemon& p = side.pokemon[k];
      const Set& s = g.sets[k];
      if (s.species == PKMN::Data::Species::None) continue;
      if (p.species != PKMN::Data::Species::None) {
        Set full = s;
        full.species = p.species;
        full.level = p.level;
        full.moves = sampler::complete(p, s);
        const PKMN::Pokemon ref = PKMN::Init::init_pokemon(full);
        for (uint8_t i = 0; i < 4; i++) {
          if (p.moves[i].id == sampler::D::None) p.moves[i] = ref.moves[i];
        }
        continue;
      }
      if (k >= teamsize_) continue;
      p = PKMN::Init::init_pokemon(s);
      for (uint8_t& o : side.order) {
        if (o == 0) {
          o = uint8_t(k + 1);
          break;
        }
      }
    }
    if (side.order[0] && !side.active.volatiles.transform()) {
      const PKMN::Pokemon& p = side.stored();
      for (uint8_t i = 0; i < 4; i++) {
        if (side.active.moves[i].id == sampler::D::None) side.active.moves[i] = p.moves[i];
      }
    }
  }

 private:
  struct Match {
    uint32_t team;
    std::array<int8_t, 6> owner; // set j -> revealed Pokémon, or -1
  };

  const Pool& pool_;
  const Bookkeeping& now_;
  uint8_t foe_;
  uint8_t teamsize_ = 6;
  uint8_t revealed_ = 0;
  std::vector<Match> matches_;
  std::vector<double> cdf_;

  template <class RNG> Guess from_team(const Match& m, RNG& rng) const {
    Guess g;
    g.team = m.team;
    const Team& team = pool_.teams[m.team];
    const size_t n = std::min<size_t>(team.size(), 6);
    for (uint8_t j = 0; j < n; j++) {
      if (m.owner[j] >= 0) g.sets[size_t(m.owner[j])] = team[j];
    }
    uint8_t k = revealed_;
    for (uint8_t j = 0; j < n && k < teamsize_; j++) {
      if (m.owner[j] < 0) g.sets[k++] = team[j];
    }
    if (k < teamsize_) {
      g.source = Guess::FromSets;
      g.team = UINT32_MAX;
      fill_unseen(g, k, rng);
    }
    return g;
  }

  // Sets for the unseen Pokémon from `k` on, species not used yet.
  template <class RNG> void fill_unseen(Guess& g, uint8_t k, RNG& rng) const {
    for (; k < teamsize_; k++) {
      const int j = draw_set(rng, [&](const Set& s) {
        return std::none_of(g.sets.begin(), g.sets.end(), [&](const Set& t) { return t.species == s.species; });
      });
      if (j < 0) break;
      g.sets[k] = pool_.sets[size_t(j)];
    }
  }

  // A pool set `ok` allows, by its weight; -1 if none.
  template <class RNG, class F> int draw_set(RNG& rng, F ok) const {
    double total = 0;
    for (size_t j = 0; j < pool_.sets.size(); j++) total += ok(pool_.sets[j]) ? pool_.set_weight[j] : 0;
    if (total <= 0) return -1;
    double u = std::uniform_real_distribution<double>(0, total)(rng);
    int last = -1;
    for (size_t j = 0; j < pool_.sets.size(); j++) {
      if (!ok(pool_.sets[j])) continue;
      last = int(j);
      u -= pool_.set_weight[j];
      if (u < 0) return int(j);
    }
    return last;
  }

  // Each foe Pokémon seen so far gets its running estimate; a fainted one
  // no status (libpkmn clears it, bookkeeping keeps it).
  static void set_hp(PKMN::Side& side, const std::array<uint16_t, 7>& est) {
    for (uint8_t k = 1; k <= 6; k++) {
      PKMN::Pokemon& p = side.pokemon[k - 1];
      if (p.species != PKMN::Data::Species::None && (est[k] || p.hp == 0)) p.hp = est[k];
      if (p.hp == 0) p.status = PKMN::Data::Status(0);
    }
  }

  // One message of our view as the foe's: its own HP exact (the running
  // estimate out of its max), ours a percent of 100. `est` follows both.
  void rewrite(Obs& o, const std::array<std::array<uint16_t, 7>, 2>& max,
               std::array<std::array<uint16_t, 7>, 2>& est) const {
    uint8_t* d = nullptr;
    if (o.type == ArgType::Faint) {
      // Explosion's user faints with no Damage line
      est[ident_player(o.data[0])][ident_slot(o.data[0])] = 0;
      return;
    }
    switch (o.type) {
    case ArgType::Switch: d = o.data + 3; break;
    case ArgType::Damage:
    case ArgType::Heal:
    case ArgType::SetHP: d = o.data + 1; break;
    default: return;
    }
    const uint8_t id = o.data[0], s = ident_player(id), k = ident_slot(id);
    const uint16_t hp = uint16_t(d[0] | d[1] << 8), m = uint16_t(d[2] | d[3] << 8);
    uint16_t nhp, nmax;
    if (s == foe_) {
      // a first sight starts full, as bookkeeping's does
      const uint16_t full = max[s][k];
      const uint16_t prev = est[s][k] ? est[s][k] : full;
      nhp = m ? bk::hp_from_percent(prev, full, hp, m) : hp;
      nmax = full;
    } else {
      const uint16_t prev = est[s][k] ? est[s][k] : m;
      nhp = view_percent(hp, m);
      nmax = m ? 100 : 0;
      // the foe's estimate of ours
      est[s][k] = m ? bk::hp_from_percent(prev, m, nhp, 100) : hp;
      d[0] = uint8_t(nhp), d[1] = uint8_t(nhp >> 8), d[2] = uint8_t(nmax), d[3] = uint8_t(nmax >> 8);
      return;
    }
    est[s][k] = nhp;
    d[0] = uint8_t(nhp), d[1] = uint8_t(nhp >> 8), d[2] = uint8_t(nmax), d[3] = uint8_t(nmax >> 8);
  }
};

} // namespace ataraxos
