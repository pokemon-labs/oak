// Checks src/rollouts.hpp on self-play (uniform Showdown choices, as
// check_history_sampler): at one decision in EVERY of each player, a search
// of WORLDS worlds and REPS rollouts per choice, played out with uniform
// picks and value 0 at the leaves. Counts:
// - "differs": rollouts whose world gives us choices other than the root's
//   (bookkeeping's mismatches, as check_history_sampler's draws);
// - "leaf too early": a leaf before its turn (must be 0);
// - "not done": a rollout with no value once no decision is left (0);
// - how rollouts end (a leaf, the game's end) and the steps they take.
//
//   check_rollouts [GAMES] [SEED] [POOL] [WORLDS] [REPS] [DEPTH] [EVERY]

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>

#include "battle_data.hpp"
#include "rollouts.hpp"
#include "teams.hpp"

using namespace ataraxos;

int main(int argc, char** argv) {
  const size_t games = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 50;
  std::mt19937_64 rng(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1);
  const std::string path = argc > 3 ? argv[3] : "teams/selfplay/oak1500.txt";
  const uint32_t worlds = argc > 4 ? uint32_t(std::strtoul(argv[4], nullptr, 10)) : 4;
  const uint32_t reps = argc > 5 ? uint32_t(std::strtoul(argv[5], nullptr, 10)) : 2;
  const uint16_t depth = argc > 6 ? uint16_t(std::strtoul(argv[6], nullptr, 10)) : 10;
  const size_t every = argc > 7 ? std::max<size_t>(1, std::strtoul(argv[7], nullptr, 10)) : 5;
  size_t errors = 0;
  std::vector<Team> all = load_teams({path}, errors);
  std::vector<Team> games_teams(all.begin() + long(all.size() / 2), all.end());
  all.resize(all.size() / 2);
  Rollouts ro(all, 31, 16, 4, true);
  std::map<std::string, size_t> counts;

  size_t decision = 0;
  for (size_t g = 0; g < games; g++) {
    BattleData bd;
    bd.track = true;
    bd.keep_after = true;
    bd.reset(init_battle(rng(), games_teams[rng() % games_teams.size()], games_teams[rng() % games_teams.size()]));
    for (int step = 0; step < 2000 && !bd.over(); step++) {
      for (uint8_t p = 0; p < 2; p++) {
        if (!bd.needs_choice(p) || bd.n_option[p] < 2 || decision++ % every) continue;
        ro.clear();
        ro.add(p, bd.bk[p], bd.history[p], bd.after[p], bd.request[p], {bd.option[p].data(), bd.n_option[p]}, bd.binds,
               worlds, reps, depth, rng());
        ro.start(rng());
        ro.release();
        counts["searches"]++;
        size_t steps = 0;
        while (!ro.done()) {
          size_t D = 0;
          for (const Shard& sh : ro.shards()) D += sh.decisions.size();
          std::vector<uint8_t> picks;
          for (size_t s = 0; s < ro.shards().size(); s++) {
            const Shard& sh = ro.shards()[s];
            for (size_t k = 0; k < sh.decisions.size(); k++) {
              const Decision& x = sh.decisions[k];
              if (ro.leaves()[s][k] && ro.battle(x.battle).turn() < ro.rollouts()[x.battle].until) counts["leaf too early"]++;
              picks.push_back(uint8_t(rng() % x.n));
            }
          }
          ro.step(picks, std::vector<float>(D, 0.f));
          steps++;
        }
        counts["steps"] += steps;
        for (const Rollout& r : ro.rollouts()) {
          counts["rollouts"]++;
          if (!r.done) counts["not done"]++;
          if (r.differs) counts["differs"]++;
          counts[ro.battle(&r - ro.rollouts().data()).over() ? "ended: game over" : "ended: leaf"]++;
        }
      }
      for (uint8_t p = 0; p < 2; p++) {
        if (!bd.needs_choice(p)) continue;
        if (bd.choose(p, bd.option[p][rng() % bd.n_option[p]]) != BattleData::Held) break;
      }
    }
  }
  counts.try_emplace("leaf too early", 0);
  counts.try_emplace("not done", 0);
  for (const auto& [what, n] : counts) std::printf("%-30s %zu\n", what.c_str(), n);
  return 0;
}
