// Checks src/pi4.hpp (2026-11-24, docs/pi4.md): random self-play in the env
// (SelfPlay with state), and at every gather, for each battle with a
// decision and each viewer, the perfect-info Bookkeeping must
//   - match the true battle by oak's compare_battles with only the tracked
//     fields left out of the key (last_selected_move, last_used_move,
//     last_moves): sets, PP, exact HP, stats, boosts, volatiles, the
//     Durations and last_damage exactly;
//   - have every Pokémon of the foe's team, and each one the viewer has
//     seen under the viewer's number;
//   - carry each side's tracked fields from the other player's Bookkeeping.
// Also how often the tracked last_used_move differs from libpkmn's and how
// often encode_last's desync bit is on.
//
//   check_pi4 [GAMES] [SEED] [TEAM_FILE|-]
// TEAM_FILE defaults to teams/selfplay/oak1500.txt; - plays truly random
// teams, whose repeated species compare_battles' bench (matched by species)
// can't tell apart, so its mismatches there are the check's.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <libpkmn/client.h>
#include <libpkmn/strings.h>

#include "env.hpp"
#include "pi4.hpp"
#include "teams.hpp"

using namespace ataraxos;

int main(int argc, char** argv) {
  const size_t games = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 200;
  const uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
  std::vector<Team> pool;
  const std::string file = argc > 3 ? argv[3] : "teams/selfplay/oak1500.txt";
  if (file != "-") {
    size_t errors = 0;
    pool = load_teams({file}, errors);
    if (pool.empty()) {
      std::printf("no team parses in %s\n", file.c_str());
      return 2;
    }
  }
  const size_t battles = 64;
  SelfPlay env(battles, seed, 8, 1, pool, 300, true);
  env.set_light(true);
  env.reset();
  std::mt19937_64 rng(seed);
  PKMN::Battle key{};
  for (uint8_t s = 0; s < 2; s++) {
    key.sides[s].last_selected_move = PKMN::Data::Move(1);
    key.sides[s].last_used_move = PKMN::Data::Move(1);
    key.last_moves[s] = {1, 1};
  }
  const PKMN::Durations dkey{};
  size_t rows = 0, bad = 0, printed = 0, used_diff = 0, desync = 0, done = 0;
  std::vector<int16_t> out(2 * PI4_FIELDS);
  while (done < games) {
    for (const Shard& sh : env.shards()) {
      uint32_t last = UINT32_MAX;
      for (const Decision& x : sh.decisions) {
        if (x.battle == last) continue;
        last = x.battle;
        const BattleData& d = env.battle(x.battle);
        const PKMN::Battle& truth = d.state();
        const PKMN::Durations truth_d =
            PKMN::view(*pkmn_gen1_battle_options_chance_durations(const_cast<pkmn_gen1_battle_options*>(&d.options)));
        rows++;
        encode_pi4(d, out.data());
        for (uint8_t v = 0; v < 2; v++) {
          const uint8_t foe = 1 - v;
          const Bookkeeping pk = pi4_bookkeeping(d, v);
          std::string reason;
          bool ok = PKMN::Client::compare_battles(pk.battle, pk.durations, truth, truth_d, key, dkey, reason);
          for (uint8_t k = 1; ok && k <= 6; k++) {
            const PKMN::Data::Species seen = d.bk[v].battle.sides[foe].pokemon[k - 1].species;
            if (seen != PKMN::Data::Species::None && pk.battle.sides[foe].pokemon[k - 1].species != seen) {
              ok = false;
              reason = "number " + std::to_string(k) + " isn't the viewer's " + PKMN::species_string(seen);
            }
          }
          for (const PKMN::Pokemon& p : truth.sides[foe].pokemon) {
            bool found = p.species == PKMN::Data::Species::None;
            for (const PKMN::Pokemon& q : pk.battle.sides[foe].pokemon) found |= q.species == p.species;
            if (ok && !found) {
              ok = false;
              reason = "missing " + PKMN::species_string(p.species);
            }
          }
          for (uint8_t s = 0; s < 2; s++) {
            const Bookkeeping& other = d.bk[1 - s];
            if (ok && (pk.battle.sides[s].last_selected_move != other.battle.sides[s].last_selected_move ||
                       pk.unshown[s] != other.unshown[s] ||
                       pk.battle.last_moves[s].counterable != other.battle.last_moves[s].counterable)) {
              ok = false;
              reason = "side " + std::to_string(s) + "'s tracked fields";
            }
            used_diff += pk.battle.sides[s].last_used_move != truth.sides[s].last_used_move;
          }
          desync += out[v * PI4_FIELDS + STATE_FIELDS + 4] != 0; // ours: unshown && counterable
          if (!ok) {
            bad++;
            if (printed++ < 5) {
              std::printf("== battle %u, turn %u, viewer p%d: %s\n", x.battle, truth.turn, v + 1, reason.c_str());
            }
          }
        }
      }
    }
    std::vector<uint8_t> picks(env.decisions());
    size_t i = 0;
    for (const Shard& sh : env.shards()) {
      for (const Decision& x : sh.decisions) picks[i++] = uint8_t(rng() % x.n);
    }
    env.step(picks);
    done = 0;
    for (const Shard& sh : env.shards()) done += sh.games;
  }
  std::printf("check_pi4: %zu games, %zu rows: %zu views mismatched; tracked last_used_move differs from "
              "libpkmn's in %zu of %zu sides; the desync bit on in %zu views\n",
              done, rows, bad, used_diff, 4 * rows, desync);
  return bad != 0;
}
