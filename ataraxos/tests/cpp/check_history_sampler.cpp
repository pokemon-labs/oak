// Checks src/history_sampler.hpp on self-play: two pool teams, uniform
// Showdown choices (BattleData with `track`, as training plays), and at
// every decision of each player, a Sampler on that player's Bookkeeping:
//
// - truth: the foe's true team is among the pool teams that fit (a miss is
//   a bookkeeping mismatch showing through);
// - with the foe's true sets, the rebuilt history against the foe's real
//   one (BattleData's chunks and requests): the same summaries, the same
//   tokens (KIND, SUB, MOVE, MOVE2, SPECIES, NUM, STATUS, the idents' sides;
//   our HP exactly, the foe's own as the same percent, its max HP exactly),
//   the same requests (kind, Pokémon by species: level, max HP, HP's
//   percent, status, moves as a set; the active's moves and PP);
// - consistency, which must be 0: in every rebuilt summary, each foe
//   Pokémon's request HP and max HP equal the last its lines showed, and
//   the world's foe HP the last request's (a Faint line shows 0);
// - DRAWS pool guesses: the world gives us the real request's legal
//   choices, is consistent as above, and plays out under random choices
//   (200 updates) without an engine error.
//
//   check_history_sampler [GAMES] [SEED] [POOL] [DRAWS] [EVERY] [half|random]
// EVERY: check one decision in EVERY (default 1). `half`: the sampler's pool
// is POOL's first half and the games' teams come from its second (foes
// outside the pool, for the fallbacks); `random`: the games' teams are
// truly random (teams.hpp's random_team, distinct species).

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <libpkmn/strings.h>

#include "battle_data.hpp"
#include "history_sampler.hpp"
#include "teams.hpp"

using namespace ataraxos;

namespace {

std::map<std::string, size_t> counts;
void count(const std::string& what, size_t n = 1) { counts[what] += n; }

uint8_t side_of(int16_t ident) { return ident ? uint8_t((ident - 1) >> 3) : 2; }
uint8_t slot_of(int16_t ident) { return uint8_t((ident - 1) & 7); }

bool has_health(int16_t kind) {
  const ArgType t = ArgType(kind);
  return t == ArgType::Switch || t == ArgType::Damage || t == ArgType::Heal || t == ArgType::SetHP;
}

pkmn_choice_kind kind_of(const RequestData& r) {
  return r.kind == RequestData::Move_ ? PKMN_CHOICE_MOVE : r.kind == RequestData::Switch ? PKMN_CHOICE_SWITCH : PKMN_CHOICE_PASS;
}

// Each rebuilt request's foe HP against its lines so far; the world's
// against the last request's. Returns the inconsistencies.
size_t inconsistencies(const Sample& s, uint8_t foe) {
  size_t bad = 0;
  std::array<uint16_t, 8> hp{}, max{};
  std::array<PKMN::Data::Species, 8> species{};
  size_t at = 0;
  for (size_t i = 0; i < s.ends.size(); i++) {
    for (; at < s.ends[i]; at++) {
      const Token& t = s.tokens[at];
      if (ArgType(t.c[KIND]) == ArgType::Faint && side_of(t.c[ACTOR]) == 0) hp[slot_of(t.c[ACTOR])] = 0;
      if (!has_health(t.c[KIND]) || side_of(t.c[ACTOR]) != 0) continue;
      const uint8_t k = slot_of(t.c[ACTOR]);
      if (ArgType(t.c[KIND]) == ArgType::Switch) species[k] = PKMN::Data::Species(t.c[SPECIES]);
      hp[k] = uint16_t(t.c[HP]);
      max[k] = uint16_t(t.c[MAX_HP]);
    }
    if (i + 1 < s.ends.size()) continue; // the newest only, as `compare`
    const RequestData& r = s.requests[i];
    for (uint8_t j = 0; j < r.size; j++) {
      for (uint8_t k = 1; k <= 6; k++) {
        if (species[k] != r.side[j].species) continue;
        const bool off = hp[k] != r.side[j].hp || max[k] != r.side[j].max_hp;
        if (off && std::getenv("DEBUG") && counts["printed inconsistent"]++ < 5) {
          std::printf("inconsistent: %s lines %u/%u, request %u/%u\n", PKMN::species_string(r.side[j].species).c_str(),
                      hp[k], max[k], r.side[j].hp, r.side[j].max_hp);
        }
        bad += off;
      }
    }
  }
  const PKMN::Side& side = PKMN::view(s.battle).sides[foe];
  // (when the foe decides now: a wait's lines come after its last request)
  if (!s.requests.empty() && s.ends.back() == s.tokens.size()) {
    const RequestData& r = s.requests.back();
    for (uint8_t j = 0; j < r.size; j++) {
      for (const PKMN::Pokemon& p : side.pokemon) {
        if (p.species != r.side[j].species || p.hp == r.side[j].hp) continue;
        if (std::getenv("DEBUG") && counts["printed world"]++ < 5) {
          std::printf("world: %s %u, request %u\n", PKMN::species_string(p.species).c_str(), p.hp, r.side[j].hp);
        }
        bad++;
      }
    }
  }
  return bad;
}

void compare(const Sample& s, const std::vector<Token>& tokens, const std::vector<uint32_t>& ends,
             const std::vector<RequestData>& requests) {
  if (s.ends.size() != ends.size()) {
    count("summaries differ");
    if (std::getenv("DEBUG") && counts["summaries differ"] <= 3) {
      std::printf("rebuilt %zu summaries, real %zu\n  rebuilt:", s.ends.size(), ends.size());
      for (size_t i = 0; i < s.ends.size(); i++) std::printf(" %u/%d", s.ends[i], int(s.requests[i].kind));
      std::printf("\n  real:   ");
      for (size_t i = 0; i < ends.size(); i++) std::printf(" %u/%d", ends[i], int(requests[i].kind));
      std::printf("\n");
    }
    return;
  }
  count("summaries", ends.size());
  // the newest summary only: the earlier ones were checked at their own
  // decisions
  for (size_t i = ends.empty() ? 0 : ends.size() - 1; i < ends.size(); i++) {
    const uint32_t a = i ? s.ends[i - 1] : 0, b = i ? ends[i - 1] : 0;
    if (s.ends[i] - a != ends[i] - b) {
      count("summary lengths differ");
      continue;
    }
    for (uint32_t j = 0; j < ends[i] - b; j++) {
      const Token& x = s.tokens[a + j];
      const Token& y = tokens[b + j];
      count("tokens");
      bool same = true;
      for (Column c : {KIND, SUB, MOVE, MOVE2, SPECIES, NUM, STATUS}) same &= x.c[c] == y.c[c];
      same &= side_of(x.c[ACTOR]) == side_of(y.c[ACTOR]) && side_of(x.c[OTHER]) == side_of(y.c[OTHER]);
      if (!same) count("token fields differ");
      if (!has_health(y.c[KIND])) continue;
      if (side_of(y.c[ACTOR]) == 1) {
        if (x.c[HP] != y.c[HP] || x.c[MAX_HP] != y.c[MAX_HP]) count("our hp differs");
      } else {
        if (x.c[MAX_HP] != y.c[MAX_HP]) count("foe max hp differs");
        if (view_percent(uint16_t(x.c[HP]), uint16_t(x.c[MAX_HP])) != view_percent(uint16_t(y.c[HP]), uint16_t(y.c[MAX_HP]))) {
          count("foe hp percent differs");
        }
        if (x.c[HP] != y.c[HP]) count("foe hp inexact (expected)");
      }
    }
    const RequestData& x = s.requests[i];
    const RequestData& y = requests[i];
    count("requests");
    if (x.kind != y.kind || x.size != y.size) {
      count("request kind/size differs");
      if (std::getenv("DEBUG") && counts["printed kind"]++ < 5) {
        std::printf("kind/size: rebuilt %d/%d, real %d/%d\n", int(x.kind), int(x.size), int(y.kind), int(y.size));
      }
      continue;
    }
    for (uint8_t j = 0; j < y.size; j++) {
      const RequestData::Pokemon& q = y.side[j];
      const RequestData::Pokemon* p = nullptr;
      for (uint8_t k = 0; k < x.size; k++) {
        if (x.side[k].species == q.species) p = &x.side[k];
      }
      if (!p) {
        count("request species differ");
        continue;
      }
      if (p->level != q.level || p->max_hp != q.max_hp) count("request level/max hp differs");
      if (view_percent(p->hp, p->max_hp) != view_percent(q.hp, q.max_hp)) count("request hp percent differs");
      if (p->status != q.status) count("request status differs");
      if ((p->status != q.status || view_percent(p->hp, p->max_hp) != view_percent(q.hp, q.max_hp)) &&
          std::getenv("DEBUG") && counts["printed request"]++ < 8) {
        std::printf("request %s: rebuilt hp %u/%u status %d active %d, real %u/%u status %d active %d\n",
                    PKMN::species_string(q.species).c_str(), p->hp, p->max_hp, int(p->status), int(p->active), q.hp,
                    q.max_hp, int(q.status), int(q.active));
      }
      if (p->active != q.active) count("request active differs");
      std::vector<PKMN::Data::Move> m1(p->moves, p->moves + 4), m2(q.moves, q.moves + 4);
      std::sort(m1.begin(), m1.end());
      std::sort(m2.begin(), m2.end());
      if (m1 != m2) count("request moves differ");
    }
    if (y.kind == RequestData::Move_) {
      if (x.active.locked != y.active.locked) count("request locked differs");
      std::vector<std::pair<int, int>> p1, p2;
      for (uint8_t k = 0; k < 4; k++) {
        p1.emplace_back(int(x.active.moves[k].id), x.active.moves[k].pp);
        p2.emplace_back(int(y.active.moves[k].id), y.active.moves[k].pp);
      }
      std::sort(p1.begin(), p1.end());
      std::sort(p2.begin(), p2.end());
      if (p1 != p2) count("request active moves/pp differ");
    }
  }
}

template <class RNG> void play_out(const Sample& s, uint8_t us, const BattleData& bd, RNG& rng) {
  pkmn_gen1_battle b = s.battle;
  const PKMN::Battle& v = PKMN::view(b);
  const uint8_t foe = 1 - us;
  pkmn_choice_kind kinds[2];
  kinds[us] = kind_of(bd.request[us]);
  kinds[foe] = v.sides[foe].stored().hp == 0 ? PKMN_CHOICE_SWITCH
               : v.sides[us].stored().hp == 0 ? PKMN_CHOICE_PASS
                                               : PKMN_CHOICE_MOVE;
  std::array<pkmn_choice, 9> out{};
  const uint8_t n = uint8_t(pkmn_gen1_battle_choices(&b, us == 0 ? PKMN_PLAYER_P1 : PKMN_PLAYER_P2, kinds[us], out.data(), out.size()));
  std::vector<pkmn_choice> mine(out.begin(), out.begin() + n), real(bd.legal[us].begin(), bd.legal[us].begin() + bd.n_legal[us]);
  std::sort(mine.begin(), mine.end());
  std::sort(real.begin(), real.end());
  if (mine != real) count("draw: our choices differ");

  std::array<uint8_t, LOG_SIZE> buf{};
  pkmn_gen1_battle_options options{};
  pkmn_gen1_chance_options chance{};
  chance.durations = std::bit_cast<pkmn_gen1_chance_durations>(s.durations);
  for (int t = 0; t < 200; t++) {
    pkmn_choice c[2];
    for (uint8_t p = 0; p < 2; p++) {
      const uint8_t m = uint8_t(pkmn_gen1_battle_choices(&b, p == 0 ? PKMN_PLAYER_P1 : PKMN_PLAYER_P2, kinds[p], out.data(), out.size()));
      c[p] = m ? out[rng() % m] : 0;
    }
    pkmn_gen1_log_options log{buf.data(), buf.size()};
    pkmn_gen1_battle_options_set(&options, &log, t == 0 ? &chance : nullptr, nullptr);
    const pkmn_result r = pkmn_gen1_battle_update(&b, c[0], c[1], &options);
    if (pkmn_result_type(r) == PKMN_RESULT_ERROR) {
      count("draw: engine error");
      return;
    }
    if (pkmn_result_type(r) != PKMN_RESULT_NONE) return;
    kinds[0] = pkmn_result_p1(r);
    kinds[1] = pkmn_result_p2(r);
  }
}

} // namespace

int main(int argc, char** argv) {
  const size_t games = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 200;
  std::mt19937_64 rng(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1);
  const std::string path = argc > 3 ? argv[3] : "teams/selfplay/oak1500.txt";
  const size_t draws = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 4;
  const size_t every = argc > 5 ? std::max<size_t>(1, std::strtoul(argv[5], nullptr, 10)) : 1;
  const std::string mode = argc > 6 ? argv[6] : "";
  size_t errors = 0;
  std::vector<Team> all = load_teams({path}, errors);
  std::vector<Team> games_teams = all;
  if (mode == "half") {
    games_teams.assign(all.begin() + long(all.size() / 2), all.end());
    all.resize(all.size() / 2);
  }
  const bool outside = !mode.empty();
  const Pool pool(all);
  std::printf("pool %zu teams, %zu distinct sets (%zu load errors)\n", pool.teams.size(), pool.sets.size(), errors);

  size_t decision = 0;
  for (size_t g = 0; g < games; g++) {
    const uint32_t ti[2] = {uint32_t(rng() % games_teams.size()), uint32_t(rng() % games_teams.size())};
    Team teams[2] = {games_teams[ti[0]], games_teams[ti[1]]};
    for (uint8_t p = 0; p < 2 && mode == "random"; p++) {
      // distinct species (Species Clause; the checks match by species)
      for (;;) {
        teams[p] = random_team(rng);
        std::vector<PKMN::Data::Species> sp;
        for (const Set& x : teams[p]) sp.push_back(x.species);
        std::sort(sp.begin(), sp.end());
        if (std::adjacent_find(sp.begin(), sp.end()) == sp.end()) break;
      }
    }
    BattleData bd;
    bd.track = true;
    bd.reset(init_battle(rng(), teams[0], teams[1]));
    std::vector<Bookkeeping> after[2];
    std::vector<Token> tokens[2];
    std::vector<uint32_t> ends[2];
    std::vector<RequestData> requests[2];
    for (int step = 0; step < 2000 && !bd.over(); step++) {
      for (uint8_t p = 0; p < 2; p++) after[p].push_back(bd.bk[p]);
      for (uint8_t p = 0; p < 2; p++) {
        if (!bd.needs_choice(p)) continue;
        tokens[p].insert(tokens[p].end(), bd.chunk[p].begin(), bd.chunk[p].end());
        bd.chunk[p].clear();
        ends[p].push_back(uint32_t(tokens[p].size()));
        requests[p].push_back(bd.request[p]);
      }
      for (uint8_t p = 0; p < 2; p++) {
        if (!bd.needs_choice(p) || decision++ % every) continue;
        const uint8_t foe = 1 - p;
        Sampler sampler(pool, bd.bk[p]);
        count("decisions");
        if (!outside && !sampler.matches(ti[foe])) count("true team not among the matches");
        count("matching teams", sampler.matches());

        // the foe's true sets, by bookkeeping's numbering
        Guess truth;
        const Team& team = teams[foe];
        std::vector<bool> used(team.size());
        const PKMN::Side& side = bd.bk[p].battle.sides[foe];
        for (uint8_t k = 0; k < 6; k++) {
          for (size_t j = 0; j < team.size(); j++) {
            if (!used[j] && side.pokemon[k].species == team[j].species) {
              truth.sets[k] = team[j];
              used[j] = true;
              break;
            }
          }
        }
        uint8_t k = 0;
        for (size_t j = 0; j < team.size(); j++) {
          if (used[j]) continue;
          while (k < 6 && side.pokemon[k].species != PKMN::Data::Species::None) k++;
          if (k < 6) truth.sets[k++] = team[j];
        }
        const Sample s = sampler.sample(truth, bd.history[p], after[p], rng());
        if (std::getenv("DEBUG") && s.ends.size() != ends[foe].size() && counts["summaries differ"] < 3) {
          std::printf("game %zu step %d us %d: foe request kind %d chosen %d over %d, updates %zu/%zu\n", g, step, int(p),
                      int(bd.request[foe].kind), int(bd.chosen[foe]), int(bd.over()), bd.history[p].size(), after[p].size());
        }
        compare(s, tokens[foe], ends[foe], requests[foe]);
        count("inconsistent (truth)", inconsistencies(s, foe));

        for (size_t d = 0; d < draws; d++) {
          const Guess guess = sampler.guess(rng);
          count(guess.source == Guess::FromTeam ? "draw: from a team" : guess.source == Guess::FromSets ? "draw: from sets" : "draw: random moves");
          const Sample w = sampler.sample(guess, bd.history[p], after[p], rng());
          count("inconsistent (draws)", inconsistencies(w, foe));
          play_out(w, p, bd, rng);
        }
      }
      for (uint8_t p = 0; p < 2; p++) {
        if (!bd.needs_choice(p)) continue;
        // one update per step, so each is recorded above
        if (bd.choose(p, bd.option[p][rng() % bd.n_option[p]]) != BattleData::Held) break;
      }
    }
  }
  for (const auto& [what, n] : counts) std::printf("%-40s %zu\n", what.c_str(), n);
  return 0;
}
