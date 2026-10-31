// Checks src/bookkeeping.hpp against the true battle, as safari's
// tests/client.py checks PSBattle: random self-play, and after every update
// each player's Bookkeeping, fed that player's view (view.hpp), must match
// the truth by oak's compare_battles (cpp/include/libpkmn/client.h) under
// safari's key: the substitute's HP within 20, Bide's damage within 50,
// binding not compared, the other side's HP within 15, its last selected
// move and the slot it selected not compared (a full paralysis or a sleep
// leaves the choice out of the log), the last damage within 15.
//
// Teams: 6 distinct species a side, level 100, 4 random moves each (the
// species distinct so the bench matches by species); a banned move is left
// out. Choices uniform over libpkmn's. A game stops at its first mismatch,
// which is counted by its reason; the first few are printed with the
// update's log (oak's text) and both battles.
//
//   check_bookkeeping [GAMES] [SEED] [BANNED,MOVES,...|-] [REASON]
// BANNED is a comma-separated list of move names as libpkmn spells them
// (e.g. Wrap,Bind,Clamp,FireSpin), or - for none; with REASON only the
// mismatches whose reason contains it are printed.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <libpkmn/client.h>
#include <libpkmn/log.h>
#include <libpkmn/strings.h>

#include "bookkeeping.hpp"
#include "request_data.hpp"
#include "teams.hpp"
#include "view.hpp"

using namespace ataraxos;

namespace {

std::vector<PKMN::Data::Move> banned_moves(const char* arg) {
  std::vector<PKMN::Data::Move> out;
  if (!arg) return out;
  std::stringstream ss(arg);
  std::string name;
  while (std::getline(ss, name, ',')) {
    bool found = false;
    for (size_t m = 1; m < MOVE_COUNT; m++) {
      if (detail::key(PKMN::Data::MOVE_CHAR_ARRAY[m].data()) == detail::key(name)) {
        out.push_back(PKMN::Data::Move(m));
        found = true;
      }
    }
    if (!found) std::printf("unknown move %s\n", name.c_str());
  }
  return out;
}

Team test_team(std::mt19937_64& rng, const std::vector<PKMN::Data::Move>& banned) {
  std::vector<uint8_t> species(151);
  for (size_t i = 0; i < 151; i++) species[i] = uint8_t(i + 1);
  std::shuffle(species.begin(), species.end(), rng);
  Team team(6);
  for (size_t i = 0; i < 6; i++) {
    Set& s = team[i];
    s.species = PKMN::Data::Species(species[i]);
    s.level = 100;
    s.moves = {};
    size_t k = 0;
    while (k < 4) {
      const auto m = PKMN::Data::Move(1 + rng() % (MOVE_COUNT - 2));
      bool skip = std::find(banned.begin(), banned.end(), m) != banned.end();
      for (size_t j = 0; j < k; j++) skip |= s.moves[j] == m;
      if (!skip) s.moves[k++] = m;
    }
  }
  return team;
}

// safari's get_match_keys, the other side being `foe`.
std::pair<PKMN::Battle, PKMN::Durations> keys(uint8_t foe) {
  PKMN::Battle b{};
  PKMN::Durations d{};
  for (uint8_t i = 0; i < 2; i++) {
    PKMN::Volatiles& v = b.sides[i].active.volatiles;
    bk::set_substitute_hp(v, 20);
    v.set_state(50);
    v.set_binding(true);
    d.get(i).set_binding(1);
  }
  b.sides[foe].pokemon[0].hp = 15;
  b.sides[foe].last_selected_move = PKMN::Data::Move(1);
  b.last_moves[foe].index = 1;
  b.last_damage = 15;
  return {b, d};
}

std::string battle_string(const PKMN::Battle& b, const PKMN::Durations& d) {
  std::stringstream out;
  for (uint8_t s = 0; s < 2; s++) {
    const PKMN::Side& side = b.sides[s];
    out << "  p" << int(s + 1) << " order";
    for (uint8_t o : side.order) out << ' ' << int(o);
    out << " | last used " << PKMN::move_string(side.last_used_move) << ", selected "
        << PKMN::move_string(side.last_selected_move) << " | durations " << std::hex << d.get(s).data << std::dec
        << "\n    active " << PKMN::species_string(side.active.species) << " stats " << side.active.stats.atk << '/'
        << side.active.stats.def << '/' << side.active.stats.spe << '/' << side.active.stats.spc << " boosts "
        << int(side.active.boosts.atk()) << ',' << int(side.active.boosts.def()) << ','
        << int(side.active.boosts.spe()) << ',' << int(side.active.boosts.spc()) << ','
        << int(side.active.boosts.acc()) << ',' << int(side.active.boosts.eva()) << " volatiles "
        << PKMN::volatiles_to_string(side.active.volatiles) << " moves";
    for (const PKMN::MoveSlot& m : side.active.moves) out << ' ' << PKMN::move_string(m.id) << ':' << int(m.pp);
    out << '\n';
    for (const PKMN::Pokemon& p : side.pokemon) {
      if (p.species == PKMN::Data::Species::None) continue;
      out << "    " << PKMN::species_string(p.species) << ' ' << p.hp << '/' << p.stats.hp << " status "
          << int(uint8_t(p.status));
      for (const PKMN::MoveSlot& m : p.moves) out << ' ' << PKMN::move_string(m.id) << ':' << int(m.pp);
      out << '\n';
    }
  }
  out << "  turn " << b.turn << ", last damage " << b.last_damage << ", last moves " << int(b.last_moves[0].index)
      << '/' << int(b.last_moves[0].counterable) << ' ' << int(b.last_moves[1].index) << '/'
      << int(b.last_moves[1].counterable) << '\n';
  return out.str();
}

} // namespace

int main(int argc, char** argv) {
  const size_t games = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 1000;
  std::mt19937_64 rng(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1);
  const std::vector<PKMN::Data::Move> banned =
      banned_moves(argc > 3 && std::string(argv[3]) != "-" ? argv[3] : nullptr);
  const std::string only = argc > 4 ? argv[4] : "";
  std::map<std::string, size_t> reasons;
  size_t failed = 0, updates = 0, printed = 0;
  for (size_t g = 0; g < games; g++) {
    const Team t1 = test_team(rng, banned), t2 = test_team(rng, banned);
    pkmn_gen1_battle b = init_battle(rng(), t1, t2);
    PKMN::Durations truth_d{};
    Bookkeeping bks[2] = {Bookkeeping(0, {6, 6}, PKMN::view(b).sides[0]),
                          Bookkeeping(1, {6, 6}, PKMN::view(b).sides[1])};
    View views[2] = {View{0}, View{1}};
    std::array<uint8_t, LOG_SIZE> buf{};
    pkmn_gen1_battle_options options{};
    pkmn_choice c[2] = {0, 0};
    std::vector<std::string> text;
    bool bad = false;
    for (int turn = 0; turn < 1000 && !bad; turn++) {
      pkmn_gen1_log_options log{buf.data(), buf.size()};
      pkmn_gen1_chance_options chance{};
      chance.durations = std::bit_cast<pkmn_gen1_chance_durations>(truth_d);
      pkmn_gen1_battle_options_set(&options, &log, &chance, nullptr);
      for (uint8_t p = 0; p < 2; p++) bks[p].chose(c[p]);
      const pkmn_result r = pkmn_gen1_battle_update(&b, c[0], c[1], &options);
      truth_d = std::bit_cast<PKMN::Durations>(*pkmn_gen1_battle_options_chance_durations(&options));
      updates++;
      {
        PKMN::Log::Parser<PKMN::Log::View::omniscient> parser{buf.data()};
        parser.battle = b;
        parser.parse();
        text.push_back("-- " + PKMN::side_choice_string(PKMN::view(b).sides[0], c[0]) + " / " +
                       PKMN::side_choice_string(PKMN::view(b).sides[1], c[1]));
        for (const std::string& l : parser.log) text.push_back(l);
      }
      size_t used = 0;
      while (buf[used] != 0) used += 1 + message_size(ArgType(buf[used]), &buf[used + 1]);
      std::array<pkmn_choice, MAX_CHOICES> opts[2];
      size_t n[2] = {0, 0};
      for (uint8_t p = 0; p < 2 && !pkmn_result_type(r); p++) {
        const pkmn_choice_kind kind = p ? pkmn_result_p2(r) : pkmn_result_p1(r);
        n[p] = pkmn_gen1_battle_choices(&b, p ? PKMN_PLAYER_P2 : PKMN_PLAYER_P1, kind, opts[p].data(), opts[p].size());
      }
      for (uint8_t p = 0; p < 2 && !bad; p++) {
        std::array<uint8_t, LOG_SIZE> v = buf;
        const size_t len = views[p].apply({v.data(), used + 1});
        bks[p].update(split({v.data(), len}));
        if (!pkmn_result_type(r)) {
          const pkmn_choice_kind kind = p ? pkmn_result_p2(r) : pkmn_result_p1(r);
          bks[p].request(request_data(PKMN::view(b), p, kind, {opts[p].data(), n[p]}));
        }
        const auto [key, dkey] = keys(uint8_t(1 - p));
        std::string reason;
        if (!PKMN::Client::compare_battles(bks[p].battle, bks[p].durations, PKMN::view(b), truth_d, key, dkey, reason)) {
          bad = true;
          failed++;
          reasons["p" + std::to_string(p + 1) + "'s view: " + reason]++;
          if (reason.find(only) != std::string::npos && printed++ < 3) {
            std::printf("== game %zu, turn %d, p%d's view: %s\n", g, turn, p + 1, reason.c_str());
            const size_t from = text.size() > 30 ? text.size() - 30 : 0;
            for (size_t j = from; j < text.size(); j++) std::printf("%s\n", text[j].c_str());
            std::printf("truth:\n%sbookkeeping:\n%s", battle_string(PKMN::view(b), truth_d).c_str(),
                        battle_string(bks[p].battle, bks[p].durations).c_str());
          }
        }
      }
      if (pkmn_result_type(r)) break;
      for (uint8_t p = 0; p < 2; p++) c[p] = opts[p][rng() % n[p]];
    }
  }
  std::printf("check_bookkeeping: %zu games, %zu updates: %zu games mismatched\n", games, updates, failed);
  std::vector<std::pair<size_t, std::string>> sorted;
  for (const auto& [r, k] : reasons) sorted.push_back({k, r});
  std::sort(sorted.rbegin(), sorted.rend());
  for (const auto& [k, r] : sorted) std::printf("  %6zu  %s\n", k, r.c_str());
  return failed != 0;
}
