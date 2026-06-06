#include <libpkmn/layout.h>
#include <libpkmn/log.h>
#include <teams/ou-sample-teams.h>
#include <util/debug-log.h>
#include <util/random.h>

#include <iostream>
#include <optional>

int rollout_sample_teams_and_stream_debug_log(int argc, char **argv) {
  using namespace PKMN::Log;

  constexpr size_t log_size{128};
  using Teams::ou_sample_teams;

  mt19937 device{std::random_device{}()};

  for (auto i = 0; i < 1; ++i) {
    auto p1 = device.random_int(ou_sample_teams.size());
    auto p2 = device.random_int(ou_sample_teams.size());

    auto battle = PKMN::battle(ou_sample_teams[p1], ou_sample_teams[p2],
                               device.uniform_64());
    pkmn_gen1_battle_options options{};
    std::array<pkmn_choice, 9> choices{};

    DebugLog<log_size> debug_log{};
    debug_log.set_header(battle);

    auto turns = 0;
    pkmn_choice c1{0};
    pkmn_choice c2{0};
    pkmn_result result;

    while (
        !pkmn_result_type(result = debug_log.update(battle, c1, c2, options))) {
      const auto *buffer = debug_log.frames.back().data();
      Parser p(buffer);
      p.battle = battle;
      p.parse();
      std::cout << PKMN::battle_data_to_string(battle, PKMN::durations(options))
                << std::endl;

      const auto m = pkmn_gen1_battle_choices(
          &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
          PKMN_GEN1_MAX_CHOICES);
      c1 = choices[device.random_int(m)];
      const auto n = pkmn_gen1_battle_choices(
          &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
          PKMN_GEN1_MAX_CHOICES);
      c2 = choices[device.random_int(n)];

      std::cout << PKMN::side_choice_string(battle.bytes, c1) << ' '
                << PKMN::side_choice_string(
                       battle.bytes + PKMN::Layout::Sizes::Side, c2)
                << std::endl;

      ++turns;
    }
    std::cout << PKMN::battle_data_to_string(battle, PKMN::durations(options))
              << std::endl;
  }
  return 0;
}

int main(int argc, char **argv) {
  return rollout_sample_teams_and_stream_debug_log(argc, argv);
}
