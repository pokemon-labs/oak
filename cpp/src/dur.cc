#include <libpkmn/pkmn.h>
#include <libpkmn/strings.h>
#include <teams/ou-sample-teams.h>
#include <util/random.h>

#include <iostream>
#include <map>

namespace Util {

template <typename F, bool debug_log = true>
pkmn_result rollout_and_exec(auto &device, pkmn_gen1_battle &battle,
                             pkmn_gen1_battle_options &options, F func) {
  auto result = PKMN::result(battle);
  std::array<pkmn_choice, 9> choices;
  while (!pkmn_result_type(result)) {

    func(battle, options);

    auto seed = device.uniform_64();
    const auto m = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    const auto c1 = choices[seed % m];
    const auto n = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    seed >>= 32;
    const auto c2 = choices[seed % n];
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
  }
  return result;
}

} // namespace Util

using Key = std::pair<int, int>;
std::map<Key, size_t> map{};

auto teams = Teams::ou_sample_teams;
pkmn_gen1_battle old_battle;
pkmn_gen1_battle_options old_options;

const auto F = [&map, &old_battle, &old_options](auto b, auto options) {
  const auto &battle = PKMN::view(b);
  const pkmn_gen1_chance_durations &durations =
      *pkmn_gen1_battle_options_chance_durations(&options);

  for (auto s = 0; s < 2; ++s) {
    const auto &vol = battle.sides[s].active.volatiles;
    const auto &duration = PKMN::view(durations).get(s);

    if (vol.binding() != 0) {
      // This is all you have to change besides setting the first move in main
      const auto set = static_cast<int>(vol.attacks());
      const auto seen = static_cast<int>(duration.binding());
      std::pair<int, int> key{seen, set};
      map[key] += 1;

      if ((seen == 0) && (set > 0)) {
        const pkmn_gen1_chance_durations &old_durations =
            *pkmn_gen1_battle_options_chance_durations(&old_options);
        std::cout << "START" << std::endl;
        std::cout << PKMN::battle_data_to_string(old_battle, old_durations);
        std::cout << "________" << std::endl;
        std::cout << PKMN::battle_data_to_string(b, durations);
        std::cout << "END" << std::endl;
      }
    }
  }

  old_battle = b;
  old_options = options;
};

int main(int argc, char **argv) {
  using enum PKMN::Data::Move;

  if (argc < 2) {
    std::cerr << "Input: num-trials" << std::endl;
    return 1;
  }

  for (auto &team : teams) {
    for (auto &set : team) {
      set.moves[0] = Wrap;
    }
  }

  mt19937 device{std::random_device{}()};

  for (auto i = 0; i < std::atoll(argv[1]); ++i) {
    std::array<uint8_t, 64> buf{};
    pkmn_gen1_log_options log_options{buf.data(), 64};
    auto battle =
        PKMN::battle(teams[device.uniform_64() % 16],
                     teams[device.uniform_64() % 16], device.uniform_64());
    auto options = PKMN::options(log_options);
    auto result = PKMN::update(battle, 0, 0, options);
    Util::rollout_and_exec(device, battle, options, F);
  }

  using Probs = std::array<float, 10>;

  std::map<int, Probs> total{};

  for (const auto [key, value] : map) {
    auto [seen, set] = key;
    auto &probs = total[seen];
    probs[set] += (float)value;
  }

  for (const auto [key, value] : total) {
    std::cout << "Duration: " << key << std::endl;
    std::cout << "Hidden:   ";
    const auto sum = std::accumulate(value.begin(), value.end(), 0);
    for (auto i : value) {
      std::cout << i / (float)sum << ' ';
    }
    std::cout << " Total: " << sum;
    std::cout << std::endl;
  }

  return 0;
}