#pragma once

#include <encode/build/compressed-trajectory.h>
#include <format/random-battles/randbat.h>
#include <nn/build/network.h>
#include <teams/ou-sample-teams.h>
#include <train/build/trajectory.h>
#include <util/file-lock.h>
#include <util/parse.h>

#include <unistd.h>

namespace TeamBuilding {

inline const auto team_string(const auto &team) {
  std::stringstream ss{};
  for (const auto &set : team) {
    ss << PKMN::species_string(set.species) << ": ";
    for (const auto moveid : set.moves) {
      ss << PKMN::move_string(moveid) << ' ';
    }
    ss << '\n';
  }
  return ss.str();
};

inline auto softmax(auto &x) {
  auto y = x;
  std::transform(y.begin(), y.end(), y.begin(),
                 [](const auto v) { return std::exp(v); });
  const auto sum = std::accumulate(y.begin(), y.end(), 0.0);
  std::transform(y.begin(), y.end(), y.begin(),
                 [sum](const auto v) { return v / sum; });
  return y;
}

[[nodiscard]] inline auto rollout_build_network(auto &device, auto &network,
                                                const auto &team) {
  using namespace Train::Build;

  Trajectory trajectory{};
  trajectory.initial = team;
  trajectory.terminal = team;

  auto input = Encode::Build::Tensorizer<>::write(team);
  std::array<float, Encode::Build::Tensorizer<>::n_dim> logits;
  auto actions = Encode::Build::Actions<>::get_singleton_additions(team);

  const auto go = [&]() {
    // get action indices
    std::vector<int> indices;
    std::transform(actions.begin(), actions.end(), std::back_inserter(indices),
                   [](auto action) {
                     return Encode::Build::Tensorizer<>::action_index(action);
                   });

    network.inference(input.data(), logits.data());

    // get legal logits, softmax, sample action, apply
    std::vector<float> legal_logits;
    std::transform(indices.begin(), indices.end(),
                   std::back_inserter(legal_logits),
                   [&logits](const auto index) { return logits[index]; });
    const auto policy = softmax(legal_logits);
    const auto index = device.sample_pdf(policy);
    const auto action = actions[index];
    apply_action(trajectory.terminal, action);
    input[index] = 1.0;

    trajectory.updates.emplace_back(
        Trajectory::Update{actions, index, policy[index]});
  };

  while (!actions.empty()) {
    go();
    actions =
        Encode::Build::Actions<>::get_singleton_additions(trajectory.terminal);
  }
  actions = Encode::Build::Actions<>::get_lead_swaps(trajectory.terminal);
  if (!actions.empty()) {
    go();
  }

  return trajectory;
}

struct Omitter {

  int max_pokemon{6};
  double pokemon_delete_prob{};
  double move_delete_prob{};
  double team_shuffle_prob{};

  bool shuffle_and_truncate(auto &device, auto &team) const {
    const auto original = team;
    if (device.uniform() < team_shuffle_prob) {
      std::mt19937 rd{device.uniform_64()};
      std::shuffle(team.begin(), team.end(), rd);
    }
    if (max_pokemon < team.size()) {
      team.resize(max_pokemon);
    }
    return (original != team);
  }

  bool delete_info(auto &device, auto &team) const {
    using PKMN::Data::Move;
    using PKMN::Data::Species;
    bool modified = false;
    for (auto &set : team) {
      if ((set.species != Species::None) &&
          (device.uniform() < pokemon_delete_prob)) {
        set = PKMN::Set{};
        modified = true;
      } else {
        for (auto &move : set.moves) {
          if ((move != Move::None) && (device.uniform() < move_delete_prob)) {
            move = Move::None;
            modified = true;
          }
        }
      }
    }
    return modified;
  }
};

struct Provider {

  bool rb;
  std::vector<std::vector<PKMN::Set>> teams;
  Omitter omitter;
  std::string network_path;
  double team_modify_prob;

  Provider() = default;

  Provider(const std::string &teams_path)
      : rb{false}, teams{}, omitter{}, network_path{} {
    if ((teams_path == "random-battles") || (teams_path == "randbats")) {
      rb = true;
    } else {
      load_teams(teams_path);
    }
  }

  void load_teams(const std::string &path) {
    if (!path.empty()) {

      // read teams file
      const auto side_to_team = [](const PKMN::Side &side) {
        std::vector<PKMN::Set> team{};
        for (const auto &pokemon : side.pokemon) {
          if (pokemon.species != PKMN::Data::Species::None) {
            PKMN::Set set{};
            set.species = pokemon.species;
            std::transform(pokemon.moves.begin(), pokemon.moves.end(),
                           set.moves.begin(),
                           [](const auto ms) { return ms.id; });
            team.emplace_back(set);
          }
        }
        return team;
      };

      std::ifstream file{path};
      while (true) {
        std::string line{};
        std::getline(file, line);
        if (line.empty()) {
          break;
        }
        const auto [side, _] = Parse::parse_side(line);
        teams.push_back(side_to_team(side));
      }
      if (teams.size() == 0) {
        throw std::runtime_error{
            "Team Provider: Did not read any teams from path: " + path};
      }
    } else {
      // use sample teams
      std::transform(Teams::ou_sample_teams.begin(),
                     Teams::ou_sample_teams.end(), std::back_inserter(teams),
                     [](const auto &team) {
                       std::vector<PKMN::Set> t{team.begin(), team.end()};
                       return t;
                     });
    }
  }

  auto get_trajectory(auto &device)
      -> std::pair<Train::Build::Trajectory, int> {

    if (rb) {
      Train::Build::Trajectory trajectory{};
      const auto seed = std::bit_cast<int64_t>(device.uniform_64());
      RandomBattles::PRNG prng{seed};
      RandomBattles::Teams t{prng};
      // completed but in weird format
      const auto partial_team = t.randomTeam();
      const auto team = t.partialToTeam(partial_team);
      std::vector<PKMN::Set> team_vec(team.begin(), team.end());
      trajectory.initial = team_vec;
      trajectory.terminal = team_vec;
      return {trajectory, -1};
    }

    assert(teams.size() > 0);

    const int team_index = device.random_int(teams.size());
    auto team = teams[team_index];
    bool changed = omitter.shuffle_and_truncate(device, team);
    if (device.uniform() < team_modify_prob) {
      changed = changed || omitter.delete_info(device, team);
    }

    if (!changed) {
      Train::Build::Trajectory trajectory{};
      trajectory.initial = trajectory.terminal = team;
      return {trajectory, team_index};
    } else {
      // loading each time allows the network params to be updated at runtime
      NN::Build::Network build_network{};
      auto [file, fd] = FileLock::try_open_file(network_path);
      FileLock::FdGuard guard{fd};

      const auto trajectory =
          TeamBuilding::rollout_build_network(device, build_network, team);
      assert(trajectory.updates.size() > 0);
      return {trajectory, -1};
    }
  }

  void read_network_parameters() const {
    const bool can_build =
        (team_modify_prob > 0) &&
        ((omitter.pokemon_delete_prob > 0) || (omitter.move_delete_prob > 0));
    if (can_build) {
      NN::Build::Network test_network{};
      std::ifstream file{network_path};
      if (!test_network.read_parameters(file)) {
        throw std::runtime_error{"Can't read build network path while kargs "
                                 "make teambuilding possible."};
      }
    }
  }
};

} // namespace TeamBuilding