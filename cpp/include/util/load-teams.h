#pragma once

#include <util/parse.h>

#include <string>
#include <vector>

auto _load_teams(std::string path) {
  using Team = std::vector<PKMN::Set>;
  using Teams = std::vector<Team>;

  // read teams file
  const auto side_to_team = [](const PKMN::Side &side) {
    std::vector<PKMN::Set> team{};
    for (const auto &pokemon : side.pokemon) {
      if (pokemon.species != PKMN::Data::Species::None) {
        PKMN::Set set{};
        set.species = pokemon.species;
        std::transform(pokemon.moves.begin(), pokemon.moves.end(),
                       set.moves.begin(), [](const auto ms) { return ms.id; });
        team.emplace_back(set);
      }
    }
    return team;
  };

  Teams teams{};

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
  return teams;
}
