#pragma once

#include <libpkmn/data.h>
#include <search/data.h>

#include <map>
#include <memory>
#include <mutex>
#include <utility>

struct CachePool {
  using Set = std::pair<PKMN::Data::Species, std::array<PKMN::Data::Move, 4>>;
  using Team = std::array<Set, 6>;

  static constexpr auto get_team(const PKMN::Side &side) {
    Team team;
    for (auto i = 0; i < 6; ++i) {
      const auto &pokemon = side.pokemon[i];
      auto &set = team[i];
      set.first = pokemon.species;
      for (auto m = 0; m < 4; ++m) {
        set.second[m] = pokemon.moves[m].id;
      }
    }
    return team;
  }

  std::mutex mutex;
  using CachePtr = std::shared_ptr<Search::SideCache>;
  std::map<Team, CachePtr> caches;

  std::shared_ptr<Search::SideCache> access(Search::Network &network,
                                            const PKMN::Side &side) {
    auto lock = std::unique_lock{mutex};
    auto &cache = caches[get_team(side)];
    if (!cache) {
      cache = std::make_shared<Search::SideCache>();
      if (network.is_quantized()) {
        cache->quantize(network.get());
      }
      for (auto i = 0; i < 6; ++i) {
        cache->precompute(network.get(), side, i);
      }
    }
    return cache;
  }

  std::shared_ptr<Search::SideCache>
  get(const Search::Eval &eval, const PKMN::Side &side, bool use_pool) {
    if (eval.is_network()) {
      Search::Network network;
      network.data =
          std::get<std::shared_ptr<NN::Battle::NetworkBase>>(eval.data);
      if (use_pool) {
        return access(network, side);
      } else {
        auto cache = std::make_shared<Search::SideCache>();
        if (network.is_quantized()) {
          cache->quantize(network.get());
        }
        for (auto i = 0; i < 6; ++i) {
          cache->precompute(network.get(), side, i);
        }
        return cache;
      }
    } else {
      return {};
    }
  }
};
