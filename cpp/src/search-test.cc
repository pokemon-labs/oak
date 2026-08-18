#include <util/argparse.h>
#include <util/parse.h>
#include <util/random.h>
#include <util/search.h>

#include <exception>
#include <iostream>

struct ProgramArgs : public BenchmarkArgs {};

constexpr float small = .03;

struct Test {
  std::string position;
  float expected;
  float error = small;

  void operator()(const auto &args) {
    mt19937 device{std::random_device{}()};
    auto [battle, durations] =
        Parse::parse_battle(position, std::random_device{}());
    auto options = PKMN::options();
    pkmn_gen1_chance_options chance_options{};
    chance_options.durations = durations;
    pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);
    auto eval = Search::Parse::eval(args.eval.value_or("mc"), args.quantize);
    const bool is_network = eval.is_network();
    auto p1_cache = std::shared_ptr<Search::SideCache>{};
    auto p2_cache = std::shared_ptr<Search::SideCache>{};
    if (is_network) {
      Search::Network network;
      network.data =
          std::get<std::shared_ptr<NN::Battle::NetworkBase>>(eval.data);
      if (args.quantize) {
        network.quantize();
      }
      p1_cache = std::make_shared<Search::SideCache>();
      p2_cache = std::make_shared<Search::SideCache>();
      for (auto i = 0; i < 6; ++i) {
        p1_cache->precompute(network.get(), PKMN::view(battle).sides[0], i);
        p2_cache->precompute(network.get(), PKMN::view(battle).sides[1], i);
      }
    }
    auto bandit = Search::Parse::bandit(args.bandit.value_or("exp3-1.0-0.1"));
    auto matrix_ucb =
        args.matrix_ucb.has_value()
            ? Search::Parse::matrix_ucb(bandit, args.matrix_ucb.value())
            : Search::MatrixUCB(bandit, 0);
    auto &params = args.matrix_ucb.has_value()
                       ? static_cast<Search::BanditParams &>(matrix_ucb)
                       : bandit;
    auto budget =
        Search::Parse::budget(args.budget.value_or(std::to_string(1 << 20)));
    auto heap = Search::Node{};
    auto output = RuntimeSearch::run(device, battle, PKMN::durations(options),
                                     budget, params, heap, eval, {},
                                     p1_cache.get(), p2_cache.get());
    bool success = std::abs(output.empirical_value - expected) <= error;
    if (!success) {
      std::cerr << position << std::endl;
      std::cerr << "value: " << output.empirical_value
                << " - expected: " << expected << std::endl;
      throw std::runtime_error{""};
    }
  }
};

void confusion_duration(const auto &args) {
  Test{
      .position = "starmie seismictoss 1hp (conf:5) | snorlax bodyslam 1hp",
      .expected = 1.0,
      .error = 0,
  }(args);
  Test{
      .position = "starmie seismictoss 1hp (conf:4) | snorlax bodyslam 1hp",
      .expected = .5 + .5 / 2,
  }(args);
  Test{
      .position = "starmie seismictoss 1hp (conf:3) | snorlax bodyslam 1hp",
      .expected = .33 + .66 / 2,
  }(args);
  Test{
      .position = "starmie seismictoss 1hp (conf:2) | snorlax bodyslam 1hp",
      .expected = .25 + .75 / 2,
  }(args);
  Test{
      .position = "starmie seismictoss 1hp (conf:1) | snorlax bodyslam 1hp",
      .expected = .5,
  }(args);
}

void sleep(const auto &args) {
  Test{
      .position = "starmie seismictoss 1hp slp6 | snorlax seismictoss 1hp",
      .expected = 0.0,
      .error = 0,
  }(args);
  Test{
      .position = "starmie seismictoss 101hp slp0 | snorlax seismictoss 1hp",
      .expected = 1.0 / 7,
  }(args);
  Test{
      .position = "starmie seismictoss 101hp slp1 | snorlax seismictoss 1hp",
      .expected = 1.0 / 6,
  }(args);
  Test{
      .position = "starmie seismictoss 101hp slp2 | snorlax seismictoss 1hp",
      .expected = 1.0 / 5,
  }(args);
  Test{
      .position = "starmie seismictoss 101hp slp3 | snorlax seismictoss 1hp",
      .expected = 1.0 / 4,
  }(args);
  Test{
      .position = "starmie seismictoss 101hp slp4 | snorlax seismictoss 1hp",
      .expected = 1.0 / 3,
  }(args);
  Test{
      .position = "starmie seismictoss 101hp slp5 | snorlax seismictoss 1hp",
      .expected = 1.0 / 2,
  }(args);
  Test{
      .position = "starmie seismictoss 101hp slp6 | snorlax seismictoss 1hp",
      .expected = 1.0,
      .error = 0,
  }(args);
}

void run_tests(const auto &args) {
  confusion_duration(args);
  sleep(args);
}

int main(int argc, char **argv) {

  auto args = std::move(argparse::parse<ProgramArgs>(argc, argv));

  run_tests(args);

  std::cout << "All tests passed!" << std::endl;

  return 0;
}
