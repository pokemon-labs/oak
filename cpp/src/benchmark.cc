#include <teams/benchmark-teams.h>
#include <util/argparse.h>
#include <util/search.h>

struct ProgramArgs : public BenchmarkArgs {
  bool &cleanup = flag("cleanup", "Include heap cleanup in time");
};

int benchmark(int argc, char **argv) {

  const uint32_t seed = 1111111;
  auto device = mt19937{seed};
  auto p1 = Teams::benchmark_teams[0];
  auto p2 = Teams::benchmark_teams[1];
  auto battle = PKMN::battle(p1, p2, seed);
  auto options = PKMN::options();
  const auto result = PKMN::update(battle, 0, 0, options);
  const auto durations = PKMN::durations();
  const auto input = MCTS::Input{battle, durations, result};

  auto args = argparse::parse<ProgramArgs>(argc, argv);
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
  auto bandit = Search::Parse::bandit(args.bandit.value_or("ucb-1.0"));
  auto matrix_ucb =
      args.matrix_ucb.has_value()
          ? Search::Parse::matrix_ucb(bandit, args.matrix_ucb.value())
          : Search::MatrixUCB(bandit, 0);
  auto &params = args.matrix_ucb.has_value()
                     ? static_cast<Search::BanditParams &>(matrix_ucb)
                     : bandit;
  auto budget =
      Search::Parse::budget(args.budget.value_or(std::to_string(1 << 20)));
  const auto start = std::chrono::high_resolution_clock::now();
  auto output = MCTS::Output{};
  {
    auto heap = Search::Parse::heap(args.use_table);
    output = RuntimeSearch::run(device, battle, PKMN::durations(options),
                                budget, params, heap, eval, output,
                                p1_cache.get(), p2_cache.get());
  }
  auto microseconds = output.duration.count();
  if (args.cleanup) {
    const auto end = std::chrono::high_resolution_clock::now();
    microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
  }
  if (microseconds >= 10000) {
    std::cout << (microseconds / 1000) << "ms." << std::endl;
  } else {
    std::cout << microseconds << "µs." << std::endl;
  }
  std::cout << output.iterations << " iterations." << std::endl;

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }
