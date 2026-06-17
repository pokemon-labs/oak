#include <teams/benchmark-teams.h>
#include <util/argparse.h>
#include <util/search.h>

struct ProgramArgs : public BenchmarkArgs {
  bool &cleanup = flag("cleanup", "Include heap cleanup in time");
};

int benchmark(int argc, char **argv) {
  auto args = argparse::parse<ProgramArgs>(argc, argv);
  auto agent_params = RuntimeSearch::AgentParams{
      .budget = args.budget.value_or(std::to_string(1 << 20)),
      .bandit = args.bandit.value_or("ucb-1.0"),
      .eval = args.eval.value_or("mc"),
      .matrix_ucb = args.matrix_ucb.value_or(""),
      .discrete = args.use_discrete,
      .table = args.use_table};
  auto agent = RuntimeSearch::Agent{agent_params};
  const uint32_t seed = 1111111;
  auto device = mt19937{seed};
  auto p1 = Teams::benchmark_teams[0];
  auto p2 = Teams::benchmark_teams[1];
  auto battle = PKMN::battle(p1, p2, seed);
  auto options = PKMN::options();
  const auto result = PKMN::update(battle, 0, 0, options);
  const auto durations = PKMN::durations();
  const auto input = MCTS::Input{battle, durations, result};
  const auto start = std::chrono::high_resolution_clock::now();
  auto output = MCTS::Output{};
  {
    auto heap = RuntimeSearch::Heap{};
    output = RuntimeSearch::run(device, input, heap, agent);
  }
  auto us = output.duration.count();
  if (args.cleanup) {
    const auto end = std::chrono::high_resolution_clock::now();
    us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }
  if (us >= 10000) {
    std::cout << (us / 1000) << "ms." << std::endl;
  } else {
    std::cout << us << "µs." << std::endl;
  }
  std::cout << output.iterations << " iterations." << std::endl;

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }
