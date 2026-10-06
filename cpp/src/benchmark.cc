#include <teams/benchmark-teams.h>
#include <util/argparse.h>
#include <util/cache-pool.h>
#include <util/search.h>

struct ProgramArgs : public BenchmarkArgs {
  bool &cleanup = flag("cleanup", "Include heap cleanup in time");
  size_t &threads =
      kwarg("threads", "All threads use the same seed.").set_default(1);
};

struct Budget {
  std::chrono::microseconds duration;
  size_t iterations;
};
std::vector<Budget> budgets;

void benchmark(ProgramArgs *args_ptr, int index) {
  const uint32_t seed = 1111111;
  auto device = mt19937{seed};
  auto p1 = Teams::benchmark_teams[0];
  auto p2 = Teams::benchmark_teams[1];
  auto battle = PKMN::battle(p1, p2, seed);
  auto options = PKMN::options();
  const auto result = PKMN::update(battle, 0, 0, options);
  const auto durations = PKMN::durations();

  const auto &args = *args_ptr;
  auto eval = Search::Parse::eval(args.eval.value_or("mc"), args.quantize);
  auto cache_pool = CachePool{};
  auto p1_cache = cache_pool.get(eval, PKMN::view(battle).sides[0], false);
  auto p2_cache = cache_pool.get(eval, PKMN::view(battle).sides[1], false);
  auto bandit = Search::Parse::bandit(args.bandit.value_or("ucb-1.0"));
  auto matrix_ucb =
      args.matrix_ucb.has_value()
          ? Search::Parse::matrix_ucb(bandit, args.matrix_ucb.value())
          : Search::MatrixUCB(bandit, 0);
  auto &params = args.matrix_ucb.has_value()
                     ? static_cast<Search::Bandit &>(matrix_ucb)
                     : bandit;
  auto budget =
      Search::Parse::budget(args.budget.value_or(std::to_string(1 << 20)));
  auto search_options = Search::Parse::options(args.options.value_or(""));

  const auto start = std::chrono::high_resolution_clock::now();
  auto output = MCTS::Output{};
  {
    auto heap = Search::Parse::heap(params);
    output = RuntimeSearch::run(device, battle, PKMN::durations(options),
                                budget, params, heap, eval, output,
                                search_options, p1_cache.get(), p2_cache.get());
  }
  auto microseconds = output.duration;
  if (args.cleanup) {
    const auto end = std::chrono::high_resolution_clock::now();
    microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  }
  budgets[index].duration = microseconds;
  budgets[index].iterations = output.iterations;
}

int main(int argc, char **argv) {
  auto args = argparse::parse<ProgramArgs>(argc, argv);
  budgets.resize(args.threads);
  std::vector<std::thread> thread_pool;
  for (int i = 0; i < args.threads; ++i) {
    thread_pool.emplace_back(&benchmark, &args, i);
  }

  for (auto &th : thread_pool) {
    th.join();
  }

  for (int i = 0; i < args.threads; ++i) {
    std::cout << "thread " << i << std::endl;
    const auto microseconds = budgets[i].duration.count();
    const auto iterations = budgets[i].iterations;
    if (microseconds >= 10000) {
      std::cout << (microseconds / 1000) << "ms. ";
    } else {
      std::cout << microseconds << "µs. ";
    }
    std::cout << iterations << " iterations." << std::endl;
  }
}
