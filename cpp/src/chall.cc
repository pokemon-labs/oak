#include <util/argparse.h>
#include <util/parse.h>
#include <util/policy.h>
#include <util/random.h>
#include <util/search.h>

#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

struct ProgramArgs : public ChallArgs {
  std::optional<uint64_t> &seed = kwarg("seed", "Global program seed");
  bool &use_budget = flag("--use-budget",
                          "Use --budget value instead of ctrl+z to end search");
};

bool search_flag = true;

void handle_suspend(int signal) {
  std::cout << "Search Suspended." << std::endl;
  search_flag = false;
}

MCTS::Input parse_input(const std::string &line, uint64_t seed) {
  auto [battle, durations] = Parse::parse_battle(line, seed);
  MCTS::randomize_hidden_variables(battle, durations);
  return {battle, durations, PKMN::result(battle)};
}

int main(int argc, char **argv) {

  auto args = argparse::parse<ProgramArgs>(argc, argv);

  auto agent_params = RuntimeSearch::AgentParams{
      .budget = args.budget.value_or(std::to_string(1 << 12)),
      .bandit = args.bandit.value_or("ucb-1.0"),
      .eval = args.eval.value_or("mc"),
      .matrix_ucb = args.matrix_ucb.value_or(""),
      .discrete = args.quantize,
      .table = args.use_table};
  auto agent = RuntimeSearch::Agent{agent_params};
  bool *const flag = args.use_budget ? nullptr : &search_flag;

  if (!args.seed.has_value()) {
    args.seed.emplace(std::random_device{}());
  }

  const auto policy_options =
      RuntimePolicy::Options{.mode = args.policy_mode.value_or("x"),
                             .temp = args.policy_temp.value_or(1.0),
                             .min = args.policy_min.value_or(0)};

  mt19937 device{args.seed.value()};
  MCTS::Input input;

  while (true) {
    std::string line;
    std::cout << "Input: battle-string" << std::endl;
    std::getline(std::cin, line);
    try {
      input = parse_input(line, args.seed.value());
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      continue;
    }
    break;
  }

  std::signal(SIGTSTP, handle_suspend);

  // set the durations inside the options to start
  auto options = PKMN::options();
  pkmn_gen1_chance_options chance_options{};
  chance_options.durations = input.durations;
  pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);

  while (!pkmn_result_type(input.result)) {
    std::cout << "\nBattle:" << std::endl;
    std::cout << PKMN::battle_data_to_string(input.battle, input.durations);
    const auto [p1_choices, p2_choices] =
        PKMN::choices(input.battle, input.result);
    const auto [p1_labels, p2_labels] =
        PKMN::choice_labels(input.battle, input.result);

    std::cout << "\nP1 choices:" << std::endl;
    for (auto i = 0; i < p1_choices.size(); ++i) {
      std::cout << i << ": " << p1_labels[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << "P2 choices:" << std::endl;
    for (auto i = 0; i < p2_choices.size(); ++i) {
      std::cout << i << ": " << p2_labels[i] << ' ';
    }
    std::cout << std::endl;

    std::cout << "Starting search. (Ctrl + Z) to pause." << std::endl;

    RuntimeSearch::Heap heap{};
    MCTS::Output output{};

    int p1_index = -1;
    int p2_index = -1;

    while (true) {
      search_flag = true;
      output = RuntimeSearch::run(device, input, heap, agent, output, flag);
      std::cout << output_string(output, input.battle, p1_labels, p2_labels);
      std::cout << "Input: P1 index (P2 index); Negative index = sample."
                << std::endl;
      std::string line;
      if (!std::getline(std::cin, line)) {
        std::cerr << "Input stream error.\n";
        continue;
      }

      std::istringstream iss(line);
      if (iss >> p1_index) {
        if (p1_index < (int)output.p1.k) {
          if (iss >> p2_index) {
            if (p2_index < (int)output.p2.k) {
              break;
            }
          } else {
            break;
          }
        }
      }
      std::cout << "Invalid index (pair). Resuming search." << std::endl;
    }

    if (p1_index < 0) {
      std::cout << "Sampling Player 1" << std::endl;
      p1_index =
          RuntimePolicy::process_and_sample(device, output.p1, policy_options);
    }
    if (p2_index < 0) {
      std::cout << "Sampling Player 2" << std::endl;
      p2_index =
          RuntimePolicy::process_and_sample(device, output.p2, policy_options);
    }

    auto c1 = p1_choices[p1_index];
    auto c2 = p2_choices[p2_index];

    std::cout << "Actions: " << p1_labels[p1_index] << ' '
              << p2_labels[p2_index] << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    input.result = PKMN::update(input.battle, c1, c2, options);
    input.durations = *pkmn_gen1_battle_options_chance_durations(&options);
  }

  std::cout << "\nBattle:" << std::endl;
  std::cout << PKMN::battle_data_to_string(input.battle, input.durations);
  std::cout << "Score: " << PKMN::score(input.result) << std::endl;

  return 0;
}
