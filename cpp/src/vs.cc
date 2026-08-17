#include <train/battle/compressed-frame.h>
#include <util/argparse.h>
#include <util/battle-frame-buffer.h>
#include <util/policy.h>
#include <util/random.h>
#include <util/search.h>
#include <util/team-building.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

// TODO claude says max_battle not used and build_traj never written (who cares)

struct ProgramArgs : public VsArgs {
  std::optional<uint64_t> &seed = kwarg("seed", "Global program seed");
  size_t &threads =
      kwarg("threads",
            "Number of parallel eval game pairs (players swap teams) to run")
          .set_default(std::max(1u, std::thread::hardware_concurrency()));
  size_t &max_games =
      kwarg("max-games", "Number of games to play before program termination")
          .set_default(1 << 14);
  bool &save = flag("save", "Sets --dir to a timestamp");
  std::string &teams_path =
      kwarg("teams", "Path to teams file").set_default("");
  bool &mirror_match = flag("mirror-match", "Use the same teams for p1, p2");
  std::optional<std::string> &working_dir = kwarg("dir", "Save directory");
  double &print_prob =
      kwarg("print-prob", "Probabilty to print any given battle state")
          .set_default(0);
  int &max_battle_length =
      kwarg("max-battle-length",
            "Battles exceeding this many updates are dropped")
          .set_default(-1);
  int &print_interval = kwarg("print-interval", "Seconds").set_default(15);

  size_t &buffer_size =
      kwarg("buffer-size", "Size of battle buffer (Mb) before write")
          .set_default(8);
  size_t &max_build_traj =
      kwarg("max-build-traj",
            "Size of build buffer (No. of traj's) before write")
          .set_default(1 << 10);

  std::optional<std::string> &p1_bandit_after = kwarg("p1-bandit-after", "");
  std::optional<std::string> &p2_bandit_after = kwarg("p2-bandit-after", "");
  std::optional<std::string> &p1_budget_after = kwarg("p1-budget-after", "");
  std::optional<std::string> &p2_budget_after = kwarg("p2-budget-after", "");
  std::optional<std::string> &p1_matrix_ucb_after =
      kwarg("p1-matrix-ucb-after", "");
  std::optional<std::string> &p2_matrix_ucb_after =
      kwarg("p2-matrix-ucb-after", "");
};

auto inverse_sigmoid(const auto x) { return std::log(x) - std::log(1 - x); }

constexpr float elo_conversion_factor = 400.0 / std::log(10);

void print(const auto &data, const bool newline = true) {
  std::cout << data;
  if (newline) {
    std::cout << '\n';
  }
}

namespace RuntimeData {
bool terminated = false;
bool suspended = false;

std::atomic<size_t> score{};
std::atomic<size_t> n{};

// filenames
std::atomic<size_t> battle_buffer_counter{};
std::atomic<size_t> build_buffer_counter{};

std::atomic<size_t> match_counter{};
std::atomic<size_t> win{};
std::atomic<size_t> loss{};
std::atomic<size_t> draw{};

std::vector<size_t> battle_lengths{};
std::vector<std::pair<MCTS::Output, MCTS::Output>> battle_outputs{};
std::atomic<size_t> thread_id{};

TeamBuilding::Provider provider;

void print_wdl() {
  std::cout << "W D L:\n";
  std::cout << win.load() << ' ' << draw.load() << ' ' << loss.load()
            << std::endl;
}

} // namespace RuntimeData

void thread_fn(const ProgramArgs *args_ptr) {
  const auto &args = *args_ptr;
  const auto id = RuntimeData::thread_id.fetch_add(1) % args.threads;
  mt19937 device = [&args, id]() {
    mt19937 d{args.seed.value()};
    for (auto i = 0; i < id; ++i) {
      d.uniform_64();
    }
    return d.uniform_64();
  }();

  // data gen
  const size_t training_frames_target_size = args.buffer_size << 20;
  const size_t thread_frame_buffer_size = (args.buffer_size + 1) << 20;

  auto p1_battle_frame_buffer = BattleFrameBuffer{thread_frame_buffer_size};
  auto p2_battle_frame_buffer = BattleFrameBuffer{thread_frame_buffer_size};

  // These are generated slowly so a vector is fine
  std::vector<Train::Build::Trajectory> build_buffer{};

  const auto save_build_buffer_to_disk = [&build_buffer, &args]() {
    if (build_buffer.empty()) {
      return;
    }
    const auto filename =
        std::to_string(RuntimeData::build_buffer_counter.fetch_add(1)) +
        ".build.data";
    const std::filesystem::path full_path =
        std::filesystem::path{args.working_dir.value()} / filename;
    std::ofstream out(full_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      std::cerr << "Failed to open " << full_path << " for writing\n";
      return;
    }
    const size_t bytes_to_write =
        build_buffer.size() *
        Encode::Build::CompressedTrajectory<>::size_no_team;
    size_t written = 0;
    for (const auto &trajectory : build_buffer) {
      const Encode::Build::CompressedTrajectory<> traj{trajectory};
      out.write(reinterpret_cast<const char *>(&traj),
                Encode::Build::CompressedTrajectory<>::size_no_team);
      written += Encode::Build::CompressedTrajectory<>::size_no_team;
    }
    if (written != bytes_to_write) {
      std::cerr << "Short write when flushing build buffer to " << full_path
                << " (" << written << '/' << bytes_to_write << " bytes)\n";
    }
    build_buffer.clear();
  };
  const auto play = [&](auto &p1_build_traj, auto &p2_build_traj) -> int {
    const auto &p1_team = p1_build_traj.terminal;
    const auto &p2_team = p2_build_traj.terminal;
    auto battle = PKMN::battle(p1_team, p2_team, device.uniform_64());
    auto options = PKMN::options();
    auto result = PKMN::update(battle, 0, 0, options);

    auto p1_eval = Search::Parse::eval(
        args.p1_eval.or_else([&] { return args.eval; }).value(),
        args.quantize || args.p1_quantize);
    auto p2_eval = Search::Parse::eval(
        args.p2_eval.or_else([&] { return args.eval; }).value(),
        args.quantize || args.p2_quantize);

    auto p1_s1_cache = Search::SideCache{};
    auto p1_s2_cache = Search::SideCache{};
    auto p2_s1_cache = Search::SideCache{};
    auto p2_s2_cache = Search::SideCache{};
    const bool p1_is_network = p1_eval.is_network();
    if (p1_is_network) {
      Search::Network network;
      network.data =
          std::get<std::shared_ptr<NN::Battle::NetworkBase>>(p1_eval.data);
      for (auto i = 0; i < 6; ++i) {
        p1_s1_cache.precompute(network.get(), PKMN::view(battle).sides[0], i);
        p1_s2_cache.precompute(network.get(), PKMN::view(battle).sides[1], i);
      }
      if (args.quantize || args.p1_quantize) {
        network.quantize();
        p1_s1_cache.quantize(network.get());
        p1_s2_cache.quantize(network.get());
      }
    }
    const bool p2_is_network = p2_eval.is_network();
    if (p2_is_network) {
      Search::Network network;
      network.data =
          std::get<std::shared_ptr<NN::Battle::NetworkBase>>(p2_eval.data);
      for (auto i = 0; i < 6; ++i) {
        p2_s1_cache.precompute(network.get(), PKMN::view(battle).sides[0], i);
        p2_s2_cache.precompute(network.get(), PKMN::view(battle).sides[1], i);
      }
      if (args.quantize || args.p2_quantize) {
        network.quantize();
        p2_s1_cache.quantize(network.get());
        p2_s2_cache.quantize(network.get());
      }
    }

    const auto p1_bandit = Search::Parse::bandit(
        args.p1_bandit.or_else([&] { return args.bandit; }).value());
    const auto p2_bandit = Search::Parse::bandit(
        args.p2_bandit.or_else([&] { return args.bandit; }).value());
    auto p1_budget = Search::Parse::budget(
        args.p1_budget.or_else([&] { return args.budget; }).value());
    auto p2_budget = Search::Parse::budget(
        args.p2_budget.or_else([&] { return args.budget; }).value());

    const auto p1_policy_options = RuntimePolicy::Options{
        .mode = args.p1_policy_mode.or_else([&] { return args.policy_mode; })
                    .value(),
        .temp = args.p1_policy_temp.or_else([&] { return args.policy_temp; })
                    .value_or(1),
        .min = args.p1_policy_min.or_else([&] { return args.policy_min; })
                   .value_or(0)};

    const auto p2_policy_options = RuntimePolicy::Options{
        .mode = args.p2_policy_mode.or_else([&] { return args.policy_mode; })
                    .value(),
        .temp = args.p2_policy_temp.or_else([&] { return args.policy_temp; })
                    .value_or(1),
        .min = args.p2_policy_min.or_else([&] { return args.policy_min; })
                   .value_or(0)};

    // const bool same_search =
    //     (p1_agent == p2_agent) && (p1_agent_after == p2_agent_after);
    const bool same_search = false;

    auto p1_battle_frames = Train::Battle::CompressedFrames{battle};
    auto p2_battle_frames = Train::Battle::CompressedFrames{battle};

    auto adjudicator = RuntimePolicy::JointValueHistory{};
    bool adjudicated = false;
    auto adj_result = PKMN::Result::None;

    size_t updates = 0;

    // playout game
    while (!pkmn_result_type(result)) {

      RuntimeData::battle_lengths[id] = updates;

      while (RuntimeData::suspended) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      if (RuntimeData::terminated) {
        using Ignored = int;
        return Ignored{};
      }

      const auto [p1_choices, p2_choices] = PKMN::choices(battle, result);

      const auto print_search_outputs = (device.uniform() < args.print_prob);
      const auto [p1_labels, p2_labels] = [&battle, result,
                                           print_search_outputs]() {
        if (print_search_outputs) {
          return PKMN::choice_labels(battle, result);
        }
        return std::pair<std::vector<std::string>, std::vector<std::string>>{};
      }();

      MCTS::Output p1_output{}, p2_output{};
      int p1_index{}, p2_index{};
      if (p1_choices.size() > 1) {
        Search::Node heap{};
        p1_output = RuntimeSearch::run(device, battle, PKMN::durations(options),
                                       p1_budget, p1_bandit, heap, p1_eval,
                                       p1_output, &p1_s1_cache, &p1_s2_cache);
        p1_index = process_and_sample(device, p1_output.p1, p1_policy_options);
        if (print_search_outputs) {
          print("P1:");
          std::cout << MCTS::output_string(p1_output, battle, p1_labels,
                                           p2_labels);
        }
      }

      if (p2_choices.size() > 1) {
        if (same_search && p1_choices.size() > 1) {
          p2_output = p1_output;
        } else {
          Search::Node heap{};
          p2_output = RuntimeSearch::run(
              device, battle, PKMN::durations(options), p2_budget, p2_bandit,
              heap, p2_eval, p2_output, &p2_s1_cache, &p2_s2_cache);
          if (print_search_outputs) {
            print("P2:");
            std::cout << MCTS::output_string(p2_output, battle, p1_labels,
                                             p2_labels);
          }
        }
        p2_index = process_and_sample(device, p2_output.p2, p2_policy_options);
      }

      RuntimeData::battle_outputs[id] = {p1_output, p2_output};

      if (updates == 0) {
        p1_build_traj.value = p1_output.empirical_value;
        p2_build_traj.value = 1 - p2_output.empirical_value;
      }

      adjudicator.update(p1_output, p1_policy_options, p2_output,
                         p2_policy_options);
      adj_result = adjudicator.check_for_consensus(args.forfeit_n.value(),
                                                   args.forfeit_value.value());
      if (adj_result != PKMN::Result::None) {
        adjudicated = true;
        break;
      }

      const auto p1_choice = p1_choices[p1_index];
      const auto p2_choice = p2_choices[p2_index];

      p1_battle_frames.updates.emplace_back(p1_output, p1_choice, p2_choice);
      p2_battle_frames.updates.emplace_back(p2_output, p1_choice, p2_choice);

      if (print_search_outputs) {
        print("GAME: " + std::to_string(RuntimeData::n.load()), false);
        print(" UPDATE: " + std::to_string(updates));
        print(PKMN::battle_data_to_string(battle, PKMN::durations(options)));
      }
      result = PKMN::update(battle, p1_choice, p2_choice, options);
      ++updates;
    }

    if (adjudicated) {
      result = PKMN::result(adj_result);
    }

    p1_battle_frames.result = result;
    p2_battle_frames.result = result;

    p1_battle_frame_buffer.write_frames(p1_battle_frames);
    p2_battle_frame_buffer.write_frames(p2_battle_frames);

    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_WIN: {
      RuntimeData::win.fetch_add(1);
      return 2;
    }
    case PKMN_RESULT_LOSE: {
      RuntimeData::loss.fetch_add(1);
      return 0;
    }
    case PKMN_RESULT_TIE: {
      RuntimeData::draw.fetch_add(1);
      return 1;
    }
    default: {
      assert(false);
      return 0;
    }
    }
  };

  while ((args.max_games < 0) ||
         (RuntimeData::match_counter.fetch_add(1) < args.max_games)) {

    auto [p1_build_traj, p1_team_index] =
        RuntimeData::provider.get_trajectory(device);
    auto p2_build_traj = p1_build_traj;
    auto p2_team_index = p1_team_index;
    if (!args.mirror_match) {
      std::tie(p2_build_traj, p2_team_index) =
          RuntimeData::provider.get_trajectory(device);
    }

    const auto s1 = play(p1_build_traj, p2_build_traj);
    auto s2 = 0;
    auto n = 0;
    if (!args.mirror_match) {
      s2 = play(p2_build_traj, p1_build_traj);
      n = 1;
    }
    if (!RuntimeData::terminated) {
      RuntimeData::score.fetch_add(s1 + s2);
      RuntimeData::n.fetch_add(1 + n);
    }
    if (args.working_dir.has_value()) {
      std::filesystem::path working_dir{args.working_dir.value()};
      if (p1_battle_frame_buffer.write_index >= training_frames_target_size) {
        p1_battle_frame_buffer.save_to_disk(working_dir / "p1",
                                            RuntimeData::battle_buffer_counter);
      }
      if (p2_battle_frame_buffer.write_index >= training_frames_target_size) {
        p2_battle_frame_buffer.save_to_disk(working_dir / "p2",
                                            RuntimeData::battle_buffer_counter);
      }
    }

    if (RuntimeData::terminated) {
      if (args.working_dir.has_value()) {
        std::filesystem::path working_dir{args.working_dir.value()};
        p1_battle_frame_buffer.save_to_disk(working_dir / "p1",
                                            RuntimeData::battle_buffer_counter);
        p2_battle_frame_buffer.save_to_disk(working_dir / "p2",
                                            RuntimeData::battle_buffer_counter);
      }
      save_build_buffer_to_disk();
      return;
    }

    if (build_buffer.size() >= args.max_build_traj) {
      save_build_buffer_to_disk();
    }
  }

  return;
}

void progress_thread_fn(const ProgramArgs *args_ptr) {
  const auto &args = *args_ptr;
  while (true) {
    for (int s = 0; s < args.print_interval; ++s) {
      if (RuntimeData::terminated) {
        return;
      }
      if ((args.max_games > 0) &&
          (RuntimeData::match_counter.load() >= args.max_games)) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    const double average_score =
        RuntimeData::score.load() / 2.0 / RuntimeData::n.load();
    const double elo_difference =
        inverse_sigmoid(average_score) * elo_conversion_factor;

    std::cout << "score: " << average_score << " over " << RuntimeData::n.load()
              << " games; Elo diff: " << elo_difference << std::endl;
    RuntimeData::print_wdl();

    std::cout << "info: " << std::endl;
    for (auto i = 0; i < args.threads; ++i) {
      const auto &outputs = RuntimeData::battle_outputs[i];
      std::cout << "\t" << i << ": " << RuntimeData::battle_lengths[i] << ", "
                << "(" << outputs.first.empirical_value << "/"
                << outputs.first.nash_value << "), " << "("
                << outputs.second.empirical_value << "/"
                << outputs.second.nash_value << ")";
      std::cout << std::endl;
    }
  }
}

void handle_suspend(int signal) {
  RuntimeData::suspended = !RuntimeData::suspended;
  std::cout << (RuntimeData::suspended ? "Suspended." : "Resumed.")
            << std::endl;
}

void handle_terminate(int signal) {
  RuntimeData::terminated = true;
  RuntimeData::suspended = false;
}

void setup(auto &args) {
  if (!args.seed.has_value()) {
    args.seed.emplace(std::random_device{}());
  }
  const auto check_args = [](auto x, auto y, auto z, const auto &name) {
    if (!x.has_value()) {
      if (!y.has_value() || !z.has_value()) {
        throw std::runtime_error{std::string{"--"} + name +
                                 " kwarg is required."};
      }
    }
  };

  check_args(args.budget, args.p1_budget, args.p2_budget, "budget");
  check_args(args.eval, args.p1_eval, args.p2_eval, "eval");
  check_args(args.bandit, args.p1_bandit, args.p2_bandit, "bandit");
  check_args(args.policy_mode, args.p1_policy_mode, args.p2_policy_mode,
             "policy-mode");
  // args
  if (args.save && !args.working_dir.has_value()) {
    args.working_dir.emplace("vs-" + get_current_datetime());
  }
  // create working dir
  if (args.working_dir.has_value()) {
    const std::filesystem::path working_dir = args.working_dir.value();
    std::error_code ec;
    bool created = std::filesystem::create_directory(working_dir, ec) &&
                   std::filesystem::create_directory(working_dir / "p1", ec) &&
                   std::filesystem::create_directory(working_dir / "p2", ec);
    if (ec) {
      std::cerr << "Error creating directory: " << ec.message() << '\n';
      throw std::runtime_error("Could not create working dir.");
    } else if (created) {
      std::cout << "Created directory " << working_dir.string() << std::endl;
    } else {
      throw std::runtime_error("Could not create working dir.");
    }

    // save args
    std::ofstream args_file(std::filesystem::path{working_dir} / "args");
    if (!args_file) {
      throw std::runtime_error("Failed to open args file for writing.");
    }
    args.print(args_file);
  }

  // teams
  RuntimeData::provider = TeamBuilding::Provider{args.teams_path};
  RuntimeData::provider.omitter = {args.max_pokemon, args.pokemon_delete_prob,
                                   args.move_delete_prob};
  RuntimeData::provider.network_path = args.build_network_path;
  RuntimeData::provider.team_modify_prob = args.team_modify_prob;
  RuntimeData::provider.read_network_parameters();

  // stats
  RuntimeData::battle_lengths.resize(args.threads);
  RuntimeData::battle_outputs.resize(args.threads);
}

void thread_fn_wrapper(ProgramArgs *args) {
  try {
    thread_fn(args);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    std::exit(1);
  }
}

int main(int argc, char **argv) {

  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  auto args = argparse::parse<ProgramArgs>(argc, argv);

  setup(args);

  std::vector<std::thread> thread_pool{};
  for (auto t = 0; t < args.threads; ++t) {
    thread_pool.emplace_back(std::thread{&thread_fn_wrapper, &args});
  }
  auto progress_thread = std::thread(&progress_thread_fn, &args);
  for (auto &thread : thread_pool) {
    thread.join();
  }
  progress_thread.join();

  std::cout << "score: "
            << (RuntimeData::score.load() / 2.0 / RuntimeData::n.load())
            << " over " << RuntimeData::n.load() << " games." << std::endl;
  RuntimeData::print_wdl();
}