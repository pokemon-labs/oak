// sprt: vs with a sequential stopping rule.
//
// Games are set up and played exactly as in vs.cc (same args, same team-swapped
// match pairs, same search calls). The difference is when it stops: after every
// completed pair the log-likelihood ratio of H1 (Elo >= elo1) against
// H0 (Elo <= elo0) is updated, and the run stops as soon as it crosses
//   lower = ln(beta / (1 - alpha))   -> accept H0
//   upper = ln((1 - beta) / alpha)   -> accept H1
// --max-games (pairs, as in vs) truncates the test; the result is then
// inconclusive.
//
// The LLR is the generalized SPRT of Michel Van den Bergh (as used by
// fishtest): the observed outcome distribution is replaced, for each
// hypothesis, by the maximum-likelihood multinomial with the hypothesised mean
// score, and LLR = N * sum_i p_i * ln(q1_i / q0_i).
//   pair (default): one sample per team-swapped pair, score in
//     {0, 1/4, 1/2, 3/4, 1} (pentanomial). Draws are rare in Pokemon, so in
//     practice this is the pair trinomial LL / split / WW = {0, 1/2, 1}.
//     Pairing removes the team-matchup variance: a 420-380 record is strong
//     evidence if most pairs split, weak if many pairs are WW or LL.
//   game: one sample per game, score in {0, 1/2, 1}. Only valid with
//     --mirror-match, where there are no team-swapped pairs.
// Elo is logistic: score = 1 / (1 + 10^(-elo / 400)), from P1's view.
//
// Counts (score, W D L, pentanomial) only include fully completed pairs that
// were recorded before the decision. Games in flight when the test stops are
// aborted and discarded; their outcome is independent of the decision, so this
// does not bias the result.

#include <train/battle/compressed-frames.h>
#include <util/argparse.h>
#include <util/battle-frame-buffer.h>
#include <util/cache-pool.h>
#include <util/policy.h>
#include <util/random.h>
#include <util/search.h>
#include <util/team-building.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

struct ProgramArgs : public VsArgs {
  std::optional<uint64_t> &seed = kwarg("seed", "Global program seed");
  size_t &threads =
      kwarg("threads",
            "Number of parallel eval game pairs (players swap teams) to run")
          .set_default(std::max(1u, std::thread::hardware_concurrency()));
  size_t &max_games =
      kwarg("max-games",
            "Max number of game pairs before the test is truncated "
            "(inconclusive). Each pair is 2 games unless --mirror-match")
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
  bool &cache_pool =
      flag("cache-pool", "Use a std::map of side caches - not appropriate for "
                         "RL but faster otherwise.");
  bool &reuse =
      flag("reuse", "Player 2 will search using Player 1's heap and output.");

  // SPRT
  double &elo0 = kwarg("elo0", "SPRT H0: P1 Elo advantage <= elo0 (logistic)")
                     .set_default(0);
  double &elo1 = kwarg("elo1", "SPRT H1: P1 Elo advantage >= elo1 (logistic)")
                     .set_default(30);
  double &alpha =
      kwarg("alpha", "SPRT false positive rate (accept H1 when H0 is true)")
          .set_default(0.05);
  double &beta =
      kwarg("beta", "SPRT false negative rate (accept H0 when H1 is true)")
          .set_default(0.05);
  std::string &sprt_model =
      kwarg("sprt-model",
            "pair (one sample per team-swapped pair) or game (one sample per "
            "game; only with --mirror-match, which it is forced by)")
          .set_default("pair");
  size_t &min_samples =
      kwarg("min-samples",
            "No stopping before this many samples (pairs, or games with "
            "--sprt-model=game)")
          .set_default(8);
};

auto inverse_sigmoid(const auto x) { return std::log(x) - std::log(1 - x); }

constexpr float elo_conversion_factor = 400.0 / std::log(10);

void print(const auto &data, const bool newline = true) {
  std::cout << data;
  if (newline) {
    std::cout << '\n';
  }
}

std::string format_duration_us(double microseconds) {
  std::ostringstream oss;
  if (microseconds >= 10000.0) {
    oss << (microseconds / 1000.0) << "ms";
  } else {
    oss << microseconds << "us";
  }
  return oss.str();
}

namespace SPRT {

double elo_to_score(double elo) {
  return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0));
}

double score_to_elo(double s) {
  s = std::clamp(s, 1e-9, 1 - 1e-9);
  return -400.0 * std::log10(1.0 / s - 1.0);
}

// Outcome scores are a_i = i / (K - 1), so a_0 = 0 and a_{K-1} = 1.
template <size_t K> constexpr double outcome(size_t i) {
  return static_cast<double>(i) / (K - 1);
}

// Empirical distribution with empty bins set to a small count (as fishtest
// does) so that the constrained MLE exists for any hypothesised mean in (0, 1).
template <size_t K>
std::array<double, K> regularized(const std::array<size_t, K> &counts) {
  std::array<double, K> p{};
  double total = 0;
  for (size_t i = 0; i < K; ++i) {
    p[i] = counts[i] > 0 ? static_cast<double>(counts[i]) : 1e-3;
    total += p[i];
  }
  for (auto &x : p) {
    x /= total;
  }
  return p;
}

// Max-likelihood distribution q with sum_i q_i a_i = s, given empirical p:
// q_i = p_i / (1 + lambda (a_i - s)), lambda the root of
// f(lambda) = sum_i p_i (a_i - s) / (1 + lambda (a_i - s)), which is strictly
// decreasing on (-1 / (1 - s), 1 / s).
template <size_t K>
std::array<double, K> mle(const std::array<double, K> &p, double s) {
  constexpr double eps = 1e-12;
  double lo = -1.0 / (1.0 - s) + eps;
  double hi = 1.0 / s - eps;
  const auto f = [&](double lambda) {
    double sum = 0;
    for (size_t i = 0; i < K; ++i) {
      const double d = outcome<K>(i) - s;
      sum += p[i] * d / (1 + lambda * d);
    }
    return sum;
  };
  for (int it = 0; it < 200; ++it) {
    const double mid = (lo + hi) / 2;
    if (f(mid) > 0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const double lambda = (lo + hi) / 2;
  std::array<double, K> q{};
  double total = 0;
  for (size_t i = 0; i < K; ++i) {
    q[i] = p[i] / (1 + lambda * (outcome<K>(i) - s));
    total += q[i];
  }
  for (auto &x : q) {
    x /= total;
  }
  return q;
}

template <size_t K>
double llr(const std::array<size_t, K> &counts, double s0, double s1) {
  size_t n = 0;
  for (const auto c : counts) {
    n += c;
  }
  if (n == 0) {
    return 0;
  }
  const auto p = regularized(counts);
  const auto q0 = mle(p, s0);
  const auto q1 = mle(p, s1);
  double sum = 0;
  for (size_t i = 0; i < K; ++i) {
    sum += p[i] * std::log(q1[i] / q0[i]);
  }
  return n * sum;
}

// Mean score per sample and its standard error, from the raw counts.
template <size_t K>
std::pair<double, double> mean_se(const std::array<size_t, K> &counts) {
  size_t n = 0;
  double mean = 0;
  for (size_t i = 0; i < K; ++i) {
    n += counts[i];
    mean += counts[i] * outcome<K>(i);
  }
  if (n == 0) {
    return {0.5, 0};
  }
  mean /= n;
  double var = 0;
  for (size_t i = 0; i < K; ++i) {
    const double d = outcome<K>(i) - mean;
    var += counts[i] * d * d;
  }
  var /= n;
  return {mean, std::sqrt(var / n)};
}

enum class Verdict { None, H0, H1, Truncated, Interrupted };

std::string verdict_string(Verdict v) {
  switch (v) {
  case Verdict::H0:
    return "H0 accepted";
  case Verdict::H1:
    return "H1 accepted";
  case Verdict::Truncated:
    return "inconclusive (max-games reached)";
  case Verdict::Interrupted:
    return "inconclusive (interrupted)";
  default:
    return "none";
  }
}

struct State {
  std::mutex m{};
  // pentanomial: index = sum of the pair's half-points (0..4), P1's view
  std::array<size_t, 5> penta{};
  // per game: 0 loss, 1 draw, 2 win, P1's view
  std::array<size_t, 3> wdl{};
  bool use_pairs = true;
  double s0 = 0.5, s1 = 0.5, lower = 0, upper = 0;
  size_t min_samples = 0;
  double last_llr = 0;
  Verdict verdict = Verdict::None;

  size_t samples() const {
    size_t n = 0;
    if (use_pairs) {
      for (auto c : penta) {
        n += c;
      }
    } else {
      for (auto c : wdl) {
        n += c;
      }
    }
    return n;
  }

  size_t games() const { return wdl[0] + wdl[1] + wdl[2]; }

  double compute_llr() const {
    return use_pairs ? llr(penta, s0, s1) : llr(wdl, s0, s1);
  }

  std::pair<double, double> estimate() const {
    return use_pairs ? mean_se(penta) : mean_se(wdl);
  }
};

State state{};

} // namespace SPRT

namespace RuntimeData {
bool terminated = false;
bool suspended = false;
bool interrupted = false;

// filenames
std::atomic<size_t> battle_buffer_counter{};
std::atomic<size_t> build_buffer_counter{};

std::atomic<size_t> match_counter{};

std::vector<size_t> battle_lengths{};
std::vector<std::pair<MCTS::Output, MCTS::Output>> battle_outputs{};
std::atomic<size_t> thread_id{};

struct WelfordStats {
  std::mutex m{};
  size_t count{};
  double mean{};
  double m2{};

  void update(double value) {
    std::lock_guard<std::mutex> lock{m};
    ++count;
    const double delta = value - mean;
    mean += delta / count;
    const double delta2 = value - mean;
    m2 += delta * delta2;
  }

  double variance() const { return count > 1 ? m2 / (count - 1) : 0.0; }
};

struct SearchStats {
  WelfordStats iterations{};
  WelfordStats duration{};
};

SearchStats p1_search_stats{};
SearchStats p2_search_stats{};

TeamBuilding::Provider provider;

CachePool p1_cache_pool{};
CachePool p2_cache_pool{};
} // namespace RuntimeData

// Record one completed match (s2 unused with mirror match). Called with the
// half-point results of P1 (0 loss, 1 draw, 2 win). Stops the run on a
// decision.
void record_match(int s1, int s2, bool mirror) {
  auto &st = SPRT::state;
  std::lock_guard<std::mutex> lock{st.m};
  if (st.verdict != SPRT::Verdict::None) {
    return;
  }
  ++st.wdl[s1];
  if (!mirror) {
    ++st.wdl[s2];
    ++st.penta[s1 + s2];
  }
  st.last_llr = st.compute_llr();
  if (st.samples() < st.min_samples) {
    return;
  }
  if (st.last_llr >= st.upper) {
    st.verdict = SPRT::Verdict::H1;
  } else if (st.last_llr <= st.lower) {
    st.verdict = SPRT::Verdict::H0;
  }
  if (st.verdict != SPRT::Verdict::None) {
    RuntimeData::terminated = true;
    RuntimeData::suspended = false;
  }
}

// vs-compatible score/W D L/search-stats block, then the SPRT lines. Caller
// holds no locks.
void print_status(const ProgramArgs &args, bool final) {
  auto &st = SPRT::state;
  std::array<size_t, 5> penta;
  std::array<size_t, 3> wdl;
  double llr;
  SPRT::Verdict verdict;
  std::pair<double, double> est;
  size_t samples;
  {
    std::lock_guard<std::mutex> lock{st.m};
    penta = st.penta;
    wdl = st.wdl;
    llr = st.last_llr;
    verdict = st.verdict;
    est = st.estimate();
    samples = st.samples();
  }
  const size_t games = wdl[0] + wdl[1] + wdl[2];
  const double score =
      games ? (wdl[2] + 0.5 * wdl[1]) / static_cast<double>(games) : 0.5;

  if (final) {
    std::cout << "score: " << score << " over " << games << " games."
              << std::endl;
  } else {
    std::cout << "score: " << score << " over " << games
              << " games; Elo diff: "
              << inverse_sigmoid(score) * elo_conversion_factor << std::endl;
  }
  std::cout << "W D L:\n" << wdl[2] << ' ' << wdl[1] << ' ' << wdl[0] << std::endl;

  std::cout << "search stats (iterations mean/stdev, duration mean/stdev):"
            << std::endl;
  const auto print_stats = [](const char *name, const auto &s) {
    std::cout << "\t" << name << ": " << s.iterations.mean << "/"
              << std::sqrt(s.iterations.variance()) << ", "
              << format_duration_us(s.duration.mean) << "/"
              << format_duration_us(std::sqrt(s.duration.variance()))
              << std::endl;
  };
  print_stats("p1", RuntimeData::p1_search_stats);
  print_stats("p2", RuntimeData::p2_search_stats);

  const auto [mean, se] = est;
  const double elo = SPRT::score_to_elo(mean);
  const double elo_lo = SPRT::score_to_elo(mean - 1.96 * se);
  const double elo_hi = SPRT::score_to_elo(mean + 1.96 * se);
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "sprt: LLR " << llr << " (" << st.lower << ", " << st.upper
            << ") elo0 " << args.elo0 << " elo1 " << args.elo1 << " alpha "
            << args.alpha << " beta " << args.beta << " model "
            << (st.use_pairs ? "pair" : "game") << std::endl;
  std::cout << std::setprecision(1);
  std::cout << "sprt: Elo " << elo << " [" << elo_lo << ", " << elo_hi
            << "] (95%) over " << samples << (st.use_pairs ? " pairs" : " games")
            << std::endl;
  std::cout << std::defaultfloat << std::setprecision(6);
  if (st.use_pairs) {
    std::cout << "sprt: penta [0 1/4 1/2 3/4 1]: " << penta[0] << ' '
              << penta[1] << ' ' << penta[2] << ' ' << penta[3] << ' '
              << penta[4] << std::endl;
  }
  if (final) {
    std::cout << "SPRT: " << SPRT::verdict_string(verdict) << std::endl;
  }
}

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

  const size_t training_frames_target_size = args.buffer_size << 20;
  const size_t thread_frame_buffer_size = (args.buffer_size + 1) << 20;

  auto p1_battle_frame_buffer = BattleFrameBuffer{thread_frame_buffer_size};
  auto p2_battle_frame_buffer = BattleFrameBuffer{thread_frame_buffer_size};

  // Returns P1's half-points: 2 win, 1 draw, 0 loss. Unchanged from vs except
  // that W/D/L are counted per completed pair in record_match.
  const auto play = [&](auto &p1_build_traj, auto &p2_build_traj) -> int {
    const auto &p1_team = p1_build_traj.terminal;
    const auto &p2_team = p2_build_traj.terminal;
    auto battle = PKMN::battle(p1_team, p2_team, device.uniform_64());
    auto options = PKMN::options();
    auto result = PKMN::update(battle, 0, 0, options);

    assert(args.p1_eval.has_value());
    assert(args.p1_bandit.has_value());
    assert(args.p1_budget.has_value());
    assert(args.p1_policy_mode.has_value());
    assert(args.p2_eval.has_value());
    assert(args.p2_bandit.has_value());
    assert(args.p2_budget.has_value());
    assert(args.p2_policy_mode.has_value());

    auto p1_eval = Search::Parse::eval(args.p1_eval.value(),
                                       args.quantize || args.p1_quantize);
    auto p2_eval = Search::Parse::eval(args.p2_eval.value(),
                                       args.quantize || args.p2_quantize);
    auto p1_s1_cache = RuntimeData::p1_cache_pool.get(
        p1_eval, PKMN::view(battle).sides[0], args.cache_pool);
    auto p1_s2_cache = RuntimeData::p1_cache_pool.get(
        p1_eval, PKMN::view(battle).sides[1], args.cache_pool);
    auto p2_s1_cache = RuntimeData::p2_cache_pool.get(
        p2_eval, PKMN::view(battle).sides[0], args.cache_pool);
    auto p2_s2_cache = RuntimeData::p2_cache_pool.get(
        p2_eval, PKMN::view(battle).sides[1], args.cache_pool);

    const auto p1_bandit = Search::Parse::bandit(args.p1_bandit.value());
    const auto p2_bandit = Search::Parse::bandit(args.p2_bandit.value());
    const auto p1_matrix_ucb =
        args.p1_matrix_ucb.has_value()
            ? Search::Parse::matrix_ucb(p1_bandit, args.p1_matrix_ucb.value())
            : Search::MatrixUCB(p1_bandit, 0);
    const auto p2_matrix_ucb =
        args.p2_matrix_ucb.has_value()
            ? Search::Parse::matrix_ucb(p2_bandit, args.p2_matrix_ucb.value())
            : Search::MatrixUCB(p2_bandit, 0);
    const auto &p1_params =
        args.p1_matrix_ucb.has_value()
            ? static_cast<const Search::Bandit &>(p1_matrix_ucb)
            : p1_bandit;
    const auto &p2_params =
        args.p2_matrix_ucb.has_value()
            ? static_cast<const Search::Bandit &>(p2_matrix_ucb)
            : p2_bandit;

    auto p1_budget = Search::Parse::budget(args.p1_budget.value());
    auto p2_budget = Search::Parse::budget(args.p2_budget.value());

    auto p1_options = Search::Parse::options(args.p1_options.value_or(""));
    auto p2_options = Search::Parse::options(args.p2_options.value_or(""));

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

    auto p1_battle_frames = Train::Battle::CompressedFrames{battle};
    auto p2_battle_frames = Train::Battle::CompressedFrames{battle};

    auto adjudicator = RuntimePolicy::JointValueHistory{};
    bool adjudicated = false;
    auto adj_result = PKMN::Result::None;

    size_t updates = 0;

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
      auto p1_heap = Search::Parse::heap(p1_params);
      auto p2_heap = Search::Parse::heap(p2_params);
      int p1_index{}, p2_index{};
      if (p1_choices.size() > 1 || args.reuse) {
        p1_output = RuntimeSearch::run(device, battle, PKMN::durations(options),
                                       p1_budget, p1_params, p1_heap, p1_eval,
                                       p1_output, p1_options, p1_s1_cache.get(),
                                       p1_s2_cache.get());
        RuntimeData::p1_search_stats.iterations.update(
            static_cast<double>(p1_output.iterations));
        RuntimeData::p1_search_stats.duration.update(
            static_cast<double>(p1_output.duration.count()));
        if (!args.reuse) {
          p1_heap.reset();
        }
        p1_index = process_and_sample(device, p1_output.p1, p1_policy_options);
        if (print_search_outputs) {
          print("P1:");
          print(MCTS::output_string(p1_output, battle, p1_labels, p2_labels));
        }
      }

      if (p2_choices.size() > 1) {
        p2_output = RuntimeSearch::run(
            device, battle, PKMN::durations(options), p2_budget, p2_params,
            args.reuse ? static_cast<Search::Heap &>(p1_heap) : p2_heap,
            p2_eval, args.reuse ? p1_output : p2_output, p2_options,
            p2_s1_cache.get(), p2_s2_cache.get());
        RuntimeData::p2_search_stats.iterations.update(
            static_cast<double>(p2_output.iterations));
        RuntimeData::p2_search_stats.duration.update(
            static_cast<double>(p2_output.duration.count()));
        p2_heap.reset();
        p2_index = process_and_sample(device, p2_output.p2, p2_policy_options);
        if (print_search_outputs) {
          print("P2:");
          print(MCTS::output_string(p2_output, battle, p1_labels, p2_labels));
        }
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
        print("MATCH: " + std::to_string(RuntimeData::match_counter.load()),
              false);
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
      return 2;
    }
    case PKMN_RESULT_LOSE: {
      return 0;
    }
    case PKMN_RESULT_TIE: {
      return 1;
    }
    default: {
      assert(false);
      return 0;
    }
    }
  };

  while (!RuntimeData::terminated &&
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
    if (!args.mirror_match) {
      s2 = play(p2_build_traj, p1_build_traj);
    }
    if (!RuntimeData::terminated) {
      record_match(s1, s2, args.mirror_match);
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
  }

  if (args.working_dir.has_value()) {
    std::filesystem::path working_dir{args.working_dir.value()};
    p1_battle_frame_buffer.save_to_disk(working_dir / "p1",
                                        RuntimeData::battle_buffer_counter);
    p2_battle_frame_buffer.save_to_disk(working_dir / "p2",
                                        RuntimeData::battle_buffer_counter);
  }
}

void progress_thread_fn(const ProgramArgs *args_ptr) {
  const auto &args = *args_ptr;
  while (true) {
    for (int s = 0; s < args.print_interval; ++s) {
      if (RuntimeData::terminated) {
        return;
      }
      if (RuntimeData::match_counter.load() >= args.max_games) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    print_status(args, false);

    std::cout << "info: " << std::endl;
    for (auto i = 0; i < args.threads; ++i) {
      const auto &outputs = RuntimeData::battle_outputs[i];
      std::cout << "\t" << i << ": " << RuntimeData::battle_lengths[i] << ", "
                << "(" << outputs.first.empirical_value << "/"
                << outputs.first.nash_value << " " << outputs.first.iterations
                << "), " << "(" << outputs.second.empirical_value << "/"
                << outputs.second.nash_value << " " << outputs.second.iterations
                << ")";
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
  RuntimeData::interrupted = true;
  RuntimeData::terminated = true;
  RuntimeData::suspended = false;
}

void setup(auto &args) {
  if (!args.seed.has_value()) {
    args.seed.emplace(std::random_device{}());
  }
  const auto check_args = [](const auto &x, auto &y, auto &z, const auto &name,
                             bool required = true) {
    if (!y.has_value()) {
      if (!x.has_value() && required) {
        throw std::runtime_error{std::string{"--"} + name +
                                 " kwarg is required."};
      } else {
        y = x.value();
        std::cout << "setting p1 " << name << " =  " << y.value() << std::endl;
      }
    }
    if (!z.has_value()) {
      if (!x.has_value() && required) {
        throw std::runtime_error{std::string{"--"} + name +
                                 " kwarg is required."};
      } else {
        z = x.value();
        std::cout << "setting p2 " << name << " =  " << z.value() << std::endl;
      }
    }
  };

  check_args(args.budget, args.p1_budget, args.p2_budget, "budget");
  check_args(args.eval, args.p1_eval, args.p2_eval, "eval");
  check_args(args.bandit, args.p1_bandit, args.p2_bandit, "bandit");
  check_args(args.policy_mode, args.p1_policy_mode, args.p2_policy_mode,
             "policy-mode");

  // SPRT
  if (!(args.elo1 > args.elo0)) {
    throw std::runtime_error{"--elo1 must be greater than --elo0."};
  }
  if (!(args.alpha > 0 && args.alpha < 1 && args.beta > 0 && args.beta < 1)) {
    throw std::runtime_error{"--alpha and --beta must be in (0, 1)."};
  }
  if (args.sprt_model != "pair" && args.sprt_model != "game") {
    throw std::runtime_error{"--sprt-model must be pair or game."};
  }
  if (args.sprt_model == "game" && !args.mirror_match) {
    throw std::runtime_error{
        "--sprt-model=game ignores the team-swapped pairing and overstates "
        "the variance; it is only valid with --mirror-match."};
  }
  auto &st = SPRT::state;
  st.use_pairs = !args.mirror_match;
  if (args.sprt_model == "pair" && args.mirror_match) {
    std::cout << "--mirror-match has no team-swapped pairs; using game model"
              << std::endl;
  }
  st.s0 = SPRT::elo_to_score(args.elo0);
  st.s1 = SPRT::elo_to_score(args.elo1);
  st.lower = std::log(args.beta / (1 - args.alpha));
  st.upper = std::log((1 - args.beta) / args.alpha);
  st.min_samples = args.min_samples;
  std::cout << "SPRT: H0 elo <= " << args.elo0 << ", H1 elo >= " << args.elo1
            << ", alpha " << args.alpha << ", beta " << args.beta
            << ", bounds (" << st.lower << ", " << st.upper << "), model "
            << (st.use_pairs ? "pair" : "game") << std::endl;

  if (args.save && !args.working_dir.has_value()) {
    args.working_dir.emplace("sprt-" + get_current_datetime());
  }
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

    std::ofstream args_file(std::filesystem::path{working_dir} / "args");
    if (!args_file) {
      throw std::runtime_error("Failed to open args file for writing.");
    }
    args.print(args_file);
  }

  RuntimeData::provider = TeamBuilding::Provider{args.teams_path};
  RuntimeData::provider.omitter = {args.max_pokemon, args.pokemon_delete_prob,
                                   args.move_delete_prob};
  RuntimeData::provider.network_path = args.build_network_path;
  RuntimeData::provider.team_modify_prob = args.team_modify_prob;
  RuntimeData::provider.read_network_parameters();

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

  {
    auto &st = SPRT::state;
    std::lock_guard<std::mutex> lock{st.m};
    if (st.verdict == SPRT::Verdict::None) {
      st.verdict = RuntimeData::interrupted ? SPRT::Verdict::Interrupted
                                            : SPRT::Verdict::Truncated;
    }
  }
  print_status(args, true);
}
