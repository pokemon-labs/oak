#pragma once

#include <search/mcts.h> // TODO remove
#include <util/strings.h>

namespace RuntimePolicy {

struct Options {
  std::string mode = "e";
  double temp = 1;
  double min = 0;
};

enum class Mode : char {
  prior = 'p',
  empirical = 'e',
  nash = 'n',
  argmax = 'x',
  beta = 'b',
};

inline std::array<double, 9> get_policy(const auto &side, const auto &options) {
  const auto &prior = side.prior;
  const auto &empirical = side.empirical;
  const auto &nash = side.nash;
  const auto &beta = side.beta;

  std::array<double, 9> policy{};

  const auto mode_split = Parse::split(options.mode, '-');

  for (std::string_view word : mode_split) {
    const double w =
        (word.size() > 1) ? std::stod(std::string(word.substr(1))) : 1.0;

    switch (static_cast<Mode>(word[0])) {
    case Mode::prior: {
      std::transform(prior.begin(), prior.end(), policy.begin(), policy.begin(),
                     [w](double q, double p) { return p + w * q; });
    }
    case Mode::empirical: {
      std::transform(empirical.begin(), empirical.end(), policy.begin(),
                     policy.begin(),
                     [w](double e, double p) { return p + w * e; });
      break;
    }
    case Mode::nash: {
      std::transform(nash.begin(), nash.end(), policy.begin(), policy.begin(),
                     [w](double n, double p) { return p + w * n; });
      break;
    }
    case Mode::argmax: {
      const auto it = std::max_element(empirical.begin(), empirical.end());
      const size_t idx = std::distance(empirical.begin(), it);
      policy[idx] += w;
      break;
    }
    case Mode::beta: {
      throw std::runtime_error{"RuntimePolicy: (b)eta mode disabled for now."};
      std::transform(beta.begin(), beta.end(), policy.begin(), policy.begin(),
                     [w](double b, double p) { return p + w * b; });
      break;
    }
    default: {
      throw std::runtime_error{"RuntimePolicy: invalid mode char: " + word[0]};
    }
    }
  }

  if (options.temp != 1) {
    double sum = 0;
    for (auto &x : policy) {
      x = std::pow(x, options.temp);
      sum += x;
    }
    for (auto &x : policy) {
      x /= sum;
    }
  }

  double sum = 0;
  for (auto &x : policy) {
    if (x < options.min) {
      x = 0;
    }
    sum += x;
  }
  if (sum == 0) {
    MCTS::print_side(side);
    throw std::runtime_error{"RuntimePolicy: zero policy, mode: " +
                             options.mode};
  }
  for (auto &x : policy) {
    x /= sum;
  }

  return policy;
}

inline int process_and_sample(auto &device, const auto &side,
                              const auto &policy_options) {
  const auto p = get_policy(side, policy_options);
  const auto index = device.sample_pdf(p);
  assert(p[index] > 0);
  return index;
}

class AdjuticateParams {
  double is = 0;

public:
  size_t consecutive = 1;

  void set_from_odds(const std::string &odds) {
    odds_split = Parse::split(odds, ':');
    try {
      double x = std::stod(odds_split[0]);
      double y = std::stod(odds_split[1]);
      is = std::abs(inverse_sigmoid(x / y));
    } catch (...) {
      is = 0;
      throw std::runtime_error{
          "ForfeitParams: Invalid format for odds. Requires 'x:y'"};
    }
  }

  void set_from_bound(const std::string &value) {
    try {
      is = std::abs(inverse_sigmoid(value));
    } catch (...) {
      is = 0;
      throw std::runtime_error{"ForfeitParams: Could not parse bounds (0, 1)"};
    }
  }

  static double inverse_sigmoid(const double x) {
    return std::log(x) - std::log(1 - x);
  }
};

struct Adjuticate : public AdjuticateParams {
  size_t record;
};

struct PlayerAdjudicate {
  Adjuticate forfeit;
  Adjuticate draw;

  void set_from_args(const auto &args) {
    *this = {};
    if (args.)
  }
};

PKMN::Result check_adjudication(PlayerAdjudicate &p1,
                                const MCTS::Output &p1_output,
                                PlayerAdjudicate &p2,
                                const MCTS::Output &p2_output) {
  // TODO
  return {};
}

} // namespace RuntimePolicy