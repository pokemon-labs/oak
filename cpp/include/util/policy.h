#pragma once

#include <libpkmn/pkmn.h>
#include <search/mcts.h>
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
      break;
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

template <PKMN::Player player>
inline double get_value(const auto &output, const auto &options) {

  const auto get = [](auto x) {
    return x;
    // if constexpr (player == Player::P1) {
    //   return x;
    // } else {
    //   return 1 - x;
    // }
  };

  double value = 0;

  const auto mode_split = Parse::split(options.mode, '-');

  for (std::string_view word : mode_split) {
    const double w =
        (word.size() > 1) ? std::stod(std::string(word.substr(1))) : 1.0;

    switch (static_cast<Mode>(word[0])) {
    case Mode::prior: {
      value += w * get(output.initial_value);
      break;
    }
    case Mode::empirical: {
      value += w * get(output.empirical_value);
      break;
    }
    case Mode::nash: {
      value += w * get(output.nash_value);
      break;
    }
    case Mode::argmax: {
      std::array<std::pair<double, size_t>, 9> data{};
      for (auto i = 0; i < 9; ++i) {
        for (auto j = 0; j < 9; ++j) {
          if constexpr (player == PKMN::Player::P1) {
            data[i].first += output.value_matrix[i][j];
            data[i].second += output.visit_matrix[i][j];
          } else {
            data[i].first += output.value_matrix[j][i];
            data[i].second += output.visit_matrix[j][i];
          }
        }
      }
      size_t max_visits = 0;
      auto argmax = 0;
      for (auto i = 0; i < 9; ++i) {
        if (data[i].second > max_visits) {
          max_visits = data[i].second;
          argmax = i;
        }
      }
      if (max_visits == 0) {
        throw std::runtime_error{"RuntimePolicy::get_score : empty visit "
                                 "matrix (Probably null MCTS::output.)"};
      }
      value += w * get(data[argmax].first / data[argmax].second);
      break;
    }
    case Mode::beta: {
      throw std::runtime_error{"RuntimePolicy: (b)eta mode disabled for now."};
      break;
    }
    default: {
      throw std::runtime_error{"RuntimePolicy: invalid mode char: " + word[0]};
    }
    }
  }

  return value;
}

inline double inverse_sigmoid(const double x) {
  return std::log(x) - std::log(1.0 - x);
}

struct JointValueMemory {

  std::vector<std::pair<double, double>> data;

  void update(const MCTS::Output &p1_output, const Options &p1_options,
              const MCTS::Output &p2_output, const Options &p2_options) {

    if (p1_output.iterations == 0 || p2_output.iterations == 0) {
      return;
    }
    data.emplace_back(get_value<PKMN::Player::P1>(p1_output, p1_options),
                      get_value<PKMN::Player::P2>(p2_output, p2_options));
  }

  PKMN::Result check_for_consensus(size_t n, double ff) const {
    if (ff == 0.0) {
      return PKMN::Result::None;
    }
    if (data.size() < n) {
      return PKMN::Result::None;
    }
    // positive iff both outputs agree the score is outside ff threshold
    std::vector<int> witnesses{};
    std::transform(data.end() - n, data.end(), std::back_inserter(witnesses),
                   [ff](const auto &x) {
                     int a = inverse_sigmoid(x.first) / ff;
                     int b = inverse_sigmoid(x.second) / ff;
                     return a * b;
                   });
    if (std::all_of(witnesses.begin(), witnesses.end(),
                    [](auto w) { return w > 0; })) {
      if (data.back().first > .5) {
        return PKMN::Result::Win;
      } else {
        return PKMN::Result::Lose;
      }
    }
    return PKMN::Result::None;
  }
};

} // namespace RuntimePolicy