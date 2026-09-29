#pragma once

namespace MCTS {
template <typename strategy_type = uint16_t> struct MatrixUCBStats {
  struct Side {
    uint8_t k;
    std::array<strategy_type, 9> policy;
    uint8_t sample(auto &device) const noexcept {
      if constexpr (std::is_integral_v<strategy_type>) {
        auto n = device.random_int(
            static_cast<size_t>(std::numeric_limits<strategy_type>::max()) + 1);
        for (auto i = 0; i < policy.size(); ++i) {
          if (policy[i] >= n) {
            return i;
          }
          n -= policy[i];
        }
        return 0;
      } else {
        return device.sample_pdf(policy);
      }
    }
  };
  struct Entry {
    size_t visits;
    double total_value;
  };

  Side p1;
  Side p2;
  std::array<std::array<Entry, 9>, 9> matrix;
  float ucb_weight;
  size_t total_visits;

  void init(const auto m, const auto n) noexcept {
    ucb_weight = std::log(2 * m * n);
    total_visits = 0;
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        matrix[i][j] = {};
      }
    }
    constexpr auto uniform = std::numeric_limits<strategy_type>::max();
    p1.k = m;
    p2.k = n;
    std::fill_n(p1.policy.begin(), m, uniform / m);
    std::fill_n(p2.policy.begin(), n, uniform / n);
  }

  bool is_init() const { return p1.k; }

  void softmax_logits(const auto &params, const float *p1_logits,
                      const float *p2_logits) {
    softmax(p1.policy.data(), p1_logits, p1.k);
    softmax(p2.policy.data(), p2_logits, p2.k);
  }

  void solve_ucb_matrix(auto &device, auto &params) noexcept {
    std::array<int, 9 * 9> p1_ucb_matrix;
    std::array<int, 9 * 9> p2_ucb_matrix;
    std::array<float, 9 + 2> p1_nash;
    std::array<float, 9 + 2> p2_nash;
    const float log_T = std::log(total_visits + 1);
    const float exp = params.c * std::sqrt(2 * (2 * log_T + ucb_weight));
    for (auto i = 0; i < p1.k; ++i) {
      for (auto j = 0; j < p2.k; ++j) {
        const auto &entry = matrix[i][j];
        float p1_entry = 0.5;
        float p2_entry = 0.5;
        uint32_t visits = 1;
        if (entry.visits > 0) {
          p1_entry = entry.total_value / entry.visits;
          p2_entry = 1 - p1_entry;
          visits = entry.visits;
        }
        const float inv_exploration_den = 1.0 / std::sqrt(visits);
        const float exploration = exp * inv_exploration_den;
        p1_entry += exploration;
        p2_entry += exploration;
        p1_ucb_matrix[i * p2.k + j] = p1_entry * params.discretize_factor;
        p2_ucb_matrix[i * p2.k + j] = p2_entry * params.discretize_factor;
      }
    }

    LRSNash::Input solve_input{static_cast<int>(p1.k), static_cast<int>(p2.k),
                               static_cast<int>(params.discretize_factor),
                               p1_ucb_matrix.data(), p2_ucb_matrix.data()};
    LRSNash::FloatOneSumOutput solve_output{p1_nash.data(), p2_nash.data(), 0};
    LRSNash::solve_full(&solve_input, &solve_output);

    if constexpr (std::is_integral_v<strategy_type>) {
      for (auto i = 0; i < p1.k; ++i) {
        p1.policy[i] = std::numeric_limits<strategy_type>::max() * p1_nash[i];
      }
      for (auto j = 0; j < p2.k; ++j) {
        p2.policy[j] = std::numeric_limits<strategy_type>::max() * p2_nash[j];
      }
    } else {
      std::copy_n(p1_nash, p1.policy, p1.k);
      std::copy_n(p2_nash, p2.policy, p1.k);
    }
  }

  void select(auto &device, const auto &params, auto &outcome) {
    if ((total_visits + 1) % params.interval == 0) {
      solve_ucb_matrix(device, params);
    }
    outcome.first.index = p1.sample(device);
    outcome.second.index = p2.sample(device);
    // do this here so grow=0 will never proc.
    ++total_visits;
  }

  void update(const auto &outcome) {
    auto &entry = matrix[outcome.first.index][outcome.second.index];
    ++entry.visits;
    entry.total_value += outcome.first.value;
  }
};
} // namespace MCTS