#pragma once

#include <search/joint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace MCTS {

struct RegretMatching {
  RegretMatching() = default;
  explicit RegretMatching(float exploration)
      : exploration{exploration}, one_minus_exploration{1 - exploration} {}
  float exploration{0};
  float one_minus_exploration{1};

#pragma pack(push, 1)
  struct Stats {
    std::array<float, 9> regrets;
    uint8_t k;

    void init(const auto k) noexcept {
      this->k = k;
      std::fill(regrets.begin(), regrets.end(), 0);
    }

    bool is_init() const noexcept { return k; }

    void select(auto &device, const RegretMatching &bandit,
                auto &outcome) const noexcept {
      if (k == 1) {
        outcome.index = 0;
        outcome.prob = 1;
      } else {
        auto &policy = outcome.policy;
        float sum = 0;
        for (auto i = 0; i < k; ++i) {
          const float x = std::max(regrets[i], 0.0f);
          policy[i] = x;
          sum += x;
        }
        // padding must have zero mass
        std::fill(policy.begin() + k, policy.end(), 0);

        const float uniform = 1.0f / k;
        if (sum > 0) {
          const float inv = 1.0f / sum;
          for (auto i = 0; i < k; ++i) {
            policy[i] *= inv;
          }
        } else {
          std::fill(policy.begin(), policy.begin() + k, uniform);
        }

        if (bandit.exploration > 0) {
          const float delta = bandit.exploration * uniform;
          for (auto i = 0; i < k; ++i) {
            policy[i] = bandit.one_minus_exploration * policy[i] + delta;
          }
        }

        outcome.index =
            std::min(static_cast<uint8_t>(device.sample_pdf(policy)),
                     static_cast<uint8_t>(k - 1));
        outcome.prob = policy[outcome.index];
      }
    }

    void update(const RegretMatching &, const auto &outcome) noexcept {
      assert(outcome.prob > 0);
      const float r = outcome.value;
      const auto a = outcome.index;
      // g_i = -r_a for i != a
      for (auto i = 0; i < k; ++i) {
        regrets[i] -= r;
      }
      // g_a = r_a / p_a - r_a
      regrets[a] += r / outcome.prob;
    }
  };

  struct Outcome {
    float value;
    float prob;
    uint8_t index;
    std::array<float, 9> policy;
  };
  using JointStats = Joint<RegretMatching>;
};
#pragma pack(pop)

} // namespace MCTS