

#pragma once

#include <search/joint.h>
#include <search/util/softmax.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace MCTS {

struct PExp3 {
  PExp3() = default;
  PExp3(float lr, float exploration, float temp)
      : lr{lr}, exploration{exploration},
        one_minus_exploration{1 - exploration}, temp{temp} {}
  float lr;
  float exploration;
  float one_minus_exploration;
  float temp;

#pragma pack(push, 1)
  struct Stats {
    std::array<float, 9> gains;
    uint8_t k;

    void softmax_logits(const PExp3 &bandit, const float *logits) noexcept {
      std::transform(logits, logits + k, this->gains.data(),
                     [&bandit](const auto x) { return bandit.temp * x; });
      // std::copy() TODO
    }

    void init(const auto k) noexcept {
      this->k = k;
      // TODO remove this we should never init without softmax
      std::fill(gains.begin(), gains.begin() + k, 0);
      std::fill(gains.begin() + k, gains.end(),
                -std::numeric_limits<float>::infinity());
    }

    bool is_init() const noexcept { return k; }

    void select(auto &device, const PExp3 &bandit,
                auto &outcome) const noexcept {
      std::array<float, 9> policy;
      if (k == 1) {
        outcome.index = 0;
        outcome.prob = 1;
      } else {
        const float exploration{bandit.exploration / k};
        softmax_9(policy, gains);
        std::transform(policy.begin(), policy.end(), policy.begin(),
                       [exploration, &bandit](const float x) {
                         return bandit.one_minus_exploration * x + exploration;
                       });
        outcome.index =
            std::min(static_cast<uint8_t>(device.sample_pdf(policy)),
                     static_cast<uint8_t>(k - 1));
        outcome.prob = policy[outcome.index];
      }
    }

    void update(const PExp3 &bandit, const auto &outcome) noexcept {
      constexpr float baseline = 0;
      const float eta{bandit.lr / k};
      if ((gains[outcome.index] +=
           eta * (outcome.value - baseline) / outcome.prob) > 0) {
        const auto max = gains[outcome.index];
        for (auto &v : gains) {
          v -= max;
        }
      }
    }
  };

  struct Outcome {
    float value;
    float prob;
    uint8_t index;
  };
  using JointStats = Joint<PExp3>;
  // static_assert(sizeof(JointStats) == 74);
};
#pragma pack(pop)

} // namespace MCTS