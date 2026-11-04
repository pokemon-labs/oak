

#pragma once

#include <search/joint.h>
#include <search/util/softmax.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace MCTS {

struct Prior {
  Prior() = default;
  Prior(float temp, float min) : temp{temp}, min{min} {}
  float temp;
  float min;

#pragma pack(push, 1)
  struct Stats {
    std::array<float, 9> policy;
    uint8_t k;

    void softmax_logits(const Prior &bandit, const float *logits) noexcept {
      // init softmax logits, apply min floor, renormalize
    }

    void init(const auto k) noexcept {
      this->k = k;
      std::fill(policy.begin(), policy.begin() + k, 0);
      std::fill(policy.begin() + k, policy.end(),
                -std::numeric_limits<float>::infinity());
    }

    bool is_init() const noexcept { return k; }

    void select(auto &device, const Prior &bandit,
                auto &outcome) const noexcept {
      if (k == 1) {
        outcome.index = 0;
      } else {
        // just sample the policy
      }
    }

    void update(const Prior &bandit, const auto &outcome) noexcept {}
  };

  struct Outcome {
    float value;
    uint8_t index;
  };
  using JointStats = Joint<Prior>;
  // static_assert(sizeof(JointStats) == 74);
};
#pragma pack(pop)

} // namespace MCTS