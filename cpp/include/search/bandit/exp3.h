

#pragma once

#include <search/joint.h>
#include <search/util/softmax.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace Exp3 {

constexpr float neg_inf = -std::numeric_limits<float>::infinity();

#pragma pack(push, 1)
struct Bandit {

  struct Outcome {
    float value;
    float prob;
    uint8_t index;
  };

  struct Params {
    float gamma;
    float one_minus_gamma;
    float alpha;
    float one_minus_alpha;
  };

  std::array<float, 9> gains;
  uint8_t k;

  void init(const auto k) noexcept {
    this->k = k;
    std::fill(gains.begin(), gains.begin() + k, 0);
    std::fill(gains.begin() + k, gains.end(), neg_inf);
  }

  bool is_init() const noexcept { return k; }

  void select(auto &device, const Params &params,
              auto &outcome) const noexcept {
    std::array<float, 9> policy;
    if (k == 1) {
      outcome.index = 0;
      outcome.prob = 1;
    } else {
      const float eta{params.gamma / k};
      const float delta{params.alpha / k};
      softmax(policy, gains, eta);
      std::transform(policy.begin(), policy.end(), policy.begin(),
                     [eta, delta, &params](const float x) {
                       return params.one_minus_alpha * x + delta;
                     });
      outcome.index = std::min(static_cast<uint8_t>(device.sample_pdf(policy)),
                               static_cast<uint8_t>(k - 1));
      outcome.prob = policy[outcome.index];
    }
  }

  void update(const auto &outcome) noexcept {
    constexpr float baseline = 0;
    if ((gains[outcome.index] += (outcome.value - baseline) / outcome.prob) >
        0) {
      const auto max = gains[outcome.index];
      for (auto &v : gains) {
        v -= max;
      }
    }
  }
};
#pragma pack(pop)

using JointBandit = Joint<Bandit>;

// static_assert(sizeof(JointBandit) == 74);

}; // namespace Exp3
