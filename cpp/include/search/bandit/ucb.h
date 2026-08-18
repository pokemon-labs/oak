

#pragma once

#include <search/joint.h>
#include <search/util/int.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace MCTS {

#pragma pack(push, 1)
struct UCB {

  float c;

  struct Outcome {
    float value;
    uint8_t index;
  };

  struct Stats {
    std::array<float, 9> scores;
    std::array<uint32_t, 9> visits;
    uint8_t k;

    void init(const auto k) noexcept {
      this->k = k;
      std::fill(scores.begin(), scores.begin() + k, 0.5);
      std::fill(visits.begin(), visits.begin() + k, 1);
    }

    bool is_init() const noexcept { return k; }

    void update(const auto &outcome) noexcept {
      assert(outcome.value >= 0);
      scores[outcome.index] += outcome.value;
      ++visits[outcome.index];
    }

    void select(auto &device, const UCB &bandit, auto &outcome) const noexcept {
      if (k == 1) {
        outcome.index = 0;
      } else {
        uint64_t N = 0;
        for (auto i = 0; i < k; ++i) {
          N += visits[i];
        }
        float sqrtN = std::sqrt(N);
        float max = 0;
        for (auto i = 0; i < k; ++i) {
          float e = bandit.c * sqrtN / k;
          float a = (e + scores[i]) / visits[i];
          if (a > max) {
            max = a;
            outcome.index = i;
          }
        }
      }
    }
  };

  using JointStats = Joint<UCB>;
  // static_assert(sizeof(JointStats) == 146);
};
#pragma pack(pop)

} // namespace MCTS
