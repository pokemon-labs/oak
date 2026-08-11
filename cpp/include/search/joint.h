#pragma once

#include <utility>

#pragma pack(push, 1)
template <typename Bandit> struct Joint {
  using Stats = typename Bandit::Stats;
  using Outcome = typename Bandit::Outcome;
  using JointOutcome = std::pair<Outcome, Outcome>;
  Stats p1;
  Stats p2;

  void init(const auto m, const auto n) noexcept {
    p1.init(m);
    p2.init(n);
  }

  bool is_init() const noexcept { return p1.is_init(); }

  void select(auto &device, const Bandit &params,
              JointOutcome &outcome) const noexcept {
    p1.select(device, params, outcome.first);
    p2.select(device, params, outcome.second);
  }

  void update(const JointOutcome &outcome) noexcept {
    p1.update(outcome.first);
    p2.update(outcome.second);
  }

  void softmax_logits(const Bandit &params, const float *p1_priors,
                      const float *p2_priors) noexcept
    requires requires(const float *ptr) {
      std::declval<Bandit>().softmax_logits(params, ptr);
    }
  {
    p1.softmax_logits(params, p1_priors);
    p2.softmax_logits(params, p2_priors);
  }

  void print_priors() const
    requires requires { std::declval<Bandit>().print_priors(); }
  {
    p1.print_priors();
    p2.print_priors();
  }
};
#pragma pack(pop)