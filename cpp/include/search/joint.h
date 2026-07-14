#pragma once

#include <utility>

#pragma pack(push, 1)
template <typename Bandit> struct Joint {
  using Params = typename Bandit::Params;
  using Outcome = typename Bandit::Outcome;

  struct JointOutcome {
    Outcome p1;
    Outcome p2;
  };

  Bandit p1;
  Bandit p2;

  void init(const auto m, const auto n) noexcept {
    p1.init(m);
    p2.init(n);
  }

  bool is_init() const noexcept { return p1.is_init(); }

  void select(auto &device, const Params &params,
              JointOutcome &outcome) const noexcept {
    p1.select(device, params, outcome.p1);
    p2.select(device, params, outcome.p2);
  }

  void update(const Params &params, const JointOutcome &outcome) noexcept {
    p1.update(params, outcome.p1);
    p2.update(params, outcome.p2);
  }

  void softmax_logits(const Params &params, const float *p1_priors,
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