#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <memory>
#include <type_traits>

#include <nn/affine.h>
#include <nn/battle/quantized/affine.h>
#include <nn/battle/quantized/clipped_relu.h>
#include <nn/battle/quantized/common.h>

namespace NN::Battle::Quantized {

template <int In, int Hidden, int ValueHidden, int PolicyHidden>
struct MainNet {
  static constexpr Activation activation{Activation::clamp};
  using T = uint8_t;
  static constexpr int PolicyOut = 320;

  AffineTransform<In, Hidden> fc0;
  ClippedReLU<Hidden> ac0;
  AffineTransform<Hidden, Hidden> fc1;
  ClippedReLU<Hidden> ac1;
  // value head
  AffineTransform<Hidden, ValueHidden> value_fc2;
  ClippedReLU<ValueHidden> value_ac2;
  AffineTransform<ValueHidden, 1> value_fc3;
  // policy head
  AffineTransform<Hidden, PolicyHidden> p1_policy_fc2;
  ClippedReLU<PolicyHidden> p1_policy_ac2;
  AffineTransform<PolicyHidden, PolicyOut> p1_policy_fc3;
  AffineTransform<Hidden, PolicyHidden> p2_policy_fc2;
  ClippedReLU<PolicyHidden> p2_policy_ac2;
  AffineTransform<PolicyHidden, PolicyOut> p2_policy_fc3;

  std::tuple<int, int, int, int> shape() const noexcept {
    return {In, Hidden, ValueHidden, PolicyHidden};
  }

  void try_copy_parameters(const auto &m) {
    fc0.try_copy_parameters(m.fc0);
    fc1.try_copy_parameters(m.fc1);
    value_fc2.try_copy_parameters(m.value_fc2);
    value_fc3.try_copy_parameters(m.value_fc3);
    p1_policy_fc2.try_copy_parameters(m.p1_policy_fc2);
    p2_policy_fc2.try_copy_parameters(m.p2_policy_fc2);
    auto p1_fc3 = m.p1_policy_fc3;
    auto p2_fc3 = m.p2_policy_fc3;
    const auto resize = [](auto &layer, auto dim) {
      auto old_dim = layer.out_dim;
      layer.weights.conservativeResize(dim, layer.in_dim);
      layer.biases.conservativeResize(dim);
      if (dim > old_dim) {
        layer.weights.bottomRows(dim - old_dim).setZero();
        layer.biases.tail(dim - old_dim).setZero();
      }
      layer.out_dim = dim;
    };
    resize(p1_fc3, 320);
    resize(p2_fc3, 320);
    p1_policy_fc3.try_copy_parameters(p1_fc3);
    p2_policy_fc3.try_copy_parameters(p2_fc3);
  }

  struct alignas(CacheLineSize) ValueBuffer {
    alignas(CacheLineSize) typename decltype(fc0)::OutputBuffer fc0_out;
    alignas(CacheLineSize) typename decltype(ac0)::OutputBuffer ac0_out;
    alignas(CacheLineSize) typename decltype(fc1)::OutputBuffer fc1_out;
    alignas(CacheLineSize) typename decltype(ac1)::OutputBuffer ac1_out;
    alignas(CacheLineSize)
        typename decltype(value_fc2)::OutputBuffer value_fc2_out;
    alignas(CacheLineSize)
        typename decltype(value_ac2)::OutputBuffer value_ac2_out;
    alignas(CacheLineSize)
        typename decltype(value_fc3)::OutputBuffer value_fc3_out;
    ValueBuffer() { std::memset(this, 0, sizeof(*this)); }
  };

  struct alignas(CacheLineSize) ValuePolicyBuffer : ValueBuffer {
    alignas(CacheLineSize)
        typename decltype(p1_policy_fc2)::OutputBuffer p1_policy_fc2_out;
    alignas(CacheLineSize)
        typename decltype(p1_policy_ac2)::OutputBuffer p1_policy_ac2_out;
    alignas(CacheLineSize)
        typename decltype(p1_policy_fc3)::OutputBuffer p1_policy_fc3_out;
    alignas(CacheLineSize)
        typename decltype(p2_policy_fc2)::OutputBuffer p2_policy_fc2_out;
    alignas(CacheLineSize)
        typename decltype(p2_policy_ac2)::OutputBuffer p2_policy_ac2_out;
    alignas(CacheLineSize)
        typename decltype(p2_policy_fc3)::OutputBuffer p2_policy_fc3_out;
    ValuePolicyBuffer() { std::memset(this, 0, sizeof(*this)); }
  };

  template <Activation activation>
  float propagate(const uint8_t *input_data) const {
    static_assert(activation == Activation::clamp);
    alignas(CacheLineSize) static thread_local ValueBuffer buffer;
    fc0.propagate(input_data, buffer.fc0_out);
    ac0.propagate(buffer.fc0_out, buffer.ac0_out);
    fc1.propagate(buffer.ac0_out, buffer.fc1_out);
    ac1.propagate(buffer.fc1_out, buffer.ac1_out);
    value_fc2.propagate(buffer.ac1_out, buffer.value_fc2_out);
    value_ac2.propagate(buffer.value_fc2_out, buffer.value_ac2_out);
    value_fc3.propagate(buffer.value_ac2_out, buffer.value_fc3_out);
    const float value = buffer.value_fc3_out[0] / float(127 * (1 << 6));
    assert(!std::isnan(value));
    return value;
  }

  template <bool use_value, Activation activation>
  auto propagate(const uint8_t *input_data, const int m, const int n,
                 const auto *p1_choice_index, const auto *p2_choice_index,
                 float *p1, float *p2)
      -> std::conditional_t<use_value, float, void> const {
    static_assert(activation == Activation::clamp);
    constexpr float conversion = 127 * (1 << 6);
    alignas(CacheLineSize) static thread_local ValuePolicyBuffer buffer;
    fc0.propagate(input_data, buffer.fc0_out);
    ac0.propagate(buffer.fc0_out, buffer.ac0_out);
    fc1.propagate(buffer.ac0_out, buffer.fc1_out);
    ac1.propagate(buffer.fc1_out, buffer.ac1_out);
    if constexpr (use_value) {
      value_fc2.propagate(buffer.ac1_out, buffer.value_fc2_out);
      value_ac2.propagate(buffer.value_fc2_out, buffer.value_ac2_out);
      value_fc3.propagate(buffer.value_ac2_out, buffer.value_fc3_out);
    }
    p1_policy_fc2.propagate(buffer.ac1_out, buffer.p1_policy_fc2_out);
    p1_policy_ac2.propagate(buffer.p1_policy_fc2_out, buffer.p1_policy_ac2_out);
    p2_policy_fc2.propagate(buffer.ac1_out, buffer.p2_policy_fc2_out);
    p2_policy_ac2.propagate(buffer.p2_policy_fc2_out, buffer.p2_policy_ac2_out);
    for (int i = 0; i < m; ++i) {
      p1[i] = p1_policy_fc3.propagate_single(buffer.p1_policy_ac2_out,
                                             p1_choice_index[i]) /
              conversion;
    }
    for (int i = 0; i < n; ++i) {
      p2[i] = p2_policy_fc3.propagate_single(buffer.p2_policy_ac2_out,
                                             p2_choice_index[i]) /
              conversion;
    }
    if constexpr (use_value) {
      const float value = buffer.value_fc3_out[0] / conversion;
      assert(!std::isnan(value));
      return value;
    }
  }
};

} // namespace NN::Battle::Quantized