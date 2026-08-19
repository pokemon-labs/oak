#pragma once

#include <encode/battle/policy.h>
#include <nn/affine.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

namespace NN::Battle {

struct MainNet {

  using T = float;

  Affine<> fc0;
  Affine<> fc1;
  Affine<> value_fc2;
  Affine<> value_fc3;
  Affine<> p1_policy_fc2;
  Affine<> p1_policy_fc3;
  Affine<> p2_policy_fc2;
  Affine<> p2_policy_fc3;

  std::vector<float> buffer0;
  std::vector<float> buffer1;
  std::vector<float> value_buffer;
  std::vector<float> p1_policy_buffer;
  std::vector<float> p2_policy_buffer;

  std::tuple<int, int, int, int> shape() const noexcept {
    return {fc0.in_dim, fc0.out_dim, value_fc2.out_dim, p1_policy_fc2.out_dim};
  }

  bool read_parameters(std::istream &stream) {
    const bool ok = fc0.read_parameters(stream) &&
                    fc1.read_parameters(stream) &&
                    value_fc2.read_parameters(stream) &&
                    value_fc3.read_parameters(stream) &&
                    p1_policy_fc2.read_parameters(stream) &&
                    p1_policy_fc3.read_parameters(stream) &&
                    p2_policy_fc2.read_parameters(stream) &&
                    p2_policy_fc3.read_parameters(stream);

    if (!ok) {
      return false;
    }
    buffer0.resize(fc0.out_dim);
    buffer1.resize(fc1.out_dim);
    value_buffer.resize(value_fc2.out_dim);
    p1_policy_buffer.resize(p1_policy_fc2.out_dim);
    p2_policy_buffer.resize(p2_policy_fc2.out_dim);
    return true;
  }

  template <Activation activation> float propagate(const float *input_data) {
    float output;
    fc0.propagate<activation>(input_data, buffer0.data());
    fc1.propagate<activation>(buffer0.data(), buffer1.data());
    value_fc2.propagate<activation>(buffer1.data(), value_buffer.data());
    value_fc3.propagate<>(value_buffer.data(), &output);
    return output;
  }

  template <bool use_value, Activation activation>
  auto propagate(const float *input_data, const auto m, const auto n,
                 const auto *p1_choice_index, const auto *p2_choice_index,
                 float *p1, float *p2)
      -> std::conditional_t<use_value, float, void> {
    float output;
    fc0.propagate<activation>(input_data, buffer0.data());
    fc1.propagate<activation>(buffer0.data(), buffer1.data());
    if constexpr (use_value) {
      value_fc2.propagate<activation>(buffer1.data(), value_buffer.data());
      value_fc3.propagate<>(value_buffer.data(), &output);
    }
    p1_policy_fc2.propagate<activation>(buffer1.data(),
                                        p1_policy_buffer.data());
    p2_policy_fc2.propagate<activation>(buffer1.data(),
                                        p2_policy_buffer.data());

    for (auto i = 0; i < m; ++i) {
      const auto p1_c = p1_choice_index[i];
      assert(p1_c < Encode::Battle::Policy::n_dim);
      const float logit =
          p1_policy_fc3.weights.row(p1_c).dot(Eigen::Map<const Eigen::VectorXf>(
              p1_policy_buffer.data(), p1_policy_fc2.out_dim)) +
          p1_policy_fc3.biases[p1_c];
      assert(!std::isnan(logit));
      p1[i] = logit;
    }

    for (auto i = 0; i < n; ++i) {
      const auto p2_c = p2_choice_index[i];
      assert(p2_c < Encode::Battle::Policy::n_dim);
      const float logit =
          p2_policy_fc3.weights.row(p2_c).dot(Eigen::Map<const Eigen::VectorXf>(
              p2_policy_buffer.data(), p2_policy_fc2.out_dim)) +
          p2_policy_fc3.biases[p2_c];
      assert(!std::isnan(logit));
      p2[i] = logit;
    }
    if constexpr (use_value) {
      return output;
    }
  }
};

// struct MainNetHalf {

//   using T = float;

//   Affine<> fc0;
//   Affine<> fc1;
//   Affine<> value_fc2;
//   Affine<> value_fc3;
//   Affine<> policy_fc2;
//   Affine<> policy_fc3;

//   std::vector<float> buffer0;
//   std::vector<float> buffer1;
//   std::vector<float> value_buffer;
//   std::vector<float> policy_buffer;

//   std::tuple<int, int, int, int> shape() const noexcept {
//     return {fc0.in_dim, fc0.out_dim, value_fc2.out_dim, policy_fc2.out_dim};
//   }

//   bool read_parameters(std::istream &stream) {
//     const bool ok = fc0.read_parameters(stream) &&
//                     fc1.read_parameters(stream) &&
//                     value_fc2.read_parameters(stream) &&
//                     value_fc3.read_parameters(stream) &&
//                     policy_fc2.read_parameters(stream) &&
//                     policy_fc3.read_parameters(stream);

//     if (!ok) {
//       return false;
//     }
//     buffer0.resize(fc0.out_dim);
//     buffer1.resize(fc1.out_dim);
//     value_buffer.resize(value_fc2.out_dim);
//     policy_buffer.resize(policy_fc2.out_dim);
//     return true;
//   }

//   template <Activation activation> float propagate(const float *input_data) {
//     float output;
//     fc0.propagate<activation>(input_data, buffer0.data());
//     fc1.propagate<activation>(buffer0.data(), buffer1.data());
//     value_fc2.propagate<activation>(buffer1.data(), value_buffer.data());
//     value_fc3.propagate<>(value_buffer.data(), &output);
//     return output;
//   }

//   //

//   template <bool use_value, Activation activation>
//   auto propagate(const float *input_data, const auto m, const auto n,
//                  const auto *p1_choice_index, const auto *p2_choice_index,
//                  float *p1, float *p2)
//       -> std::conditional_t<use_value, float, void> {
//     float output;
//     fc0.propagate<activation>(input_data, buffer0.data());
//     fc0.propagate<activation>(input_data + fc0.in_dim,
//                               buffer0.data() + fc0.out_dim);
//     fc1.propagate<activation>(buffer0.data(), buffer1.data());
//     fc1.propagate<activation>(buffer0.data() + fc1.in_dim,
//                               buffer1.data() + fc1.out_dim);
//     if constexpr (use_value) {
//       value_fc2.propagate<activation>(buffer1.data(), value_buffer.data());
//       value_fc3.propagate<>(value_buffer.data(), &output);
//     }
//     p1_policy_fc2.propagate<activation>(buffer1.data(),
//                                         p1_policy_buffer.data());
//     p2_policy_fc2.propagate<activation>(buffer1.data(),
//                                         p2_policy_buffer.data());

//     for (auto i = 0; i < m; ++i) {
//       const auto p1_c = p1_choice_index[i];
//       assert(p1_c < Encode::Battle::Policy::n_dim);
//       const float logit =
//           p1_policy_fc3.weights.row(p1_c).dot(Eigen::Map<const
//           Eigen::VectorXf>(
//               p1_policy_buffer.data(), p1_policy_fc2.out_dim)) +
//           p1_policy_fc3.biases[p1_c];
//       assert(!std::isnan(logit));
//       p1[i] = logit;
//     }

//     for (auto i = 0; i < n; ++i) {
//       const auto p2_c = p2_choice_index[i];
//       assert(p2_c < Encode::Battle::Policy::n_dim);
//       const float logit =
//           p2_policy_fc3.weights.row(p2_c).dot(Eigen::Map<const
//           Eigen::VectorXf>(
//               p2_policy_buffer.data(), p2_policy_fc2.out_dim)) +
//           p2_policy_fc3.biases[p2_c];
//       assert(!std::isnan(logit));
//       p2[i] = logit;
//     }
//     if constexpr (use_value) {
//       return output;
//     }
//   }
// };

} // namespace NN::Battle