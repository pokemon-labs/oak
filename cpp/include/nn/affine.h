#pragma once

#include <algorithm>
#include <cstddef>
#include <istream>

#include <Eigen/Core>

namespace NN {

enum class Activation {
  same = -1,
  none = 0,
  relu = 1,
  clamp = 2,
  relu_scaled = 3,
};

template <int Order = Eigen::RowMajor> class Affine {
public:
  using enum Activation;
  using Vector = Eigen::VectorXf;
  using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Order>;
  using MatrixRowMajor =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  uint32_t in_dim;
  uint32_t out_dim;
  Matrix weights;
  Vector biases;

  bool operator==(const Affine &other) const noexcept {
    return (biases == other.biases) && (weights == other.weights);
  }

  bool read_parameters(std::istream &stream) {
    if (!stream.read(reinterpret_cast<char *>(&in_dim), sizeof(uint32_t))) {
      return false;
    }
    if (!stream.read(reinterpret_cast<char *>(&out_dim), sizeof(uint32_t))) {
      return false;
    }
    weights.resize(out_dim, in_dim);
    biases.resize(out_dim);
    if (!stream.read(reinterpret_cast<char *>(biases.data()),
                     out_dim * sizeof(float))) {
      return false;
    }
    if constexpr (Order == Eigen::RowMajor) {
      if (!stream.read(reinterpret_cast<char *>(weights.data()),
                       out_dim * in_dim * sizeof(float))) {
        return false;
      }
    } else {
      MatrixRowMajor row_major_weights;
      row_major_weights.resize(out_dim, in_dim);
      if (!stream.read(reinterpret_cast<char *>(row_major_weights.data()),
                       out_dim * in_dim * sizeof(float))) {
        return false;
      }
      weights = row_major_weights;
    }

    for (auto i = 0; i < out_dim; ++i) {
      assert(!std::isnan(biases(i)));
      for (auto j = 0; j < in_dim; ++j) {
        assert(!std::isnan(weights(i, j)));
      }
    }
    return true;
  }

  template <Activation act = none, Activation pre = none>
  void propagate(const float *input_data, float *output_data) const {
    constexpr auto activation = (act == same) ? pre : act;
    const auto input = Eigen::Map<const Vector>(input_data, in_dim);
    Eigen::Map<Vector> output(output_data, out_dim);
    output.noalias() = weights * input + biases;
    if constexpr (activation == none) {
      return;
    } else if constexpr (activation == relu) {
      output = output.cwiseMax(0.0f);
    } else if constexpr (activation == clamp) {
      output = output.cwiseMax(0.0f).cwiseMin(1.0f);
    } else if constexpr (activation == relu_scaled) {
      output = output.cwiseMax(0.0f);
      const int chunks = out_dim / 32;
      for (int i = 0; i < chunks; ++i) {
        auto chunk = output.segment(i * 32, 32);
        float chunk_max = chunk.maxCoeff();
        if (chunk_max > 1.0f)
          chunk /= chunk_max;
      }
    }
  }

  template <Activation act = none, Activation pre = none>
  void propagate(const float *input_data, const auto *index_data,
                 float *output_data, uint32_t n) const {
    constexpr auto activation = (act == same) ? pre : act;
    Eigen::Map<Vector> output(output_data, out_dim);
    output = biases;
    for (auto k = 0; k < n; ++k) {
      output.noalias() += weights.col(index_data[k]) * input_data[k];
    }
    if constexpr (activation == none) {
      return;
    } else if constexpr (activation == relu) {
      output = output.cwiseMax(0.0f);
    } else if constexpr (activation == clamp) {
      output = output.cwiseMax(0.0f).cwiseMin(1.0f);
    } else if constexpr (activation == relu_scaled) {
      output = output.cwiseMax(0.0f);
      const int chunks = out_dim / 32;
      for (int i = 0; i < chunks; ++i) {
        auto chunk = output.segment(i * 32, 32);
        float chunk_max = chunk.maxCoeff();
        if (chunk_max > 1.0f)
          chunk /= chunk_max;
      }
    }
  }

  void initialize(auto &device) {
    const float k = 1.0f / std::sqrt(static_cast<float>(in_dim));
    for (auto i = 0; i < out_dim; ++i) {
      biases(i) = device.uniform() * 2 * k - k;
    }
    for (auto i = 0; i < out_dim; ++i) {
      for (auto j = 0; j < in_dim; ++j) {
        weights(i, j) = device.uniform() * 2 * k - k;
      }
    }
  }
};

inline uint64_t combine_hash(uint64_t h1, uint64_t h2) {
  return (h1 ^ (h2 + 0x9E3779B97F4A7C15 + (h1 << 6) + (h1 >> 2))) &
         0xFFFFFFFFFFFFFFFF;
}

} // namespace NN