#pragma once

#include <nn/affine.h>

#include <fstream>
#include <vector>

namespace NN {

template <typename... Layers> struct FeedForwardNetwork {
  std::tuple<Layers...> layers;
  std::vector<float> buffer_a;
  std::vector<float> buffer_b;

  static constexpr auto NumLayers = sizeof...(Layers);
  static_assert(NumLayers > 1,
                "FeedForwardNetwork requires more than 1 layer.");

  template <size_t I> auto &layer() { return std::get<I>(layers); }

  bool read_parameters(std::istream &stream) {
    bool ok = true;
    std::apply([&](auto &...l) { ((ok &= l.read_parameters(stream)), ...); },
               layers);
    if (!ok) {
      return false;
    }
    size_t max_out = 0;
    std::apply(
        [&](auto &...l) {
          ((max_out = std::max(max_out, size_t(l.out_dim))), ...);
        },
        layers);
    buffer_a.resize(max_out);
    buffer_b.resize(max_out);
    return true;
  }

  template <Activation First, Activation... Rest>
  void propagate(const float *input, float *output) {
    static_assert((sizeof...(Rest) + 1) == NumLayers);
    layer<0>().template propagate<First>(input, buffer_a.data());
    propagate_impl<1, First, Rest...>(output);
  }

  template <Activation First, Activation... Rest>
  void propagate(const float *input, const auto *index, float *output, auto n) {
    static_assert((sizeof...(Rest) + 1) == NumLayers);
    layer<0>().template propagate<First>(input, index, buffer_a.data(), n);
    propagate_impl<1, First, Rest...>(output);
  }

  template <size_t I, Activation Prev, Activation Curr, Activation... Rest>
  void propagate_impl(float *final) {
    auto *in = (I & 1) ? buffer_a.data() : buffer_b.data();
    auto *out = (I & 1) ? buffer_b.data() : buffer_a.data();
    if constexpr (I == NumLayers - 1) {
      layer<I>().template propagate<Curr, Prev>(in, final);
    } else {
      layer<I>().template propagate<Curr, Prev>(in, out);
      propagate_impl<I + 1, Curr, Rest...>(final);
    }
  }

  template <size_t I, Activation Prev> void propagate_impl(float *final) {
    static_assert(I == NumLayers,
                  "propagate_impl reached terminal with unexpected index");
  }

  void initialize(auto &device) {
    std::apply([&](auto &...l) { (l.initialize(device), ...); }, layers);
  }
};

using EmbeddingNet = FeedForwardNetwork<Affine<Eigen::ColMajor>, Affine<>>;
using EmbeddingNet3 =
    FeedForwardNetwork<Affine<Eigen::ColMajor>, Affine<>, Affine<>>;

using TeamBuildingNet =
    FeedForwardNetwork<Affine<Eigen::ColMajor>, Affine<>, Affine<>>;

// gives the FFNs the 'module' interface of the Battle::Network refactor
// i.e. only one activation param
template <Activation last = Activation::same>
struct MLP2 : public FeedForwardNetwork<Affine<>, Affine<>> {
  template <Activation act> void propagate(const float *input, float *output) {
    propagate<act, std::is_same_v<last, Activation::same> ? act : last>(input,
                                                                        output);
  }
  template <Activation act>
  void propagate(const float *input, const auto *index, float *output, auto n) {
    propagate<act, std::is_same_v<last, Activation::same> ? act : last>(
        input, index, output, n);
  }
};
template <Activation last = Activation::same>
struct MLP3 : public FeedForwardNetwork<Affine<>, Affine<>, Affine<>> {
  template <Activation act> void propagate(const float *input, float *output) {
    propagate<act, act, std::is_same_v<last, Activation::same> ? act : last>(
        input, output);
  }
  template <Activation act>
  void propagate(const float *input, const auto *index, float *output, auto n) {
    propagate<act, act, std::is_same_v<last, Activation::same> ? act : last>(
        input, index, output, n);
  }
};

// i.e.
using Trunk2 = MLP2<>;
using Head2 = MLP2<Activation::none>;

} // namespace NN