#pragma once

#include <encode/battle/policy.h>
#include <nn/default-hyperparameters.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Py::Battle {

namespace py = pybind11;

namespace {

struct OutputBuffer {
  size_t size;
  size_t pokemon_out_dim;
  size_t active_out_dim;
  size_t side_out_dim;
  py::array_t<float> pokemon;
  py::array_t<float> active_pokemon;
  py::array_t<float> sides;
  py::array_t<float> value;
  py::array_t<float> logit;        // raw
  py::array_t<float> policy_logit; // selected
  py::array_t<float> policy;       // softmax

  // last dim is neg inf, invalid actions map to it
  static constexpr size_t policy_out_dim = Encode::Battle::Policy::n_dim + 1;

  OutputBuffer(size_t size, size_t pod = NN::Battle::Default::pokemon_out_dim,
               size_t aod = NN::Battle::Default::active_out_dim)
      : size{size}, pokemon_out_dim{pod}, active_out_dim{aod},
        side_out_dim{(1 + active_out_dim) + 5 * (1 + pokemon_out_dim)} {
    pokemon =
        py::array_t<float>(std::vector<size_t>{size, 2, 5, pokemon_out_dim});
    active_pokemon =
        py::array_t<float>(std::vector<size_t>{size, 2, 1, active_out_dim});
    sides = py::array_t<float>(std::vector<size_t>{size, 2, 1, side_out_dim});
    value = py::array_t<float>(std::vector<size_t>{size, 1});
    logit = py::array_t<float>(std::vector<size_t>{size, 2, policy_out_dim});
    policy_logit = py::array_t<float>(std::vector<size_t>{size, 2, 9});
    policy = py::array_t<float>(std::vector<size_t>{size, 2, 9});
    clear();
  }

  void clear() {
    std::fill_n(pokemon.mutable_data(), pokemon.size(), 0.0f);
    std::fill_n(active_pokemon.mutable_data(), active_pokemon.size(), 0.0f);
    std::fill_n(sides.mutable_data(), sides.size(), 0.0f);
    std::fill_n(value.mutable_data(), value.size(), 0.0f);
    std::fill_n(logit.mutable_data(), logit.size(), 0.0f);
    std::fill_n(policy_logit.mutable_data(), policy_logit.size(), 0.0f);
    std::fill_n(policy.mutable_data(), policy.size(), 0.0f);
    auto l = logit.mutable_unchecked<3>();
    for (auto s = 0; s < 2; ++s) {
      for (size_t i = 0; i < l.shape(0); ++i) {
        l(i, s, l.shape(2) - 1) = -std::numeric_limits<float>::infinity();
      }
    }
  }
};

} // namespace

} // namespace Py::Battle