#pragma once

#include <train/battle/compressed-frame.h>

#include <array>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace Py::Battle {

struct Target {
  size_t size;
  py::array_t<uint8_t> k; // num actions
  py::array_t<uint8_t> choice;
  py::array_t<uint32_t> iterations;
  py::array_t<float> empirical_policies;
  py::array_t<float> nash_policies;
  py::array_t<float> empirical_value;
  py::array_t<float> nash_value;
  py::array_t<float> score;

  Target(size_t size) : size{size} {
    std::vector<size_t> shape1{size, 1};
    std::vector<size_t> shape2{size, 2, 1};
    std::vector<size_t> shape9{size, 2, 9};
    k = py::array_t<uint8_t>(shape2);
    choice = py::array_t<uint8_t>(shape2);
    iterations = py::array_t<uint32_t>(shape1);
    empirical_policies = py::array_t<float>(shape9);
    nash_policies = py::array_t<float>(shape9);
    empirical_value = py::array_t<float>(shape1);
    nash_value = py::array_t<float>(shape1);
    score = py::array_t<float>(shape1);
  }

  void clear() {
    std::fill_n(k.mutable_data(), k.size(), uint8_t{});
    std::fill_n(choice.mutable_data(), choice.size(), uint8_t{});
    std::fill_n(iterations.mutable_data(), iterations.size(), uint32_t{});
    std::fill_n(empirical_policies.mutable_data(), empirical_policies.size(),
                float{});
    std::fill_n(nash_policies.mutable_data(), nash_policies.size(), float{});
    std::fill_n(empirical_value.mutable_data(), empirical_value.size(),
                float{});
    std::fill_n(nash_value.mutable_data(), nash_value.size(), float{});
    std::fill_n(score.mutable_data(), score.size(), float{});
  }

  void write(const auto index,
             const Train::Battle::CompressedFrames::Update &update) {
    auto k_ = k.mutable_data() + index * (2 * 1);
    auto choice_ = choice.mutable_data() + index * (2 * 1);
    auto iterations_ = iterations.mutable_data() + index * (1);
    auto empirical_policies_ =
        empirical_policies.mutable_data() + index * (2 * 9);
    auto nash_policies_ = nash_policies.mutable_data() + index * (2 * 9);
    auto empirical_value_ = empirical_value.mutable_data() + index * (1);
    auto nash_value_ = nash_value.mutable_data() + index * (1);
    auto score_ = score.mutable_data() + index * (1);
    update.write_to_tensor(k_, choice_, iterations_, empirical_policies_,
                           nash_policies_, empirical_value_, nash_value_);
  }
};

} // namespace Py::Battle