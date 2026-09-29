#pragma once

#include <cmath>
#include <numeric>

template <typename T>
inline void softmax_k(T *output, const float *logits, auto k) {
  constexpr bool is_integral = std::is_integral_v<T>;
  static thread_local float out[9];
  auto *o = [&]() {
    if constexpr (is_integral) {
      return out;
    } else {
      return output;
    }
  }();
  float sum = 0;
  for (auto i = 0; i < k; ++i) {
    const float y = std::exp(logits[i]);
    o[i] = y;
    sum += y;
  }
  for (auto i = 0; i < k; ++i) {
    o[i] /= sum;
    if constexpr (is_integral) {
      output[i] = o[i] * std::numeric_limits<T>::max();
    }
  }
}

inline void softmax_9_temp(auto &forecast, const auto &gains, float temp) {
  float sum = 0;
  for (auto i = 0; i < 9; ++i) {
    const float y = std::exp(gains[i] * temp);
    forecast[i] = y;
    sum += y;
  }
  for (auto i = 0; i < 9; ++i) {
    forecast[i] /= sum;
  }
}

inline void softmax_9(auto &forecast, const auto &gains) {
  float sum = 0;
  for (auto i = 0; i < 9; ++i) {
    const float y = std::exp(gains[i]);
    forecast[i] = y;
    sum += y;
  }
  for (auto i = 0; i < 9; ++i) {
    forecast[i] /= sum;
  }
}