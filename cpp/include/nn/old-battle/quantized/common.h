/*
  Quantized, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Quantized developers (see AUTHORS file)

  Quantized is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Quantized is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Constants used in NNUE evaluation function

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include <immintrin.h>

namespace NN::Battle::Quantized {

// Constant used in evaluation value calculation
constexpr int OutputScale = 16;
constexpr int WeightScaleBits = 6;

// Size of cache line (in bytes)
constexpr std::size_t CacheLineSize = 64;

// SIMD width (in bytes)
constexpr std::size_t SimdWidth = 32;

constexpr std::size_t MaxSimdWidth = 32;

// Type of input feature after conversion
using IndexType = std::uint32_t;

// Round n up to be a multiple of base
template <typename IntType>
constexpr IntType ceil_to_multiple(IntType n, IntType base) {
  return (n + base - 1) / base * base;
}

} // namespace NN::Battle::Quantized
