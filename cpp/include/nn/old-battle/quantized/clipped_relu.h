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

// Definition of layer ClippedReLU of NNUE evaluation function

#pragma once

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include <nn/battle/quantized/common.h>

namespace NN::Battle::Quantized {

// Clipped ReLU
template <IndexType InDims> class ClippedReLU {
public:
  // Input/output type
  using InputType = std::int32_t;
  using OutputType = std::uint8_t;

  // Number of input/output dimensions
  static constexpr IndexType InputDimensions = InDims;
  static constexpr IndexType OutputDimensions = InputDimensions;
  static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, 32);

  using OutputBuffer = OutputType[PaddedOutputDimensions];

  // Hash value embedded in the evaluation file
  static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
    std::uint32_t hashValue = 0x538D24C7u;
    hashValue += prevHash;
    return hashValue;
  }

  // Forward propagation
  void propagate(const InputType *input, OutputType *output) const {

    if constexpr (InputDimensions % SimdWidth == 0) {
      constexpr IndexType NumChunks = InputDimensions / SimdWidth;
      const __m256i Offsets = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
      const auto in = reinterpret_cast<const __m256i *>(input);
      const auto out = reinterpret_cast<__m256i *>(output);
      for (IndexType i = 0; i < NumChunks; ++i) {
        const __m256i words0 = _mm256_srli_epi16(
            _mm256_packus_epi32(_mm256_load_si256(&in[i * 4 + 0]),
                                _mm256_load_si256(&in[i * 4 + 1])),
            WeightScaleBits);
        const __m256i words1 = _mm256_srli_epi16(
            _mm256_packus_epi32(_mm256_load_si256(&in[i * 4 + 2]),
                                _mm256_load_si256(&in[i * 4 + 3])),
            WeightScaleBits);
        _mm256_store_si256(&out[i],
                           _mm256_permutevar8x32_epi32(
                               _mm256_packs_epi16(words0, words1), Offsets));
      }
    } else {
      constexpr IndexType NumChunks = InputDimensions / (SimdWidth / 2);
      const auto in = reinterpret_cast<const __m128i *>(input);
      const auto out = reinterpret_cast<__m128i *>(output);
      for (IndexType i = 0; i < NumChunks; ++i) {
        const __m128i words0 =
            _mm_srli_epi16(_mm_packus_epi32(_mm_load_si128(&in[i * 4 + 0]),
                                            _mm_load_si128(&in[i * 4 + 1])),
                           WeightScaleBits);
        const __m128i words1 =
            _mm_srli_epi16(_mm_packus_epi32(_mm_load_si128(&in[i * 4 + 2]),
                                            _mm_load_si128(&in[i * 4 + 3])),
                           WeightScaleBits);
        _mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
      }
    }
    constexpr IndexType Start =
        InputDimensions % SimdWidth == 0
            ? InputDimensions / SimdWidth * SimdWidth
            : InputDimensions / (SimdWidth / 2) * (SimdWidth / 2);

    for (IndexType i = Start; i < InputDimensions; ++i) {
      output[i] = static_cast<OutputType>(
          std::clamp(input[i] >> WeightScaleBits, 0, 127));
    }
  }
};

} // namespace NN::Battle::Quantized