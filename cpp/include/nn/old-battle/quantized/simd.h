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

#pragma once

#include <immintrin.h>

namespace NN::Battle::Quantized::Simd {

[[maybe_unused]] inline static int m256_hadd(__m256i sum, int bias) {
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum),
                                 _mm256_extracti128_si256(sum, 1));
  sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
  sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
  return _mm_cvtsi128_si32(sum128) + bias;
}

[[maybe_unused]] inline static void
m256_add_dpbusd_epi32(__m256i &acc, __m256i a, __m256i b) {

  __m256i product0 = _mm256_maddubs_epi16(a, b);
  product0 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
  acc = _mm256_add_epi32(acc, product0);
}
} // namespace NN::Battle::Quantized::Simd