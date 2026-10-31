// The bookkeeping state as network input (phase 2, not written yet): one
// player's Bookkeeping flattened to STATE_FIELDS int16s. This stub writes
// zeros.

#pragma once

#include <cstdint>
#include <cstring>

#include "bookkeeping.hpp"

namespace ataraxos {

inline constexpr size_t STATE_FIELDS = 1;

inline void encode_state(const Bookkeeping&, int16_t* out) { std::memset(out, 0, STATE_FIELDS * sizeof(int16_t)); }

} // namespace ataraxos
