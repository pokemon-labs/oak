// Bookkeeping: the part of the state that one player's view of the log
// determines (phase 2, not written yet). This stub keeps the env's `track`
// path compiling; with it the state input is all zeros.

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "obs.hpp"

namespace ataraxos {

struct Bookkeeping {
  uint8_t player = 0;
  std::array<uint8_t, 2> teamsize{};

  Bookkeeping() = default;
  Bookkeeping(uint8_t p, std::array<uint8_t, 2> sizes) : player(p), teamsize(sizes) {}

  void update(std::span<const Obs>, std::span<const std::vector<Obs>>) {}
};

} // namespace ataraxos
