#pragma once

// Exact Nash solves for oak, backed by the header-only lrsnash. Only
// src/nash.cc includes lrsnash itself, so the solver's templates are compiled
// once instead of in every translation unit that includes mcts.h. Each thread
// keeps one reused solver (int64 -> int128 -> GMP, exact in every type).

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace Nash {

// Mirrors LRSNash::Mode.
//   Last  - legacy lrsnash result exactly (the full enumeration).
//   First - first equilibrium in lrsnash's order (early exit).
//   Any   - constant-sum: one LP; general-sum: Lemke-Howson. Much faster, but
//           may pick a different equilibrium than Last in degenerate games.
enum class Mode : uint8_t { Last, First, Any };

// P1 payoff data (row-major rows x cols), P2 payoff den - data. Writes rows
// and cols floats; returns P1's payoff / den.
float solve_constant_sum(int rows, int cols, const int *data, int den,
                         float *row_strategy, float *col_strategy,
                         Mode mode = Mode::Last);

// P1 payoff row_data, P2 payoff col_data. missing_label in [0, rows + cols)
// is used by Mode::Any (Lemke-Howson) only. Returns P1's payoff / den.
float solve_general_sum(int rows, int cols, const int *row_data,
                        const int *col_data, int den, float *row_strategy,
                        float *col_strategy, Mode mode = Mode::Last,
                        long missing_label = 0);

// How MatrixUCB turns its two optimistic matrices U1, U2 into a joint policy.
// Chosen at runtime so methods can be compared without a rebuild.
//   Legacy - one general-sum solve of (U1, U2), Mode::Last. Old behaviour.
//   LH     - the same general-sum game by Lemke-Howson, with a uniformly
//            random missing label each solve (a fixed label biases selection).
//   TwoLP  - "Matrix Games with Bandit Feedback": each player solves max-min
//            on its own optimistic matrix, as two constant-sum LPs.
enum class MatrixUCBSolver : uint8_t { Legacy, LH, TwoLP };

inline MatrixUCBSolver parse_matrix_ucb_solver(std::string_view s) {
  if (s == "legacy") {
    return MatrixUCBSolver::Legacy;
  } else if (s == "lh") {
    return MatrixUCBSolver::LH;
  } else if (s == "two_lp") {
    return MatrixUCBSolver::TwoLP;
  }
  throw std::runtime_error{"matrix-ucb: SOLVER must be legacy/lh/two_lp, got " +
                           std::string{s}};
}

inline const char *to_string(MatrixUCBSolver s) {
  switch (s) {
  case MatrixUCBSolver::Legacy:
    return "legacy";
  case MatrixUCBSolver::LH:
    return "lh";
  case MatrixUCBSolver::TwoLP:
    return "two_lp";
  }
  return "?";
}

inline Mode parse_mode(std::string_view s) {
  if (s == "last") {
    return Mode::Last;
  } else if (s == "first") {
    return Mode::First;
  } else if (s == "any") {
    return Mode::Any;
  }
  throw std::runtime_error{"nash mode must be last/first/any, got " +
                           std::string{s}};
}

} // namespace Nash
