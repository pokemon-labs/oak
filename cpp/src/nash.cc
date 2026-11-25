#include <search/nash.h>

#include <lrsnash/solve.hpp>

namespace Nash {

namespace {

LRSNash::Solver &solver() {
  static thread_local LRSNash::Solver s;
  return s;
}

LRSNash::Mode convert(Mode mode) {
  switch (mode) {
  case Mode::Last:
    return LRSNash::Mode::Last;
  case Mode::First:
    return LRSNash::Mode::First;
  case Mode::Any:
    return LRSNash::Mode::Any;
  }
  return LRSNash::Mode::Last;
}

} // namespace

float solve_constant_sum(int rows, int cols, const int *data, int den,
                         float *row_strategy, float *col_strategy, Mode mode) {
  LRSNash::ConstantSumGame g{rows, cols, data, den};
  LRSNash::FloatEquilibrium out{row_strategy, col_strategy, 0.f};
  solver().solve(&g, &out, convert(mode));
  return out.value;
}

float solve_general_sum(int rows, int cols, const int *row_data,
                        const int *col_data, int den, float *row_strategy,
                        float *col_strategy, Mode mode, long missing_label) {
  LRSNash::GeneralSumGame g{rows, cols, den, row_data, col_data};
  LRSNash::FloatEquilibrium out{row_strategy, col_strategy, 0.f};
  solver().solve(&g, &out, convert(mode), missing_label);
  return out.value;
}

} // namespace Nash
