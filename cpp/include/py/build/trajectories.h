#pragma once

#include <encode/build/compressed-trajectory.h>
#include <encode/build/tensorizer.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Py::Build {

using PKMN::Data::Move;
using PKMN::Data::Species;

// using Tensorizer = Encode::Build::Tensorizer<>;
using Encode::Build::Tensorizer;

// These two structs store what is *missing* so they can quickly write the
// actions masks
template <typename F = Format::OU> struct SetHelper {
  Species species;
  std::array<Move, F::max_move_pool_size> move_pool;
  uint32_t n_moves;
  uint32_t move_pool_size;

  SetHelper() = default;
  SetHelper(const auto species)
      : species{static_cast<Species>(species)}, n_moves{},
        move_pool_size{F::move_pool_size(species)},
        move_pool{F::move_pool(species)} {}

  auto begin() { return move_pool.begin(); }
  const auto begin() const { return move_pool.begin(); }
  auto end() { return move_pool.begin() + move_pool_size; }
  const auto end() const { return move_pool.begin() + move_pool_size; }

  void add_move(const auto m) {
    const auto move_pool_end =
        std::remove(begin(), end(), static_cast<Move>(m));
    if (move_pool_end != end()) {
      --move_pool_size;
      ++n_moves;
    }
  }

  bool complete() const { return (n_moves >= 4) || (move_pool_size == 0); }
};

template <typename F = Format::OU> struct TeamHelper {
  TeamHelper() : sets{}, size{} {
    available_species = {F::legal_species.begin(), F::legal_species.end()};
  }

  std::array<SetHelper<F>, 6> sets;
  int size;
  std::vector<Species> available_species;

  auto begin() { return sets.begin(); }
  const auto begin() const { return sets.begin(); }
  auto end() { return sets.begin() + size; }
  const auto end() const { return sets.begin() + size; }

  void apply_action(const auto action) {
    const auto [s, m] = Tensorizer<F>::species_move_list(action);
    const auto species = static_cast<Species>(s);
    const auto move = static_cast<Move>(m);
    if (move == Move::None) {
      sets[size++] = SetHelper{species};
      std::erase(available_species, species);
    } else {
      auto it = std::find_if(begin(), end(), [species](const auto &set) {
        return set.species == species;
      });
      assert(it != end());
      auto &set = *it;
      set.add_move(move);
    }
  }

  auto write_moves(auto *mask) const {
    for (const auto &set : (*this)) {
      if (!set.complete()) {
        for (const auto move : set) {
          *mask++ = Tensorizer<F>::species_move_table(set.species, move);
        }
      }
    }
    return mask;
  }

  auto write_species(auto *mask) {
    for (const auto species : available_species) {
      *mask++ = Tensorizer<F>::species_move_table(species, Move::None);
    }
    return mask;
  }

  auto write_swaps(auto *mask) const {
    for (const auto &set : (*this)) {
      *mask++ = Tensorizer<F>::species_move_table(set.species, Move::None);
    }
    return mask;
  }
};

namespace {

struct Trajectories {
  size_t size;
  py::array_t<int64_t> action;
  py::array_t<int64_t> mask;
  py::array_t<float> policy;
  py::array_t<float> value;
  py::array_t<float> score;
  py::array_t<int64_t> start;
  py::array_t<int64_t> end;

  Trajectories(size_t size) : size{size} {
    action = py::array_t<int64_t>(std::vector<size_t>{size, 31, 1});
    mask = py::array_t<int64_t>(
        std::vector<size_t>{size, 31, Tensorizer<>::max_actions});
    start = py::array_t<int64_t>(std::vector<size_t>{size, 1});
    end = py::array_t<int64_t>(std::vector<size_t>{size, 1});
    policy = py::array_t<float>(std::vector<size_t>{size, 31, 1});
    value = py::array_t<float>(std::vector<size_t>{size, 1});
    score = py::array_t<float>(std::vector<size_t>{size, 1});
  }

  void clear() {
    std::fill_n(action.mutable_data(), action.size(), int64_t{});
    std::fill_n(mask.mutable_data(), mask.size(), int64_t{});
    std::fill_n(start.mutable_data(), start.size(), int64_t{});
    std::fill_n(end.mutable_data(), end.size(), int64_t{});
    std::fill_n(policy.mutable_data(), policy.size(), float{});
    std::fill_n(value.mutable_data(), value.size(), float{});
    std::fill_n(score.mutable_data(), score.size(), float{});
  }

  auto view(const size_t index) {
    return std::make_tuple(
        action.mutable_data() + index * (31),
        mask.mutable_data() + index * (31 * Tensorizer<>::max_actions),
        policy.mutable_data() + index * (31),
        value.mutable_data() + index * (1), score.mutable_data() + index * (1),
        start.mutable_data() + index * (1), end.mutable_data() + index * (1));
  }

  void write(const size_t index,
             const Encode::Build::CompressedTrajectory<> &traj) {
    constexpr float den = std::numeric_limits<uint16_t>::max();

    auto [action_, mask_, policy_, value_, score_, start_, end_] = view(index);

    TeamHelper<> helper{};

    // get bounds for mask logic
    auto start = 0;
    auto full = 0; // turn off species picks
    auto swap = 0;
    auto end = 0;
    // find start
    for (auto i = 0; i < 31; ++i) {
      const auto update = traj.updates[i];
      if (update.probability != 0) {
        start = i;
        break;
      }
    }
    // notice the reversed for loop. find end, swap, full
    for (auto i = 30; i >= 0; --i) {
      const auto update = traj.updates[i];
      if (update.probability != 0) {
        const auto [s, m] = Tensorizer<>::species_move_list(update.action);
        if (end == 0) {
          // + 1 because we use these like '< end' later
          end = i + 1;
          // is last move a species pick? (e.g. team size > 1)
          // must be a swap
          swap = end - (m == Move::None);
        } else {
          // need to find the last actual addition
          // the last species add is always before the swap/end
          // because movesets must be maximal
          if (m == Move::None) {
            full = i + 1;
            break;
          }
        }
      }
    }

    for (auto i = 0; i < 31; ++i) {
      const auto &update = traj.updates[i];
      auto mask_begin = mask_;
      auto mask_end = mask_ + Tensorizer<>::max_actions;
      if (i < start) {
        *action_++ = -1;
        *policy_++ = 0;
        std::fill(mask_begin, mask_end, -1);
        helper.apply_action(update.action);
      } else {
        // started
        if (i < end) {
          *action_++ = update.action;
          *policy_++ = update.probability / den;
          if (i < swap) {
            if (i < full) {
              mask_ = helper.write_species(mask_);
            }
            mask_ = helper.write_moves(mask_);
          } else {
            mask_ = helper.write_swaps(mask_);
          }
          std::fill(mask_, mask_end, -1);
          auto chosen = std::find(mask_begin, mask_, update.action);
          std::swap(*chosen, *mask_begin);
          helper.apply_action(update.action);
        } else {
          // ended
          *action_++ = -1;
          *policy_++ = 0;
          std::fill(mask_, mask_end, -1);
        }
      }
      mask_ = mask_end;
    } // end update loop

    *start_++ = start;
    *end_++ = end;
    *value_++ = traj.header.value / den;
    if (traj.header.score == std::numeric_limits<uint16_t>::max()) {
      *score_++ = -1;
    } else {
      *score_++ = traj.header.score / 2.0;
    }
  }
};

} // namespace

} // namespace Py::Build