#pragma once

#include <libpkmn/layout.h>
#include <py/battle/target.h>
#include <train/battle/compressed-frame.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Py::Battle {

namespace py = pybind11;
using namespace PKMN::Layout;

struct Frames : public Target {

  py::array_t<uint8_t> battle;
  py::array_t<uint8_t> durations;
  py::array_t<uint8_t> result;
  py::array_t<uint8_t> choices;

  Frames(size_t size) : Target{size} {
    battle = py::array_t<uint8_t>(std::vector<size_t>{size, Sizes::Battle});
    durations =
        py::array_t<uint8_t>(std::vector<size_t>{size, Sizes::Durations});
    result = py::array_t<uint8_t>(std::vector<size_t>{size, 1});
    choices = py::array_t<uint8_t>(std::vector<size_t>{size, 2, 9});
    clear();
  }

  void clear() {
    Target::clear();
    std::fill_n(battle.mutable_data(), battle.size(), uint8_t{});
    std::fill_n(durations.mutable_data(), durations.size(), uint8_t{});
    std::fill_n(result.mutable_data(), result.size(), uint8_t{});
    std::fill_n(choices.mutable_data(), choices.size(), uint8_t{});
  }

  void write(const auto index, const pkmn_gen1_battle &b,
             const pkmn_gen1_chance_durations &d, pkmn_result r,
             const Train::Battle::CompressedFrames::Update &update,
             float terminal) {
    Target::write(index, update);
    score.mutable_data()[index] = terminal;
    std::memcpy(battle.mutable_data() + (index * Sizes::Battle), b.bytes,
                Sizes::Battle);
    std::memcpy(durations.mutable_data() + (index * Sizes::Durations), d.bytes,
                Sizes::Durations);
    const auto [p1_choices, p2_choices] = PKMN::choices(b, r);
    std::fill_n(choices.mutable_data() + (index * 18), 0, 18);
    // std::copy()
    result.mutable_data()[index] = r;
  }

  void uncompress_from_bytes(const py::bytes &data) {
    std::string_view sv(data);
    const char *raw_data = sv.data();
    Train::Battle::CompressedFrames compressed_frames{};
    compressed_frames.read(raw_data);
    auto battle = compressed_frames.battle;
    auto options = PKMN::options();
    auto result = PKMN::result();
    const auto score = PKMN::score(compressed_frames.result);
    for (auto i = 0; i < compressed_frames.updates.size(); ++i) {
      const auto &update = compressed_frames.updates[i];
      write(i, battle, PKMN::durations(options), result, update, score);
      result =
          PKMN::update(battle, update.p1.choice, update.p2.choice, options);
    }
    assert(result == compressed_frames.result);
  }

  static Frames from_bytes(const py::bytes &data, size_t size) {
    Frames f(size);
    f.uncompress_from_bytes(data);
    return f;
  }
};

} // namespace Py::Battle