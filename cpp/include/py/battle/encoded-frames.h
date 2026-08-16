#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/policy.h>
#include <py/battle/frames.h>
#include <py/battle/target.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <fstream>
#include <random>
#include <thread>

namespace Py::Battle {

namespace py = pybind11;

struct EncodedFrames : public Target {
  py::array_t<float> pokemon;
  py::array_t<float> active;
  py::array_t<float> moves;
  py::array_t<float> hp;
  py::array_t<int64_t> choice_indices;

  static constexpr size_t pokemon_in_dim = Encode::Battle::Pokemon::n_dim;
  static constexpr size_t active_in_dim = Encode::Battle::Active::n_dim;
  static constexpr size_t moves_in_dim = Encode::Battle::Moves::n_dim;
  using PokemonEncoding = std::array<float, pokemon_in_dim>;
  using ActiveEncoding = std::array<float, active_in_dim>;
  using MovesEncoding = std::array<float, moves_in_dim>;

  EncodedFrames(size_t sz) : Py::Battle::Target{sz} {
    pokemon =
        py::array_t<float>(std::vector<size_t>{size, 2, 6, pokemon_in_dim});
    active = py::array_t<float>(std::vector<size_t>{size, 2, 1, active_in_dim});
    moves = py::array_t<float>(std::vector<size_t>{size, 2, 6, moves_in_dim});
    hp = py::array_t<float>(std::vector<size_t>{size, 2, 6, 1});
    choice_indices = py::array_t<int64_t>(std::vector<size_t>{size, 2, 9});
    clear();
  }

  void clear() {
    Target::clear();
    std::fill_n(choice_indices.mutable_data(), choice_indices.size(),
                int64_t(0));
    std::fill_n(pokemon.mutable_data(), pokemon.size(), 0.0f);
    std::fill_n(active.mutable_data(), active.size(), 0.0f);
    std::fill_n(moves.mutable_data(), moves.size(), 0.0f);
    std::fill_n(hp.mutable_data(), hp.size(), 0.0f);
  }

  void write(const auto index, const pkmn_gen1_battle &b,
             const pkmn_gen1_chance_durations &d, pkmn_result result,
             const Train::Battle::CompressedFrames::Update &update,
             float terminal) {

    Py::Battle::Target::write(index, update);
    score.mutable_data()[index] = terminal;

    auto [hp_, pokemon_, active_, moves_, choice_] = view(index);
    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    const auto [p1_choices, p2_choices] = PKMN::choices(b, result);

    for (auto s = 0; s < 2; ++s) {
      const auto &side = battle.sides[s];
      const auto &duration = durations.get(s);
      const auto &stored = side.stored();

      if (stored.hp == 0) {
        hp_[s][0] = {};
        pokemon_[s][0] = {};
        active_[s][0] = {};
        moves_[s][0] = {};
      } else {
        hp_[s][0] = (float)stored.hp / stored.stats.hp;
        Encode::Battle::Pokemon::write(stored, duration.sleep(0),
                                       pokemon_[s][0].data());
        Encode::Battle::Active::write(side.active, duration,
                                      active_[s][0].data());
        Encode::Battle::Moves::write(side.active.moves, moves_[s][0].data());
      }

      for (auto slot = 2; slot <= 6; ++slot) {
        const auto id = side.order[slot - 1];
        if (id == 0) {
          hp_[s][slot - 1] = {};
          pokemon_[s][slot - 1] = {};
          moves_[s][slot - 1] = {};
        } else {
          const auto &poke = side.pokemon[id - 1];
          if (poke.hp == 0) {
            hp_[s][slot - 1] = {};
            pokemon_[s][slot - 1] = {};
            moves_[s][slot - 1] = {};
          } else {
            const auto sleep = duration.sleep(slot - 1);
            hp_[s][slot - 1] = (float)poke.hp / poke.stats.hp;
            Encode::Battle::Pokemon::write(poke, sleep,
                                           pokemon_[s][slot - 1].data());
            Encode::Battle::Moves::write(poke.moves,
                                         moves_[s][slot - 1].data());
          }
        }
      }

      std::fill_n(choice_[s].data(), 9, Encode::Battle::Policy::n_dim);
      auto j = s ? update.p2.k : update.p1.k;
      auto &choices = s ? p2_choices : p1_choices;
      for (auto i = 0; i < j; ++i) {
        choice_[s][i] =
            Encode::Battle::Policy::get_index(battle.sides[s], choices[i]);
      }
    }
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

  static EncodedFrames from_bytes(const py::bytes &data, size_t sz) {
    EncodedFrames f(sz);
    f.uncompress_from_bytes(data);
    return f;
  }

  auto view(const auto index) {
    using Bench = std::array<std::array<PokemonEncoding, 6>, 2>;
    using Actives = std::array<std::array<ActiveEncoding, 1>, 2>;
    using Moves = std::array<std::array<MovesEncoding, 6>, 2>;
    using ChoiceIndices = std::array<std::array<int64_t, 9>, 2>;
    auto &hp_ = *reinterpret_cast<std::array<std::array<float, 6>, 2> *>(
        hp.mutable_data() + index * (2 * 6 * 1));
    auto &pokemon_ = *reinterpret_cast<Bench *>(
        pokemon.mutable_data() + index * (2 * 6 * pokemon_in_dim));
    auto &active_ = *reinterpret_cast<Actives *>(
        active.mutable_data() + index * (2 * 1 * active_in_dim));
    auto &moves_ = *reinterpret_cast<Moves *>(moves.mutable_data() +
                                              index * (2 * 6 * moves_in_dim));
    auto &choice_ = *reinterpret_cast<ChoiceIndices *>(
        choice_indices.mutable_data() + index * (2 * 9 * 1));
    return std::tie(hp_, pokemon_, active_, moves_, choice_);
  }
};

} // namespace Py::Battle