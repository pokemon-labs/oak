#include <encode/battle/battle.h>
#include <encode/battle/policy.h>
#include <encode/build/compressed-trajectory.h>
#include <libpkmn/log.h>
#include <libpkmn/pkmn.h>
#include <nn/battle/network.h>
#include <nn/default-hyperparameters.h>
#include <py/battle/encoded-frames.h>
#include <py/battle/frames.h>
#include <py/battle/output-buffer.h>
#include <py/build/trajectories.h>
#include <py/libpkmn/data.h>
#include <train/battle/compressed-frame.h>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pkmn.h>

namespace {

template <size_t N, size_t M>
std::vector<std::string>
dim_labels_to_vec(const std::array<std::array<char, M>, N> &data) {
  std::vector<std::string> result;
  result.reserve(N);
  for (auto &arr : data) {
    result.emplace_back(arr.data());
  }
  return result;
}

namespace py = pybind11;
using namespace PKMN;
using namespace PKMN::Data;

py::list read_battle_data(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("read_battle_data: Failed to open file: " + path);
  }
  py::list result;
  size_t total_battles = 0;
  std::vector<char> buffer{};
  while (file.peek() != EOF) {
    uint32_t offset;
    uint16_t frame_count;
    file.read(reinterpret_cast<char *>(&offset), sizeof(offset));
    if (file.gcount() != sizeof(offset))
      throw std::runtime_error("read_battle_data: bad offset read");
    file.read(reinterpret_cast<char *>(&frame_count), sizeof(frame_count));
    if (file.gcount() != sizeof(frame_count))
      throw std::runtime_error("read_battle_data: bad frame count read");
    file.seekg(-(sizeof(offset) + sizeof(frame_count)), std::ios::cur);
    buffer.resize(std::max(static_cast<size_t>(offset), buffer.size()));
    file.read(buffer.data(), offset);
    if (file.gcount() != offset) {
      throw std::runtime_error("read_battle_data: bad battle buffer read");
    }
    result.append(py::make_tuple(py::bytes(buffer.data(), offset),
                                 static_cast<int>(frame_count)));
    ++total_battles;
  }
  return result;
}

struct SampleIndexer {
  std::unordered_map<std::string, py::list> data;

  SampleIndexer() = default;

  size_t size() const { return data.size(); }

  py::list get(const std::string &path) {
    auto it = data.find(path);
    if (it != data.end()) {
      return it->second;
    }
    py::list output;
    int total_offset = 0;
    py::object py_battle_data =
        read_battle_data(path); // Python list of (bytes,int)
    for (auto bf : py_battle_data) {
      py::tuple t = bf.cast<py::tuple>();
      py::bytes battle_bytes = t[0].cast<py::bytes>();
      int frame_count = t[1].cast<int>();
      output.append(py::make_tuple(total_offset, frame_count));
      total_offset += static_cast<int>(PyBytes_Size(battle_bytes.ptr()));
    }
    data[path] = output;
    return output;
  }

  void prune(const std::vector<std::string> &paths) {
    for (auto it = data.begin(); it != data.end();) {
      if (std::find(paths.begin(), paths.end(), it->first) == paths.end()) {
        it = data.erase(it);
      } else {
        ++it;
      }
    }
  }
};

size_t sample(Py::Battle::EncodedFrames &encoded_frames,
              const SampleIndexer &indexer, size_t threads,
              size_t max_battle_length, size_t min_iterations) {

  // flatten indexer data into C++ arrays
  std::vector<const char *> paths;
  std::vector<int> n_battles;
  std::vector<std::vector<int>> offsets;
  std::vector<std::vector<int>> n_frames;
  for (const auto &[path, lst] : indexer.data) {
    paths.push_back(path.c_str());
    n_battles.push_back(py::len(lst));
    offsets.emplace_back();
    n_frames.emplace_back();
    for (auto item : lst) {
      py::tuple t = item.template cast<py::tuple>();
      offsets.back().push_back(t[0].cast<int>());
      n_frames.back().push_back(t[1].cast<int>());
    }
  }

  size_t total_battles = 0;
  for (auto n : n_battles) {
    total_battles += n;
  }

  std::atomic<size_t> count{};
  std::atomic<size_t> errors{};

  const auto start_reading = [&]() {
    std::mt19937 mt{std::random_device{}()};
    std::uniform_int_distribution<size_t> battle_dist(0, total_battles - 1);
    std::vector<char> buffer{};

    auto report_error = [&errors](const std::string &msg) {
      std::cerr << msg << std::endl;
      errors.fetch_add(1);
    };

    try {
      while (!errors.load()) {
        // battle_index is sampled globally and then subtracted until the
        // path_index is found and then battle_index is local to the path's
        // data
        size_t battle_index = battle_dist(mt);
        size_t path_index = 0;
        while (battle_index >= static_cast<size_t>(n_battles[path_index])) {
          battle_index -= n_battles[path_index];
          ++path_index;
        }
        if (path_index >= paths.size()) {
          report_error("bad path index");
          return;
        }

        std::ifstream file(paths[path_index], std::ios::binary);
        if (!file) {
          report_error("unable to open file");
          return;
        }

        const auto battle_offset = offsets[path_index][battle_index];
        file.seekg(battle_offset, std::ios::beg);

        auto offset = Train::Battle::CompressedFrames::Offset{};
        auto frame_count = Train::Battle::CompressedFrames::FrameCount{};

        file.read(reinterpret_cast<char *>(&offset), sizeof(offset));
        if (file.gcount() < sizeof(offset)) {
          report_error("bad offset read");
          return;
        }
        file.read(reinterpret_cast<char *>(&frame_count), sizeof(frame_count));
        if (file.gcount() < sizeof(frame_count)) {
          report_error("bad frame count read");
          return;
        }
        if (frame_count > max_battle_length) {
          continue;
        }
        file.seekg(-(sizeof(offset) + sizeof(frame_count)), std::ios::cur);
        if (offset > 200000 || offset < sizeof(pkmn_gen1_battle)) {
          report_error("bad offset length");
          return;
        }

        buffer.resize(std::max(static_cast<size_t>(offset), buffer.size()));
        file.read(buffer.data(), offset);

        Train::Battle::CompressedFrames compressed;
        compressed.read(buffer.data());

        std::vector<size_t> valid;
        for (size_t i = 0; i < compressed.updates.size(); ++i) {
          if (compressed.updates[i].iterations >= min_iterations) {
            valid.push_back(i);
          }
        }
        if (valid.empty()) {
          continue;
        }

        const auto selected = valid[std::uniform_int_distribution<size_t>(
            0, valid.size() - 1)(mt)];

        auto battle = compressed.battle;
        auto options = PKMN::options();
        auto result = PKMN::result();
        for (auto i = 0; i < selected; ++i) {
          const auto &update = compressed.updates[i];
          result =
              PKMN::update(battle, update.p1.choice, update.p2.choice, options);
        }

        size_t write_index = count.fetch_add(1);
        if (write_index >= encoded_frames.size) {
          return;
        }
        encoded_frames.write(write_index, battle, PKMN::durations(options),
                             result, compressed.updates[selected],
                             PKMN::score(compressed.result));
      }
    } catch (const std::exception &e) {
      report_error(e.what());
    }
  };

  std::vector<std::thread> pool;
  for (size_t i = 0; i < threads; ++i)
    pool.emplace_back(start_reading);
  for (auto &t : pool)
    t.join();

  return errors.load() ? 0 : std::min(count.load(), encoded_frames.size);
}

size_t read_build_trajectories(Py::Build::Trajectories &trajectories,
                               py::list paths, size_t threads) {

  constexpr auto traj_size =
      Encode::Build::CompressedTrajectory<>::size_no_team;

  std::atomic<size_t> count{};
  std::atomic<size_t> errors{};

  const auto start_reading = [&]() {
    std::mt19937 mt{std::random_device{}()};
    std::uniform_int_distribution<size_t> file_dist{0, paths.size() - 1};

    const auto report_error = [&](const auto &msg) -> void {
      std::cerr << msg << std::endl;
      errors.fetch_add(1);
      return;
    };

    try {
      while (!errors.load()) {

        const auto path_index = file_dist(mt);
        std::ifstream file(paths[path_index].cast<std::string>(),
                           std::ios::binary);
        if (!file) {
          return report_error("Failed to open file " +
                              std::to_string(path_index));
        }
        file.seekg(0, std::ios::end);
        auto size = file.tellg();
        file.seekg(0);
        if ((size % traj_size) != 0) {
          return report_error("File " + std::to_string(path_index) + " size " +
                              std::to_string(size) + " is not a multiple of " +
                              std::to_string(traj_size));
        }

        const auto n_trajectories = size / traj_size;

        const auto trajectory_index =
            std::uniform_int_distribution<size_t>{0, n_trajectories - 1}(mt);
        file.seekg(trajectory_index * traj_size);

        Encode::Build::CompressedTrajectory<> traj;
        file.read(reinterpret_cast<char *>(&traj), traj_size);
        if (file.gcount() < traj_size) {
          return report_error("Bad trajectory read");
        }
        const auto format = static_cast<uint8_t>(traj.header.format);
        if (format != 0) {
          return report_error("Only NoTeam trajectories are supported");
        }

        const auto write_index = count.fetch_add(1);
        if (write_index >= trajectories.size) {
          return;
        } else {
          // input.index(write_index).write(traj);
          trajectories.write(write_index, traj);
        }
      }
    } catch (const std::exception &e) {
      return report_error(e.what());
    }
  };

  const auto start_ = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> thread_pool{};
  for (auto i = 0; i < threads; ++i) {
    thread_pool.emplace_back(std::thread{start_reading});
  }
  for (auto i = 0; i < threads; ++i) {
    thread_pool[i].join();
  }

  const auto end_ = std::chrono::high_resolution_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
  // std::cout << ms.count() << std::endl;
  return errors.load() ? 0 : std::min(count.load(), trajectories.size);
}

PYBIND11_MODULE(pyoaktrain, m) {
  // Network

  py::class_<Py::Battle::Frames>(m, "BattleFrames")
      .def(py::init<size_t>())
      .def_static("from_bytes", &Py::Battle::Frames::from_bytes)
      .def_readonly("size", &Py::Battle::Frames::size)
      .def_readonly("k", &Py::Battle::Frames::k)
      .def_readonly("iterations", &Py::Battle::Frames::iterations)
      .def_readonly("empirical_policies",
                    &Py::Battle::Frames::empirical_policies)
      .def_readonly("nash_policies", &Py::Battle::Frames::nash_policies)
      .def_readonly("empirical_value", &Py::Battle::Frames::empirical_value)
      .def_readonly("nash_value", &Py::Battle::Frames::nash_value)
      .def_readonly("score", &Py::Battle::Frames::score)
      .def_readonly("battle", &Py::Battle::Frames::battle)
      .def_readonly("durations", &Py::Battle::Frames::durations)
      .def_readonly("result", &Py::Battle::Frames::result)
      .def_readonly("choices", &Py::Battle::Frames::choices);

  py::class_<Py::Battle::EncodedFrames>(m, "EncodedBattleFrames")
      .def(py::init<size_t>())
      .def("clear", &Py::Battle::EncodedFrames::clear)
      .def_static("from_bytes", &Py::Battle::EncodedFrames::from_bytes)
      .def_readonly("size", &Py::Battle::EncodedFrames::size)
      .def_readonly("k", &Py::Battle::EncodedFrames::k)
      .def_readonly("iterations", &Py::Battle::EncodedFrames::iterations)
      .def_readonly("empirical_policies",
                    &Py::Battle::EncodedFrames::empirical_policies)
      .def_readonly("nash_policies", &Py::Battle::EncodedFrames::nash_policies)
      .def_readonly("empirical_value",
                    &Py::Battle::EncodedFrames::empirical_value)
      .def_readonly("nash_value", &Py::Battle::EncodedFrames::nash_value)
      .def_readonly("score", &Py::Battle::EncodedFrames::score)
      .def_readonly("pokemon", &Py::Battle::EncodedFrames::pokemon)
      .def_readonly("active", &Py::Battle::EncodedFrames::active)
      .def_readonly("hp", &Py::Battle::EncodedFrames::hp)
      .def_readonly("choice_indices",
                    &Py::Battle::EncodedFrames::choice_indices);

  py::class_<Py::Battle::OutputBuffer>(m, "OutputBuffer")
      .def(py::init<size_t, size_t, size_t>(), py::arg("size"),
           py::arg("pokemon_out_dim") = NN::Battle::Default::pokemon_out_dim,
           py::arg("active_out_dim") = NN::Battle::Default::active_out_dim)
      .def_readonly("size", &Py::Battle::OutputBuffer::size)
      .def_readonly("pokemon_out_dim",
                    &Py::Battle::OutputBuffer::pokemon_out_dim)
      .def_readonly("active_out_dim", &Py::Battle::OutputBuffer::active_out_dim)
      .def_readonly("pokemon", &Py::Battle::OutputBuffer::pokemon)
      .def_readonly("active_pokemon", &Py::Battle::OutputBuffer::active_pokemon)
      .def_readonly("sides", &Py::Battle::OutputBuffer::sides)
      .def_readonly("value", &Py::Battle::OutputBuffer::value)
      .def_readonly("logit", &Py::Battle::OutputBuffer::logit)
      .def_readonly("policy_logit", &Py::Battle::OutputBuffer::policy_logit)
      .def_readonly("policy", &Py::Battle::OutputBuffer::policy)
      .def("clear", &Py::Battle::OutputBuffer::clear);

  py::class_<SampleIndexer>(m, "SampleIndexer")
      .def(py::init<>())
      .def("get", &SampleIndexer::get)
      .def("prune", &SampleIndexer::prune)
      .def("size", &SampleIndexer::size);

  py::class_<Py::Build::Trajectories>(m, "BuildTrajectories")
      .def(py::init<size_t>())
      .def_readonly("size", &Py::Build::Trajectories::size)
      .def_readonly("action", &Py::Build::Trajectories::action)
      .def_readonly("mask", &Py::Build::Trajectories::mask)
      .def_readonly("policy", &Py::Build::Trajectories::policy)
      .def_readonly("value", &Py::Build::Trajectories::value)
      .def_readonly("score", &Py::Build::Trajectories::score)
      .def_readonly("start", &Py::Build::Trajectories::start)
      .def_readonly("end", &Py::Build::Trajectories::end)
      .def("clear", &Py::Build::Trajectories::clear);

  m.def("read_battle_data", &read_battle_data, py::arg("path"));
  m.def("read_build_trajectories", &read_build_trajectories);
  m.def(
      "sample",
      [](Py::Battle::EncodedFrames &encoded_frames,
         const SampleIndexer &indexer, size_t threads, size_t max_battle_length,
         size_t min_iterations) {
        return sample(encoded_frames, indexer, threads, max_battle_length,
                      min_iterations);
      },
      py::arg("encoded_frames"), py::arg("indexer"), py::arg("threads"),
      py::arg("max_battle_length"), py::arg("min_iterations"));

  {
    std::vector<std::pair<int, int>> species_move_list;
    species_move_list.reserve(Py::Build::Tensorizer<>::species_move_list_size);
    for (int i = 0; i < Py::Build::Tensorizer<>::species_move_list_size; ++i) {
      auto p = Py::Build::Tensorizer<>::species_move_list(i);
      species_move_list.emplace_back(static_cast<int>(p.first),
                                     static_cast<int>(p.second));
    }
    m.attr("species_move_list") = std::move(species_move_list);
  }
  {
    const auto &src = Py::Build::Tensorizer<>::SPECIES_MOVE_TABLE;
    std::vector<std::vector<int>> species_move_table;
    species_move_table.reserve(src.size());
    std::transform(src.begin(), src.end(),
                   std::back_inserter(species_move_table), [](const auto &row) {
                     return std::vector<int>(row.begin(), row.end());
                   });
    m.attr("species_move_table") = std::move(species_move_table);
  }

  // Battle net hyperparams
  m.attr("pokemon_in_dim") = Encode::Battle::Pokemon::n_dim;
  m.attr("active_in_dim") = Encode::Battle::ActivePokemon::n_dim;
  m.attr("pokemon_hidden_dim") = NN::Battle::Default::pokemon_hidden_dim;
  m.attr("pokemon_out_dim") = NN::Battle::Default::pokemon_out_dim;
  m.attr("active_hidden_dim") = NN::Battle::Default::active_hidden_dim;
  m.attr("active_out_dim") = NN::Battle::Default::active_out_dim;
  m.attr("side_out_dim") = NN::Battle::Default::side_out_dim;
  m.attr("hidden_dim") = NN::Battle::Default::hidden_dim;
  m.attr("value_hidden_dim") = NN::Battle::Default::value_hidden_dim;
  m.attr("policy_hidden_dim") = NN::Battle::Default::policy_hidden_dim;
  m.attr("policy_out_dim") = Encode::Battle::Policy::n_dim;

  // Build net hyperparams
  m.attr("build_policy_hidden_dim") = NN::Build::Default::policy_hidden_dim;
  m.attr("build_value_hidden_dim") = NN::Build::Default::value_hidden_dim;
  m.attr("build_max_actions") = Py::Build::Tensorizer<>::max_actions;
  m.attr("active_dim_labels") =
      dim_labels_to_vec(Encode::Battle::Active::dim_labels);
  m.attr("pokemon_dim_labels") =
      dim_labels_to_vec(Encode::Battle::Pokemon::dim_labels);
  m.attr("active_pokemon_dim_labels") =
      dim_labels_to_vec(Encode::Battle::ActivePokemon::dim_labels);
  auto v = dim_labels_to_vec(Encode::Battle::Policy::dim_labels);
  v.push_back(""); // preserve extra empty string
  m.attr("policy_dim_labels") = v;
}

} // namespace
