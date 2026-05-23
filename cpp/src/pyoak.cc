#include <encode/battle/battle.h>
#include <encode/battle/policy.h>
#include <encode/build/compressed-trajectory.h>
#include <libpkmn/data.h>
#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/layout.h>
#include <libpkmn/strings.h>
#include <nn/battle/network.h>
#include <nn/default-hyperparameters.h>
#include <py/battle/encoded-frames.h>
#include <py/battle/frames.h>
#include <py/battle/output.h>
#include <py/build/trajectories.h>
#include <train/battle/compressed-frame.h>
#include <util/parse.h>
#include <util/search.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pkmn.h>

namespace {

namespace py = pybind11;
using namespace PKMN;
using namespace PKMN::Data;

template <std::size_t N, std::size_t M>
std::vector<std::string>
dim_labels_to_vec(const std::array<std::array<char, M>, N> &data) {
  std::vector<std::string> result;
  result.reserve(N);
  for (auto &arr : data) {
    result.emplace_back(arr.data());
  }
  return result;
}

using Py::Battle::Output;

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

Output cpp_inference(const Py::Battle::Frames &battle_frames,
                     std::string network_path, bool discrete = false,
                     std::string budget = "0") {

  RuntimeSearch::AgentParams agent_params{};
  agent_params.eval = network_path;
  agent_params.bandit = "pucb-1.0";
  agent_params.discrete = discrete;
  // TODO fix 1
  agent_params.budget = budget;
  RuntimeSearch::Agent agent{agent_params};
  auto battle =
      *reinterpret_cast<const pkmn_gen1_battle *>(battle_frames.battle.data());
  auto options = PKMN::options();
  auto result = PKMN::result();
  agent.initialize_network(battle);
  mt19937 device{std::random_device{}()};

  Output buffer{battle_frames.size};
  auto value = buffer.value.mutable_data();
  auto p1_logit = buffer.policy_logit.mutable_data();
  auto p2_logit = buffer.policy_logit.mutable_data() + 9;
  auto p1_policy = buffer.policy.mutable_data();
  auto p2_policy = buffer.policy.mutable_data() + 9;
  auto battle_ptr = battle_frames.battle.data();
  auto durations_ptr = battle_frames.durations.data();
  auto k = battle_frames.k.data();
  auto choice = battle_frames.choice.data();
  auto p1_choices = battle_frames.choices.data();
  auto p2_choices = battle_frames.choices.data() + 9;

  for (auto i = 0; i < battle_frames.size; ++i) {
    MCTS::Input input{
        .battle = battle,
        .durations = PKMN::durations(options),
        .result = result,
    };
    RuntimeSearch::Heap heap{};
    const auto output = RuntimeSearch::run(device, input, heap, agent);
    *value = output.initial_value;
    std::copy_n(output.p1.logit.data(), output.p1.k, p1_logit);
    std::copy_n(output.p2.logit.data(), output.p2.k, p2_logit);
    std::copy_n(output.p1.prior.data(), output.p1.k, p1_policy);
    std::copy_n(output.p2.prior.data(), output.p2.k, p2_policy);
    result = PKMN::update(battle, choice[0], choice[1], options);
    // out
    value += 1;
    p1_logit += 18;
    p2_logit += 18;
    p1_policy += 18;
    p2_policy += 18;
    // in
    battle_ptr += sizeof(pkmn_gen1_battle);
    durations_ptr += sizeof(pkmn_gen1_chance_durations);
    k += 2;
    choice += 2;
    p1_choices += 18;
    p2_choices += 18;
  }

  return buffer;
}

const auto solve_matrix(py::array_t<float> p1_payoffs,
                        const int discretize_factor = 256) {
  if (p1_payoffs.ndim() != 2) {
    throw std::runtime_error{"Expecting 2d array"};
  }
  const auto m = p1_payoffs.shape(0);
  const auto n = p1_payoffs.shape(1);
  auto r = p1_payoffs.unchecked<2>();
  std::vector<int> disc;
  disc.resize(m * n);
  for (auto i = 0; i < m; ++i) {
    for (auto j = 0; j < n; ++j) {
      disc[i * n + j] = static_cast<int>(r(i, j) * discretize_factor);
    }
  }
  LRSNash::FastInput solve_input{static_cast<int>(m), static_cast<int>(n),
                                 disc.data(), discretize_factor};
  std::vector<float> nash1;
  std::vector<float> nash2;
  nash1.resize(m + 2);
  nash2.resize(n + 2);
  LRSNash::FloatOneSumOutput solve_output{nash1.data(), nash2.data(), 0};
  LRSNash::solve_fast(&solve_input, &solve_output);
  auto p1 = py::array_t<float>(std::vector<ssize_t>{m});
  auto p2 = py::array_t<float>(std::vector<ssize_t>{n});
  for (int i = 0; i < m; ++i) {
    p1.mutable_unchecked<1>()(i) = nash1[i];
  }
  for (int j = 0; j < n; ++j) {
    p2.mutable_unchecked<1>()(j) = nash2[j];
  }
  return py::make_tuple(p1, p2, solve_output.value);
}

// ============================================================================
// Helpers
// ============================================================================

static inline std::string_view bytes_sv(const py::bytes &b) {
  return {PyBytes_AS_STRING(b.ptr()),
          static_cast<size_t>(PyBytes_GET_SIZE(b.ptr()))};
}

// Pack/unpack helpers for bit fields that PKMN::Volatiles doesn't provide a
// setter for. All operate directly on the bits field.

// Read an arbitrary N-bit unsigned field from `bits` starting at `shift`.
template <int N> static constexpr uint64_t bf_get(uint64_t bits, int shift) {
  return (bits >> shift) & ((1ULL << N) - 1);
}

// Write an arbitrary N-bit unsigned field into `bits` at `shift`.
template <int N>
static constexpr uint64_t bf_set(uint64_t bits, int shift, uint64_t val) {
  const uint64_t mask = ((1ULL << N) - 1) << shift;
  return (bits & ~mask) | ((val & ((1ULL << N) - 1)) << shift);
}

// ============================================================================
// Proxy structs
// (Each is a thin wrapper around a pointer into the owned Battle buffer.)
// ============================================================================

// ----------------------------------------------------------------------------
// StatsProxy
// ----------------------------------------------------------------------------
struct StatsProxy {
  Stats *p;

  uint16_t get_hp() const { return p->hp; }
  uint16_t get_atk() const { return p->atk; }
  uint16_t get_def() const { return p->def; }
  uint16_t get_spe() const { return p->spe; }
  uint16_t get_spc() const { return p->spc; }

  void set_hp(uint16_t v) { p->hp = v; }
  void set_atk(uint16_t v) { p->atk = v; }
  void set_def(uint16_t v) { p->def = v; }
  void set_spe(uint16_t v) { p->spe = v; }
  void set_spc(uint16_t v) { p->spc = v; }
};

// ----------------------------------------------------------------------------
// MoveSlotProxy
// ----------------------------------------------------------------------------
struct MoveSlotProxy {
  MoveSlot *p;

  uint8_t get_id() const { return static_cast<uint8_t>(p->id); }
  uint8_t get_pp() const { return p->pp; }
  void set_id(uint8_t v) { p->id = static_cast<Move>(v); }
  void set_pp(uint8_t v) { p->pp = v; }

  std::string name() const { return std::string(PKMN::move_char_array(p->id)); }
};

// ----------------------------------------------------------------------------
// BoostsProxy
//
// Raw access:  boosts.bytes  -> list[int] of the 4 underlying bytes.
//              boosts.raw    -> uint32_t of the whole thing.
// Named fields use the existing encode_i4 / decode_i4 logic in the C++ type.
// ----------------------------------------------------------------------------
struct BoostsProxy {
  Boosts *p;

  // Getters — named
  int8_t get_atk() const { return p->atk(); }
  int8_t get_def() const { return p->def(); }
  int8_t get_spe() const { return p->spe(); }
  int8_t get_spc() const { return p->spc(); }
  int8_t get_acc() const { return p->acc(); }
  int8_t get_eva() const { return p->eva(); }

  // Setters — named
  void set_atk(int8_t v) { p->set_atk(v); }
  void set_def(int8_t v) { p->set_def(v); }
  void set_spe(int8_t v) { p->set_spe(v); }
  void set_spc(int8_t v) { p->set_spc(v); }
  void set_acc(int8_t v) { p->set_acc(v); }
  void set_eva(int8_t v) { p->set_eva(v); }

  // Raw access — uint32 of the 4 bytes
  uint32_t get_raw() const {
    uint32_t out;
    std::memcpy(&out, p->bytes, 4);
    return out;
  }
  void set_raw(uint32_t v) { std::memcpy(p->bytes, &v, 4); }
};

// ----------------------------------------------------------------------------
// VolatilesProxy
//
// Named flag setters delegate to the C++ helpers that already exist.
// For the bit-packed counter fields where no setter exists in the C++ type
// (state, substitute_hp, transform_species, toxic_counter, disable_move) we
// do the bit math directly using the bf_set helper above.
// `bits` is also exposed raw for any operation not covered by the above.
//
// Bit layout (from data.h / Layout::Volatiles):
//   [0]      bide
//   [1]      thrashing
//   [2]      multi_hit
//   [3]      flinch
//   [4]      charging
//   [5]      binding
//   [6]      invulnerable
//   [7]      confusion (flag)
//   [8]      mist
//   [9]      focus_energy
//   [10]     substitute (flag)
//   [11]     recharging
//   [12]     rage
//   [13]     leech_seed
//   [14]     toxic (flag)
//   [15]     light_screen
//   [16]     reflect
//   [17]     transform (flag)
//   [20:18]  confusion_left (3 bits)
//   [23:21]  attacks       (3 bits)
//   [39:24]  state         (16 bits)
//   [47:40]  substitute_hp (8 bits)
//   [51:48]  transform_species (4 bits)
//   [55:52]  disable_left  (4 bits)
//   [58:56]  disable_move  (3 bits)
//   [63:59]  toxic_counter (5 bits)
// ----------------------------------------------------------------------------
struct VolatilesProxy {
  Volatiles *p;

  // --- boolean flags (setters exist in C++) ---
  bool get_bide() const { return p->bide(); }
  bool get_thrashing() const { return p->thrashing(); }
  bool get_multi_hit() const { return p->multi_hit(); }
  bool get_flinch() const { return p->flinch(); }
  bool get_charging() const { return p->charging(); }
  bool get_binding() const { return p->binding(); }
  bool get_invulnerable() const { return p->invulnerable(); }
  bool get_confusion() const { return p->confusion(); }
  bool get_mist() const { return p->mist(); }
  bool get_focus_energy() const { return p->focus_energy(); }
  bool get_substitute() const { return p->substitute(); }
  bool get_recharging() const { return p->recharging(); }
  bool get_rage() const { return p->rage(); }
  bool get_leech_seed() const { return p->leech_seed(); }
  bool get_toxic() const { return p->toxic(); }
  bool get_light_screen() const { return p->light_screen(); }
  bool get_reflect() const { return p->reflect(); }
  bool get_transform() const { return p->transform(); }

  void set_bide(bool v) { p->set_bide(v); }
  void set_thrashing(bool v) { p->set_thrashing(v); }
  void set_multi_hit(bool v) { p->set_multi_hit(v); }
  void set_flinch(bool v) { p->set_flinch(v); }
  void set_charging(bool v) { p->set_charging(v); }
  void set_binding(bool v) { p->set_binding(v); }
  void set_invulnerable(bool v) { p->set_invulnerable(v); }
  void set_confusion(bool v) { p->set_confusion(v); }
  void set_mist(bool v) { p->set_mist(v); }
  void set_focus_energy(bool v) { p->set_focus_energy(v); }
  void set_substitute(bool v) { p->set_substitute(v); }
  void set_recharging(bool v) { p->set_recharging(v); }
  void set_rage(bool v) { p->set_rage(v); }
  void set_leech_seed(bool v) { p->set_leech_seed(v); }
  void set_toxic(bool v) { p->set_toxic(v); }
  void set_light_screen(bool v) { p->set_light_screen(v); }
  void set_reflect(bool v) { p->set_reflect(v); }
  void set_transform(bool v) { p->set_transform(v); }

  // --- counters with existing C++ setters ---
  uint8_t get_confusion_left() const { return p->confusion_left(); }
  uint8_t get_attacks() const { return p->attacks(); }
  uint8_t get_disable_left() const { return p->disable_left(); }

  void set_confusion_left(uint8_t v) { p->set_confusion_left(v); }
  void set_attacks(uint8_t v) { p->set_attacks(v); }
  void set_disable_left(uint8_t v) { p->set_disable_left(v); }

  // --- bit-packed fields without C++ setters — manual bit math ---
  uint16_t get_state() const { return p->state(); }
  uint8_t get_substitute_hp() const { return p->substitute_hp(); }
  uint8_t get_transform_species() const { return p->transform_species(); }
  uint8_t get_disable_move() const { return p->disable_move(); }
  uint8_t get_toxic_counter() const { return p->toxic_counter(); }

  void set_state(uint16_t v) {
    p->bits = bf_set<16>(p->bits, 24, static_cast<uint64_t>(v));
  }
  void set_substitute_hp(uint8_t v) {
    p->bits = bf_set<8>(p->bits, 40, static_cast<uint64_t>(v));
  }
  void set_transform_species(uint8_t v) {
    p->bits = bf_set<4>(p->bits, 48, static_cast<uint64_t>(v));
  }
  void set_disable_move(uint8_t v) {
    p->bits = bf_set<3>(p->bits, 56, static_cast<uint64_t>(v));
  }
  void set_toxic_counter(uint8_t v) {
    p->bits = bf_set<5>(p->bits, 59, static_cast<uint64_t>(v));
  }

  // --- raw 64-bit access ---
  uint64_t get_bits() const { return p->bits; }
  void set_bits(uint64_t v) { p->bits = v; }

  std::string to_string() const { return volatiles_to_string(*p); }
};

// ----------------------------------------------------------------------------
// PokemonProxy
// ----------------------------------------------------------------------------
struct PokemonProxy {
  Pokemon *p;

  StatsProxy stats() { return {&p->stats}; }
  MoveSlotProxy move(int i) {
    if (i < 0 || i > 3)
      throw std::out_of_range("move index must be 0-3");
    return {&p->moves[static_cast<size_t>(i)]};
  }

  uint16_t get_hp() const { return p->hp; }
  uint8_t get_status() const { return static_cast<uint8_t>(p->status); }
  uint8_t get_species() const { return static_cast<uint8_t>(p->species); }
  uint8_t get_types() const { return p->types; }
  uint8_t get_level() const { return p->level; }

  void set_hp(uint16_t v) { p->hp = v; }
  void set_status(uint8_t v) { p->status = static_cast<Status>(v); }
  void set_species(uint8_t v) { p->species = static_cast<Species>(v); }
  void set_types(uint8_t v) { p->types = v; }
  void set_level(uint8_t v) { p->level = v; }

  int percent() const { return p->percent(); }

  std::string species_name() const {
    return std::string(PKMN::species_char_array(p->species));
  }
  std::string status_name() const { return status_string(p->status); }
  std::string to_string() const {
    return pokemon_to_string(reinterpret_cast<const uint8_t *>(p));
  }
};

// ----------------------------------------------------------------------------
// ActivePokemonProxy
// ----------------------------------------------------------------------------
struct ActivePokemonProxy {
  ActivePokemon *p;

  StatsProxy stats() { return {&p->stats}; }
  BoostsProxy boosts() { return {&p->boosts}; }
  VolatilesProxy volatiles() { return {&p->volatiles}; }
  MoveSlotProxy move(int i) {
    if (i < 0 || i > 3)
      throw std::out_of_range("move index must be 0-3");
    return {&p->moves[static_cast<size_t>(i)]};
  }

  uint8_t get_species() const { return static_cast<uint8_t>(p->species); }
  uint8_t get_types() const { return p->types; }
  void set_species(uint8_t v) { p->species = static_cast<Species>(v); }
  void set_types(uint8_t v) { p->types = v; }

  std::string species_name() const {
    return std::string(PKMN::species_char_array(p->species));
  }
};

// ----------------------------------------------------------------------------
// SideProxy
// ----------------------------------------------------------------------------
struct SideProxy {
  Side *p;

  PokemonProxy pokemon(int i) {
    if (i < 0 || i > 5)
      throw std::out_of_range("pokemon index must be 0-5");
    return {&p->pokemon[static_cast<size_t>(i)]};
  }
  // slot is 1-indexed, uses order[] just like the C++ Side::get()
  PokemonProxy slot(int s) {
    if (s < 1 || s > 6)
      throw std::out_of_range("slot must be 1-6");
    return {&p->get(s)};
  }
  ActivePokemonProxy active() { return {&p->active}; }
  PokemonProxy stored() { return {&p->stored()}; }

  std::array<uint8_t, 6> get_order() const { return p->order; }
  void set_order(std::array<uint8_t, 6> v) { p->order = v; }

  uint8_t get_last_selected_move() const {
    return static_cast<uint8_t>(p->last_selected_move);
  }
  uint8_t get_last_used_move() const {
    return static_cast<uint8_t>(p->last_used_move);
  }
  void set_last_selected_move(uint8_t v) {
    p->last_selected_move = static_cast<Move>(v);
  }
  void set_last_used_move(uint8_t v) {
    p->last_used_move = static_cast<Move>(v);
  }
};

// ----------------------------------------------------------------------------
// DurationProxy / DurationsView
// ----------------------------------------------------------------------------
struct DurationProxy {
  Duration *p;

  uint8_t sleep(int slot) const {
    if (slot < 0 || slot > 5)
      throw std::out_of_range("slot must be 0-5");
    return p->sleep(slot);
  }
  void set_sleep(int slot, uint8_t v) {
    if (slot < 0 || slot > 5)
      throw std::out_of_range("slot must be 0-5");
    p->set_sleep(slot, v);
  }

  uint8_t get_confusion() const { return p->confusion(); }
  uint8_t get_disable() const { return p->disable(); }
  uint8_t get_attacking() const { return p->attacking(); }
  uint8_t get_binding() const { return p->binding(); }

  void set_confusion(uint8_t v) { p->set_confusion(v); }
  void set_disable(uint8_t v) { p->set_disable(v); }
  void set_attacking(uint8_t v) { p->set_attacking(v); }
  void set_binding(uint8_t v) { p->set_binding(v); }

  // raw 32-bit access
  uint32_t get_raw() const { return p->data; }
  void set_raw(uint32_t v) { p->data = v; }
};

struct DurationsView {
  pkmn_gen1_chance_durations raw{};

  explicit DurationsView() = default;
  explicit DurationsView(pkmn_gen1_chance_durations &durations)
      : raw{durations} {}
  explicit DurationsView(py::bytes b) {
    auto sv = bytes_sv(b);
    if (sv.size() != PKMN_GEN1_CHANCE_DURATIONS_SIZE)
      throw std::runtime_error("Durations: expected " +
                               std::to_string(PKMN_GEN1_CHANCE_DURATIONS_SIZE) +
                               " bytes, got " + std::to_string(sv.size()));
    std::memcpy(&raw, sv.data(), PKMN_GEN1_CHANCE_DURATIONS_SIZE);
  }

  Durations &durations() { return view(raw); }
  const Durations &durations() const { return view(raw); }

  DurationProxy get(int i) {
    if (i < 0 || i > 1)
      throw std::out_of_range("index must be 0 or 1");
    return {&durations().get(i)};
  }

  py::bytes bytes() const {
    return py::bytes(reinterpret_cast<const char *>(&raw),
                     PKMN_GEN1_CHANCE_DURATIONS_SIZE);
  }
};

// ----------------------------------------------------------------------------
// BattleView — owns the 384 bytes
// ----------------------------------------------------------------------------
struct BattleView {
  pkmn_gen1_battle raw{};

  explicit BattleView() = default;
  explicit BattleView(pkmn_gen1_battle battle) : raw{battle} {}
  explicit BattleView(py::bytes b) {
    auto sv = bytes_sv(b);
    if (sv.size() != PKMN_GEN1_BATTLE_SIZE)
      throw std::runtime_error("Battle: expected " +
                               std::to_string(PKMN_GEN1_BATTLE_SIZE) +
                               " bytes, got " + std::to_string(sv.size()));
    std::memcpy(&raw, sv.data(), PKMN_GEN1_BATTLE_SIZE);
  }

  Battle &battle() { return view(raw); }
  const Battle &battle() const { return view(raw); }

  SideProxy side(int i) {
    if (i < 0 || i > 1)
      throw std::out_of_range("side index must be 0 or 1");
    return {&battle().sides[static_cast<size_t>(i)]};
  }

  uint16_t get_turn() const { return battle().turn; }
  uint16_t get_last_damage() const { return battle().last_damage; }
  uint64_t get_rng() const { return battle().rng; }

  void set_turn(uint16_t v) { battle().turn = v; }
  void set_last_damage(uint16_t v) { battle().last_damage = v; }
  void set_rng(uint64_t v) { battle().rng = v; }

  py::bytes bytes() const {
    return py::bytes(reinterpret_cast<const char *>(&raw),
                     PKMN_GEN1_BATTLE_SIZE);
  }

  std::string to_string() const { return PKMN::to_string(raw); }
};

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(oak, m) {
  m.doc() =
      "oak — low-level structured view of pkmn_gen1_battle bytes.\n\n"
      "  b    = oak.Battle(raw_bytes)   # 384 bytes\n"
      "  d    = oak.Durations(dur_bytes) # 8 bytes (or Durations() for "
      "zeroed)\n"
      "  side = b.side(0)                   # P1\n"
      "  pkmn = side.pokemon(0)             # storage index 0\n"
      "  lead = side.slot(1)                # battle slot 1 (respects "
      "order[])\n"
      "  act  = side.active\n"
      "  new_bytes = b.bytes()";

  // Module-level constants
  m.attr("BATTLE_SIZE") = static_cast<int>(PKMN_GEN1_BATTLE_SIZE);
  m.attr("DURATIONS_SIZE") = static_cast<int>(PKMN_GEN1_CHANCE_DURATIONS_SIZE);

  // ---- Stats ---------------------------------------------------------------
  py::class_<StatsProxy>(
      m, "Stats", "Base/active stats: hp, atk, def_, spe, spc (all uint16).")
      .def_property("hp", &StatsProxy::get_hp, &StatsProxy::set_hp)
      .def_property("atk", &StatsProxy::get_atk, &StatsProxy::set_atk)
      .def_property("def_", &StatsProxy::get_def, &StatsProxy::set_def)
      .def_property("spe", &StatsProxy::get_spe, &StatsProxy::set_spe)
      .def_property("spc", &StatsProxy::get_spc, &StatsProxy::set_spc)
      .def("__repr__", [](const StatsProxy &s) {
        return "<Stats hp=" + std::to_string(s.get_hp()) +
               " atk=" + std::to_string(s.get_atk()) +
               " def=" + std::to_string(s.get_def()) +
               " spe=" + std::to_string(s.get_spe()) +
               " spc=" + std::to_string(s.get_spc()) + ">";
      });

  // ---- MoveSlot ------------------------------------------------------------
  py::class_<MoveSlotProxy>(m, "MoveSlot", "id (uint8) + pp (uint8).")
      .def_property("id", &MoveSlotProxy::get_id, &MoveSlotProxy::set_id,
                    "Move enum value as uint8.")
      .def_property("pp", &MoveSlotProxy::get_pp, &MoveSlotProxy::set_pp)
      .def("name", &MoveSlotProxy::name, "Move name string.")
      .def("__repr__", [](const MoveSlotProxy &ms) {
        return "<MoveSlot " + ms.name() + " pp=" + std::to_string(ms.get_pp()) +
               ">";
      });

  // ---- Boosts --------------------------------------------------------------
  py::class_<BoostsProxy>(
      m, "Boosts",
      "Stat boosts (int8, range -6..+6). raw exposes the underlying uint32.")
      .def_property("atk", &BoostsProxy::get_atk, &BoostsProxy::set_atk)
      .def_property("def_", &BoostsProxy::get_def, &BoostsProxy::set_def)
      .def_property("spe", &BoostsProxy::get_spe, &BoostsProxy::set_spe)
      .def_property("spc", &BoostsProxy::get_spc, &BoostsProxy::set_spc)
      .def_property("acc", &BoostsProxy::get_acc, &BoostsProxy::set_acc)
      .def_property("eva", &BoostsProxy::get_eva, &BoostsProxy::set_eva)
      .def_property("raw", &BoostsProxy::get_raw, &BoostsProxy::set_raw,
                    "Raw uint32 of the 4 packed boost bytes.")
      .def("__repr__", [](const BoostsProxy &b) {
        return "<Boosts atk=" + std::to_string(b.get_atk()) +
               " def=" + std::to_string(b.get_def()) +
               " spe=" + std::to_string(b.get_spe()) +
               " spc=" + std::to_string(b.get_spc()) +
               " acc=" + std::to_string(b.get_acc()) +
               " eva=" + std::to_string(b.get_eva()) + ">";
      });

  // ---- Volatiles -----------------------------------------------------------
  py::class_<VolatilesProxy>(
      m, "Volatiles",
      "Active-pokemon volatile flags and counters backed by a single uint64.\n"
      "All 18 boolean flags, counters with C++ setters, and the remaining\n"
      "bit-packed fields (state, substitute_hp, transform_species,\n"
      "disable_move, toxic_counter) are individually writable.\n"
      "`bits` exposes the raw uint64 directly.")
      // boolean flags
      .def_property("bide", &VolatilesProxy::get_bide,
                    &VolatilesProxy::set_bide)
      .def_property("thrashing", &VolatilesProxy::get_thrashing,
                    &VolatilesProxy::set_thrashing)
      .def_property("multi_hit", &VolatilesProxy::get_multi_hit,
                    &VolatilesProxy::set_multi_hit)
      .def_property("flinch", &VolatilesProxy::get_flinch,
                    &VolatilesProxy::set_flinch)
      .def_property("charging", &VolatilesProxy::get_charging,
                    &VolatilesProxy::set_charging)
      .def_property("binding", &VolatilesProxy::get_binding,
                    &VolatilesProxy::set_binding)
      .def_property("invulnerable", &VolatilesProxy::get_invulnerable,
                    &VolatilesProxy::set_invulnerable)
      .def_property("confusion", &VolatilesProxy::get_confusion,
                    &VolatilesProxy::set_confusion)
      .def_property("mist", &VolatilesProxy::get_mist,
                    &VolatilesProxy::set_mist)
      .def_property("focus_energy", &VolatilesProxy::get_focus_energy,
                    &VolatilesProxy::set_focus_energy)
      .def_property("substitute", &VolatilesProxy::get_substitute,
                    &VolatilesProxy::set_substitute)
      .def_property("recharging", &VolatilesProxy::get_recharging,
                    &VolatilesProxy::set_recharging)
      .def_property("rage", &VolatilesProxy::get_rage,
                    &VolatilesProxy::set_rage)
      .def_property("leech_seed", &VolatilesProxy::get_leech_seed,
                    &VolatilesProxy::set_leech_seed)
      .def_property("toxic", &VolatilesProxy::get_toxic,
                    &VolatilesProxy::set_toxic)
      .def_property("light_screen", &VolatilesProxy::get_light_screen,
                    &VolatilesProxy::set_light_screen)
      .def_property("reflect", &VolatilesProxy::get_reflect,
                    &VolatilesProxy::set_reflect)
      .def_property("transform", &VolatilesProxy::get_transform,
                    &VolatilesProxy::set_transform)
      // counters (C++ setters exist)
      .def_property("confusion_left", &VolatilesProxy::get_confusion_left,
                    &VolatilesProxy::set_confusion_left)
      .def_property("attacks", &VolatilesProxy::get_attacks,
                    &VolatilesProxy::set_attacks)
      .def_property("disable_left", &VolatilesProxy::get_disable_left,
                    &VolatilesProxy::set_disable_left)
      // bit-packed fields (manual setters via bf_set)
      .def_property("state", &VolatilesProxy::get_state,
                    &VolatilesProxy::set_state,
                    "16-bit Bide/Rage/binding damage accumulator (bits 39:24).")
      .def_property("substitute_hp", &VolatilesProxy::get_substitute_hp,
                    &VolatilesProxy::set_substitute_hp,
                    "Substitute HP (bits 47:40).")
      .def_property("transform_species", &VolatilesProxy::get_transform_species,
                    &VolatilesProxy::set_transform_species,
                    "Transformed-into species index (bits 51:48).")
      .def_property("disable_move", &VolatilesProxy::get_disable_move,
                    &VolatilesProxy::set_disable_move,
                    "Disabled move slot index 0-3 (bits 58:56).")
      .def_property("toxic_counter", &VolatilesProxy::get_toxic_counter,
                    &VolatilesProxy::set_toxic_counter,
                    "Toxic damage counter N (bits 63:59).")
      // raw
      .def_property("bits", &VolatilesProxy::get_bits,
                    &VolatilesProxy::set_bits,
                    "Raw uint64 — all 64 bits at once.")
      .def("__str__", &VolatilesProxy::to_string)
      .def("__repr__", [](const VolatilesProxy &v) {
        return "<Volatiles 0x" + [&] {
          char buf[17];
          std::snprintf(buf, sizeof(buf), "%016llx",
                        (unsigned long long)v.get_bits());
          return std::string(buf);
        }() + ">";
      });

  // ---- Pokemon -------------------------------------------------------------
  py::class_<PokemonProxy>(m, "Pokemon")
      .def("stats", &PokemonProxy::stats,
           py::return_value_policy::reference_internal)
      .def("move", &PokemonProxy::move, py::arg("index"), "Move slot 0-3.",
           py::return_value_policy::reference_internal)
      .def_property("hp", &PokemonProxy::get_hp, &PokemonProxy::set_hp)
      .def_property("status", &PokemonProxy::get_status,
                    &PokemonProxy::set_status,
                    "Status byte (see Data::Status enum values).")
      .def_property("species", &PokemonProxy::get_species,
                    &PokemonProxy::set_species, "Species enum value as uint8.")
      .def_property("types", &PokemonProxy::get_types, &PokemonProxy::set_types,
                    "Packed type byte.")
      .def_property("level", &PokemonProxy::get_level, &PokemonProxy::set_level)
      .def("percent", &PokemonProxy::percent, "Current HP as a percentage.")
      .def("species_name", &PokemonProxy::species_name, "Species name string.")
      .def("status_name", &PokemonProxy::status_name,
           "Status abbreviation string.")
      .def("__str__", &PokemonProxy::to_string)
      .def("__repr__", [](const PokemonProxy &p) {
        return "<Pokemon " + p.species_name() +
               " hp=" + std::to_string(p.get_hp()) + ">";
      });

  // ---- ActivePokemon -------------------------------------------------------
  py::class_<ActivePokemonProxy>(
      m, "ActivePokemon",
      "The in-battle Pokemon for a side: has stats, boosts, volatiles, moves.")
      .def("stats", &ActivePokemonProxy::stats,
           py::return_value_policy::reference_internal)
      .def("boosts", &ActivePokemonProxy::boosts,
           py::return_value_policy::reference_internal)
      .def("volatiles", &ActivePokemonProxy::volatiles,
           py::return_value_policy::reference_internal)
      .def("move", &ActivePokemonProxy::move, py::arg("index"),
           "Move slot 0-3.", py::return_value_policy::reference_internal)
      .def_property("species", &ActivePokemonProxy::get_species,
                    &ActivePokemonProxy::set_species)
      .def_property("types", &ActivePokemonProxy::get_types,
                    &ActivePokemonProxy::set_types)
      .def("species_name", &ActivePokemonProxy::species_name)
      .def("__repr__", [](const ActivePokemonProxy &a) {
        return "<ActivePokemon " + a.species_name() + ">";
      });

  // ---- Side ----------------------------------------------------------------
  py::class_<SideProxy>(m, "Side")
      .def("pokemon", &SideProxy::pokemon, py::arg("index"),
           "Pokemon at storage index 0-5 (independent of battle order).",
           py::return_value_policy::reference_internal)
      .def("slot", &SideProxy::slot, py::arg("slot"),
           "Pokemon in battle slot 1-6, respecting order[].",
           py::return_value_policy::reference_internal)
      .def_property_readonly("active", &SideProxy::active,
                             py::return_value_policy::reference_internal)
      .def("stored", &SideProxy::stored,
           "Shorthand for slot(1) — the currently active (stored) Pokemon.",
           py::return_value_policy::reference_internal)
      .def_property(
          "order", &SideProxy::get_order, &SideProxy::set_order,
          "Six-element array mapping slot (1-indexed) to storage index.")
      .def_property("last_selected_move", &SideProxy::get_last_selected_move,
                    &SideProxy::set_last_selected_move)
      .def_property("last_used_move", &SideProxy::get_last_used_move,
                    &SideProxy::set_last_used_move)
      .def("__repr__", [](const SideProxy &) { return "<Side>"; });

  // ---- Duration ------------------------------------------------------------
  py::class_<DurationProxy>(m, "Duration",
                            "Per-side duration counters for one player.")
      .def("sleep", &DurationProxy::sleep, py::arg("slot"),
           "Sleep counter for Pokemon at order slot 0-5.")
      .def("set_sleep", &DurationProxy::set_sleep, py::arg("slot"),
           py::arg("value"))
      .def_property("confusion", &DurationProxy::get_confusion,
                    &DurationProxy::set_confusion)
      .def_property("disable", &DurationProxy::get_disable,
                    &DurationProxy::set_disable)
      .def_property("attacking", &DurationProxy::get_attacking,
                    &DurationProxy::set_attacking)
      .def_property("binding", &DurationProxy::get_binding,
                    &DurationProxy::set_binding)
      .def_property("raw", &DurationProxy::get_raw, &DurationProxy::set_raw,
                    "Raw uint32 of all packed duration bits.")
      .def("__repr__", [](const DurationProxy &) { return "<Duration>"; });

  // ---- Durations -----------------------------------------------------------
  py::class_<DurationsView>(m, "Durations",
                            "8-byte chance durations for both sides.\n\n"
                            "  d = oak.Durations(dur_bytes)  # from bytes\n"
                            "  d = oak.Durations()           # zeroed\n"
                            "  d.get(0).confusion = 3\n"
                            "  dur_bytes = d.bytes()")
      .def(py::init<>(), "Construct zeroed Durations.")
      .def(py::init<py::bytes>(), py::arg("data"),
           "Construct from 8 raw bytes.")
      .def("get", &DurationsView::get, py::arg("side"),
           "Duration for side 0 or 1.",
           py::return_value_policy::reference_internal)
      .def("bytes", &DurationsView::bytes,
           "Return current (possibly mutated) 8-byte representation.")
      .def("__repr__", [](const DurationsView &) { return "<Durations>"; });

  // ---- Battle --------------------------------------------------------------
  py::class_<BattleView>(m, "Battle",
                         "Structured view of a 384-byte pkmn_gen1_battle.\n\n"
                         "  b = oak.Battle(raw_bytes)\n"
                         "  b.side(0).slot(1).hp = 200\n"
                         "  new_bytes = b.bytes()")
      .def(py::init<py::bytes>(), py::arg("data"),
           "Construct from 384 raw bytes.")
      .def("side", &BattleView::side, py::arg("index"),
           "Side 0 (P1) or Side 1 (P2).",
           py::return_value_policy::reference_internal)
      .def_property("turn", &BattleView::get_turn, &BattleView::set_turn)
      .def_property("last_damage", &BattleView::get_last_damage,
                    &BattleView::set_last_damage)
      .def_property("rng", &BattleView::get_rng, &BattleView::set_rng,
                    "RNG seed as uint64.")
      .def("bytes", &BattleView::bytes,
           "Return the current (possibly mutated) 384-byte battle "
           "representation.")
      .def("__str__", &BattleView::to_string)
      .def("__repr__", [](const BattleView &b) {
        return "<Battle turn=" + std::to_string(b.get_turn()) + ">";
      });

  // Search

  py::class_<RuntimeSearch::Heap>(m, "Heap")
      .def(py::init<>())
      .def("empty", &RuntimeSearch::Heap::empty)
      .def("type", &RuntimeSearch::Heap::type);

  py::class_<RuntimeSearch::Agent>(m, "Agent")
      .def(py::init<>())
      .def_readwrite("budget", &RuntimeSearch::Agent::budget)
      .def_readwrite("bandit", &RuntimeSearch::Agent::bandit)
      .def_readwrite("eval", &RuntimeSearch::Agent::eval)
      .def_readwrite("matrix_ucb", &RuntimeSearch::Agent::matrix_ucb)
      .def_readwrite("discrete", &RuntimeSearch::Agent::discrete)
      .def_readwrite("table", &RuntimeSearch::Agent::table);

  py::class_<MCTS::Output>(m, "Output")
      .def(py::init<>())
      // .def_readonly("m", &MCTS::Output::m)
      // .def_readonly("n", &MCTS::Output::n)
      .def_readonly("iterations", &MCTS::Output::iterations)
      .def_readonly("empirical_value", &MCTS::Output::empirical_value)
      .def_readonly("nash_value", &MCTS::Output::nash_value)
      .def_property_readonly(
          "duration_ms",
          [](const MCTS::Output &o) { return o.duration.count(); })
      .def_property_readonly("visit_matrix",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<size_t>({9, 9});
                               auto r = arr.mutable_unchecked<2>();
                               for (size_t i = 0; i < 9; ++i)
                                 for (size_t j = 0; j < 9; ++j)
                                   r(i, j) = (i < o.p1.k && j < o.p2.k)
                                                 ? o.visit_matrix[i][j]
                                                 : 0;
                               return arr;
                             })
      .def_property_readonly("value_matrix",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>({9, 9});
                               auto r = arr.mutable_unchecked<2>();
                               for (size_t i = 0; i < 9; ++i)
                                 for (size_t j = 0; j < 9; ++j)
                                   r(i, j) = (i < o.p1.k && j < o.p2.k)
                                                 ? o.value_matrix[i][j]
                                                 : 0.0;
                               return arr;
                             })
      // 1D vectors
      .def_property_readonly("p1_prior",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p1.prior[i];
                               return arr;
                             })
      .def_property_readonly("p2_prior",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p2.prior[i];
                               return arr;
                             })
      .def_property_readonly("p1_empirical",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p1.empirical[i];
                               return arr;
                             })
      .def_property_readonly("p2_empirical",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p2.empirical[i];
                               return arr;
                             })
      .def_property_readonly("p1_nash",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p1.nash[i];
                               return arr;
                             })

      .def_property_readonly("p2_nash", [](const MCTS::Output &o) {
        auto arr = py::array_t<double>(9);
        auto r = arr.mutable_unchecked<1>();
        for (size_t i = 0; i < 9; ++i)
          r(i) = o.p2.nash[i];
        return arr;
      });

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

  py::class_<Output>(m, "OutputBuffer")
      .def(py::init<size_t, size_t, size_t>(), py::arg("size"),
           py::arg("pokemon_out_dim") = NN::Battle::Default::pokemon_out_dim,
           py::arg("active_out_dim") = NN::Battle::Default::active_out_dim)
      .def_readonly("size", &Output::size)
      .def_readonly("pokemon_out_dim", &Output::pokemon_out_dim)
      .def_readonly("active_out_dim", &Output::active_out_dim)
      .def_readonly("pokemon", &Output::pokemon)
      .def_readonly("active_pokemon", &Output::active_pokemon)
      .def_readonly("sides", &Output::sides)
      .def_readonly("value", &Output::value)
      .def_readonly("logit", &Output::logit)
      .def_readonly("policy_logit", &Output::policy_logit)
      .def_readonly("policy", &Output::policy)
      .def("clear", &Output::clear);

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

  m.def("cpp_inference", &cpp_inference, py::arg("battle_frames"),
        py::arg("network_path"), py::arg("discrete"), py::arg("budget"));
  m.def("solve_matrix", &solve_matrix, py::arg("row_payoff"),
        py::arg("discretize_factor"));

  m.def("read_battle_data", &read_battle_data, py::arg("path"));
  m.def("read_build_trajectories", &read_build_trajectories);
  // Methods

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

  m.def(
      "parse_battle",
      [](const std::string &battle_string, uint64_t seed = 0x123456) {
        auto [battle, durations] = Parse::parse_battle(battle_string, seed);
        MCTS::Input input{};
        input.battle = battle;
        input.durations = durations;
        input.result = PKMN::result(battle);
        return py::make_tuple(BattleView{battle}, DurationsView{durations},
                              input.result);
      },
      py::arg("battle_string"), py::arg("seed") = 0x123456);

  m.def(
      "update",
      [](BattleView &battle, DurationsView &durations, uint8_t c1, uint8_t c2) {
        auto options = PKMN::options();
        pkmn_gen1_chance_options chance_options{};
        chance_options.durations = durations.raw;
        PKMN::set(options, chance_options);
        auto result = PKMN::update(battle.raw, c1, c2, options);
        durations.raw = PKMN::durations(options);
        return result;
      },
      py::arg("battle"), py::arg("durations"), py::arg("c1"), py::arg("c2"));

  m.def(
      "battle_string",
      [](const BattleView &battle, const DurationsView &durations) {
        return PKMN::battle_data_to_string(battle.raw, durations.raw);
      },
      py::arg("battle"), py::arg("durations"));

  m.def(
      "format",
      [](const BattleView &battle, const DurationsView &durations,
         const MCTS::Output &output) {
        return MCTS::output_string(output,
                                   MCTS::Input{battle.raw, durations.raw});
      },
      py::arg("battle"), py::arg("durations"), py::arg("output"));

  m.def(
      "search",
      [](const MCTS::Input &input, RuntimeSearch::Heap &heap,
         RuntimeSearch::Agent &agent, MCTS::Output output = {}) {
        mt19937 device{std::random_device{}()};
        return RuntimeSearch::run(device, input, heap, agent, output);
      },
      py::arg("input"), py::arg("heap"), py::arg("agent"),
      py::arg("output") = MCTS::Output{});

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

  {
    // Strings
    m.attr("move_names") = dim_labels_to_vec(PKMN::Data::MOVE_CHAR_ARRAY);
    m.attr("species_names") = dim_labels_to_vec(PKMN::Data::SPECIES_CHAR_ARRAY);
    m.attr("active_dim_labels") =
        dim_labels_to_vec(Encode::Battle::Active::dim_labels);
    m.attr("pokemon_dim_labels") =
        dim_labels_to_vec(Encode::Battle::Pokemon::dim_labels);
    m.attr("active_pokemon_dim_labels") =
        dim_labels_to_vec(Encode::Battle::ActivePokemon::dim_labels);
    {
      auto v = dim_labels_to_vec(Encode::Battle::Policy::dim_labels);
      v.push_back(""); // preserve extra empty string
      m.attr("policy_dim_labels") = v;
    }
  }
}

} // namespace