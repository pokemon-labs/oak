#include <libpkmn/client.h>
#include <libpkmn/log.h>
#include <libpkmn/pkmn.h>
#include <py/battle/frames.h>
#include <py/battle/output-buffer.h>
#include <py/libpkmn/data.h>
#include <search/data.h>
#include <util/search.h>
#include <util/strings.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pkmn.h>

#include <fstream>
#include <string_view>

namespace {

Py::Battle::OutputBuffer cpp_inference(const Py::Battle::Frames &battle_frames,
                                       Search::Network &network,
                                       Search::SideCache *p1_cache,
                                       Search::SideCache *p2_cache,
                                       Search::Budget budget, int frames) {

  auto battle =
      *reinterpret_cast<const pkmn_gen1_battle *>(battle_frames.battle.data());

  auto options = ::PKMN::options();
  auto result = ::PKMN::result();
  mt19937 device{std::random_device{}()};

  Py::Battle::OutputBuffer buffer{battle_frames.size};
  auto *value = buffer.value.mutable_data();
  auto *p1_logit = buffer.policy_logit.mutable_data();
  auto *p2_logit = buffer.policy_logit.mutable_data() + 9;
  auto *p1_policy = buffer.policy.mutable_data();
  auto *p2_policy = buffer.policy.mutable_data() + 9;
  auto *battle_ptr = battle_frames.battle.data();
  auto *durations_ptr = battle_frames.durations.data();
  auto *k = battle_frames.k.data();

  auto *choice = battle_frames.choice.data();
  auto *p1_choices = battle_frames.choices.data();
  auto *p2_choices = battle_frames.choices.data() + 9;

  auto *p1_side = buffer.sides.mutable_data();
  auto *p2_side = buffer.sides.mutable_data() + buffer.side_out_dim;

  for (auto i = 0; i < std::min(frames, (int)battle_frames.size); ++i) {
    auto heap = Search::Node{};
    const auto output = RuntimeSearch::run(
        device, battle, PKMN::durations(options), budget, Search::PUCB{1.0},
        heap, network, MCTS::Output{}, p1_cache, p2_cache);
    *value = output.initial_value;

    auto& net = *std::dynamic_pointer_cast<NN::Battle::NetworkClamped>(network.get());

    NN::Battle::write_side_embedding<float, NN::Activation::clamp>(
        p1_side, PKMN::view(battle).sides[0], PKMN::view(PKMN::durations(options)).get(0), net);
    NN::Battle::write_side_embedding<float, NN::Activation::clamp>(
        p2_side, PKMN::view(battle).sides[1], PKMN::view(PKMN::durations(options)).get(1), net);

    std::copy_n(output.p1.logit.data(), output.p1.k, p1_logit);
    std::copy_n(output.p2.logit.data(), output.p2.k, p2_logit);
    std::copy_n(output.p1.prior.data(), output.p1.k, p1_policy);
    std::copy_n(output.p2.prior.data(), output.p2.k, p2_policy);
    result = ::PKMN::update(battle, choice[0], choice[1], options);
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
    p1_side += buffer.side_out_dim;
    p2_side += buffer.side_out_dim;
  }

  return buffer;
}

// --- Zero-copy tensor views over NN::Affine<> layers -----------------------
//
// These build a strided py::array that aliases an Affine layer's live Eigen
// storage directly -- no copy. `self` is the owning Search::Network
// python object; passing it as the array's `base` makes pybind11 keep it
// (and therefore the underlying NetworkBase/Eigen buffers) alive for as long
// as the returned array is alive.
//
// IMPORTANT: these views are only valid between a call to resize() (or
// read_parameters(), which resizes internally) and the next resize() call.
// resize() reallocates the Eigen matrices, which silently invalidates any
// outstanding view -- the caller is responsible for not holding views across
// a resize().
//
// weights is always logically (out_dim, in_dim) regardless of Eigen storage
// order; the strides below encode RowMajor vs ColMajor storage so the
// logical shape/indexing seen from Python (and from e.g. torch.from_numpy)
// is identical either way. Only EmbeddingNet's first layer (fc0 of
// pokemon_net/active_net/moves_net) is ColMajor -- see nn/ffn.h.

template <int Order>
py::array affine_weights_view(NN::Affine<Order> &affine, py::object self) {
  constexpr bool col_major = (Order == Eigen::ColMajor);
  const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(affine.out_dim),
                                       static_cast<py::ssize_t>(affine.in_dim)};
  const std::vector<py::ssize_t> strides =
      col_major
          ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(float)),
                                     static_cast<py::ssize_t>(affine.out_dim *
                                                              sizeof(float))}
          : std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(affine.in_dim * sizeof(float)),
                static_cast<py::ssize_t>(sizeof(float))};
  return py::array_t<float>(shape, strides, affine.weights.data(),
                            std::move(self));
}

template <int Order>
py::array affine_biases_view(NN::Affine<Order> &affine, py::object self) {
  const std::vector<py::ssize_t> shape{
      static_cast<py::ssize_t>(affine.out_dim)};
  const std::vector<py::ssize_t> strides{
      static_cast<py::ssize_t>(sizeof(float))};
  return py::array_t<float>(shape, strides, affine.biases.data(),
                            std::move(self));
}

// Invokes F(name, affine_layer) for every float Affine<> layer in a
// NetworkBase: the three embedding nets (each 2 layers) plus MainNet's 8
// layers. Throws for quantized networks, which have no float layers/no
// Affine<float> storage to view. Order of iteration is stable and is the
// order used by write_parameters() below, matching
// Search::Network::read_parameters()'s on-disk layer order.
void for_each_float_layer(NN::Battle::NetworkBase &network, const auto &F) {
  auto *main = network.main_net_float();
  if (!main) {
    throw std::runtime_error{
        "Network: no float layers to expose (network is quantized)."};
  }
  F("pokemon_net.fc0", network.pokemon_net.layer<0>());
  F("pokemon_net.fc1", network.pokemon_net.layer<1>());
  F("active_net.fc0", network.active_net.layer<0>());
  F("active_net.fc1", network.active_net.layer<1>());
  F("moves_net.fc0", network.moves_net.layer<0>());
  F("moves_net.fc1", network.moves_net.layer<1>());
  F("main_net.fc0", main->fc0);
  F("main_net.fc1", main->fc1);
  F("main_net.value_fc2", main->value_fc2);
  F("main_net.value_fc3", main->value_fc3);
  F("main_net.p1_policy_fc2", main->p1_policy_fc2);
  F("main_net.p1_policy_fc3", main->p1_policy_fc3);
  F("main_net.p2_policy_fc2", main->p2_policy_fc2);
  F("main_net.p2_policy_fc3", main->p2_policy_fc3);
}

// Small FNV-1a byte hash. Deterministic across runs/platforms (unlike
// std::hash<string_view>, which is only guaranteed stable within a single
// process), but NOT the same algorithm as oak.torch's Python-side
// hash_bytes() (blake2b) -- this is for cheap local integrity/diffing
// checks (e.g. "did read_parameters round-trip correctly"), not for
// comparing against hashes computed on the old torch.py path.
uint64_t fnv1a(std::string_view bytes, uint64_t h = 0xcbf29ce484222325ULL) {
  for (const unsigned char byte : bytes) {
    h ^= byte;
    h *= 0x100000001b3ULL;
  }
  return h;
}

} // namespace

namespace Search {

namespace py = pybind11;
using namespace ::PKMN::Data;
using namespace Py::PKMN;

PYBIND11_MODULE(pyoaksearch, m) {
  py::module_::import("oak");

  // Heap
  py::class_<Heap>(m, "Heap").def("reset", &Heap::reset);
  py::class_<Node, Heap>(m, "Node").def(py::init<>());
  py::class_<Table, Heap>(m, "Table").def(py::init<>());

  // Eval
  py::class_<Eval>(m, "Eval");
  py::class_<PokeEngine, Eval>(m, "PokeEngine").def(py::init<>());
  py::class_<MonteCarlo, Eval>(m, "MonteCarlo").def(py::init<>());
  py::class_<Network, Eval>(m, "Network")
      .def(py::init<int>(), py::arg("activation") = 1)
      .def("read_parameters", &Network::read_parameters, py::arg("path"))
      .def(
          "resize",
          [](Network &net, uint32_t ph, uint32_t po, uint32_t ah, uint32_t ao,
             uint32_t mh, uint32_t mo, uint32_t h, uint32_t value,
             uint32_t policy) {
            auto network = net.get();
            network->resize(ph, po, ah, ao, mh, mo, h, value, policy);
          },
          py::arg("pokemon_hidden"), py::arg("pokemon_out"),
          py::arg("active_hidden"), py::arg("active_out"),
          py::arg("moves_hidden"), py::arg("moves_out"), py::arg("main_hidden"),
          py::arg("value_hidden"), py::arg("policy_hidden"))
      .def(
          "initialize",
          [](Network &net, uint64_t seed) {
            auto network = net.get();
            mt19937 device{seed};
            network->initialize(device);
          },
          py::arg("seed"))
      .def("activation",
           [](const Network &network) -> int {
             return network.get()->activation_type();
           })
      .def("quantize", &Network::quantize)
      .def(
          "forward_side",
          [](Network &net, const Py::PKMN::SideProxy &side,
             const Py::PKMN::DurationProxy &duration,
             std::optional<std::reference_wrapper<Search::SideCache>> cache)
              -> py::array {
            auto network = net.get();
            py::array result;
            const auto write_embedding = [&](auto &net) {
              using Network = typename std::remove_cvref_t<decltype(net)>;
              using T = Network::T;
              constexpr auto activation = Network::act;

              const auto dim = net.side_embedding_dim();
              py::array_t<T> arr(static_cast<py::ssize_t>(dim));
              T *embedding = arr.mutable_data();
              if (cache) {
                auto &side_cache =
                    std::get<NN::Battle::SideCache<T>>(cache->get().data);
                NN::Battle::write_side_embedding<T, activation>(
                    embedding, *side.p, *duration.p, net, side_cache);
              } else {
                NN::Battle::write_side_embedding<T, activation>(
                    embedding, *side.p, *duration.p, net);
              }
              result = std::move(arr);
            };
            NN::Battle::visit_network(network, write_embedding);
            return result;
          },
          py::arg("side"), py::arg("duration"), py::arg("cache") = std::nullopt)
      .def(
          "named_parameters",
          [](py::object self) -> py::dict {
            Network &net = self.cast<Network &>();
            auto network = net.get();
            py::dict result;
            for_each_float_layer(*network, [&](const char *name, auto &affine) {
              result[py::str(name)] =
                  py::make_tuple(affine_weights_view(affine, self),
                                 affine_biases_view(affine, self));
            });
            return result;
          },
          "Zero-copy {name: (weights, biases)} view over every float "
          "Affine<> layer (embedding nets' fc0/fc1, and MainNet's 8 "
          "layers). Each array aliases this Network's live Eigen storage "
          "directly -- wrap with torch.from_numpy(...) (and "
          "torch.nn.Parameter(...) to make it optimizable) to train this "
          "Network in place, with no copy back to disk/C++ required. "
          "Invalid after a subsequent call to resize(); take views only "
          "once the network's final shape is set. Raises if this Network "
          "is quantized (no float layers).")
      .def(
          "weights",
          [](py::object self, const std::string &layer) -> py::array {
            Network &net = self.cast<Network &>();
            auto network = net.get();
            std::optional<py::array> result;
            for_each_float_layer(*network, [&](const char *name, auto &affine) {
              if (!result && layer == name) {
                result = affine_weights_view(affine, self);
              }
            });
            if (!result) {
              throw std::runtime_error{"Network: unknown layer '" + layer +
                                       "'"};
            }
            return std::move(*result);
          },
          py::arg("layer"),
          "Zero-copy view of a single layer's weight matrix by name (see "
          "named_parameters() for the full name list). Same aliasing/"
          "lifetime rules as named_parameters().")
      .def(
          "biases",
          [](py::object self, const std::string &layer) -> py::array {
            Network &net = self.cast<Network &>();
            auto network = net.get();
            std::optional<py::array> result;
            for_each_float_layer(*network, [&](const char *name, auto &affine) {
              if (!result && layer == name) {
                result = affine_biases_view(affine, self);
              }
            });
            if (!result) {
              throw std::runtime_error{"Network: unknown layer '" + layer +
                                       "'"};
            }
            return std::move(*result);
          },
          py::arg("layer"),
          "Zero-copy view of a single layer's bias vector by name. Same "
          "aliasing/lifetime rules as named_parameters().")
      .def(
          "write_parameters",
          [](Network &net, const std::string &path) {
            auto network = net.get();
            if (!network->main_net_float()) {
              throw std::runtime_error{
                  "write_parameters: cannot serialize a quantized network."};
            }
            std::ofstream file(path, std::ios::binary);
            if (!file) {
              throw std::runtime_error{"write_parameters: could not open '" +
                                       path + "'"};
            }
            uint8_t header[8] = {};
            // Mirrors Network::read_parameters()'s header parsing
            // (py/search/data.h): byte 0 is 0 for relu, 1 for clamp.
            // Bytes 1-7 are reserved/unused there too.
            header[0] =
                dynamic_cast<NN::Battle::NetworkClamped *>(network.get()) ? 1
                                                                          : 0;
            file.write(reinterpret_cast<const char *>(header), sizeof(header));
            for_each_float_layer(*network, [&](const char *, auto &affine) {
              file.write(reinterpret_cast<const char *>(&affine.in_dim),
                         sizeof(uint32_t));
              file.write(reinterpret_cast<const char *>(&affine.out_dim),
                         sizeof(uint32_t));
              file.write(reinterpret_cast<const char *>(affine.biases.data()),
                         affine.out_dim * sizeof(float));
              // On-disk weight layout is always row-major, matching
              // Affine::read_parameters (nn/affine.h), regardless of this
              // layer's in-memory Eigen storage order.
              using Layer = std::remove_reference_t<decltype(affine)>;
              const typename Layer::MatrixRowMajor row_major_weights =
                  affine.weights;
              file.write(
                  reinterpret_cast<const char *>(row_major_weights.data()),
                  affine.out_dim * affine.in_dim * sizeof(float));
            });
            if (!file) {
              throw std::runtime_error{"write_parameters: write failed for '" +
                                       path + "'"};
            }
          },
          py::arg("path"),
          "Serialize this network to disk in the format read_parameters() "
          "expects. New counterpart to read_parameters() -- previously "
          "only implemented Python-side in oak.torch.")
      .def(
          "hash",
          [](Network &net) {
            auto network = net.get();
            uint64_t h = 0;
            for_each_float_layer(*network, [&](const char *, auto &affine) {
              using Layer = std::remove_reference_t<decltype(affine)>;
              const typename Layer::MatrixRowMajor row_major_weights =
                  affine.weights;
              const std::string_view weight_bytes(
                  reinterpret_cast<const char *>(row_major_weights.data()),
                  affine.out_dim * affine.in_dim * sizeof(float));
              const std::string_view bias_bytes(
                  reinterpret_cast<const char *>(affine.biases.data()),
                  affine.out_dim * sizeof(float));
              h = NN::combine_hash(h, fnv1a(weight_bytes));
              h = NN::combine_hash(h, fnv1a(bias_bytes));
            });
            return h;
          },
          "Order-dependent FNV-1a fingerprint of every float layer's "
          "parameters. Useful for e.g. asserting read_parameters() round-"
          "tripped correctly, or that two Network objects hold identical "
          "weights. NOT bit-for-bit comparable to oak.torch's old "
          "hash_bytes()/blake2b-based hash() -- different algorithm.");

  py::class_<SideCache>(m, "SideCache")
      .def(py::init<>())
      .def(
          "precompute",
          [](SideCache &cache, Network &network,
             const Py::PKMN::SideProxy &side,
             int index) { cache.precompute(network.get(), *side.p, index); },
          py::arg("network"), py::arg("side"), py::arg("index"))
      .def("quantize", [](SideCache &cache,
                          Network &network) { cache.quantize(network.get()); })
      .def("is_quantized",
           [](SideCache &cache) { return cache.is_quantized(); })
      .def(
          "pokemon_embedding",
          [](const SideCache &cache, std::size_t side_index, std::size_t key,
             uint32_t dim) -> py::object {
            return std::visit(
                [&](const auto &c) -> py::object {
                  using T = typename std::decay_t<decltype(c)>::value_type;
                  const T *ptr = c.pokemon_cache[side_index].data[key].get();
                  if (!ptr) {
                    return py::none();
                  }
                  return py::array_t<T>(dim, ptr);
                },
                cache.data);
          },
          py::arg("side_index"), py::arg("key"), py::arg("dim"))
      .def(
          "moves_embedding",
          [](const SideCache &cache, std::size_t side_index, std::size_t key,
             uint32_t dim) -> py::object {
            return std::visit(
                [&](const auto &c) -> py::object {
                  using T = typename std::decay_t<decltype(c)>::value_type;
                  const T *ptr =
                      c.pokemon_moves_cache[side_index].data[key].get();
                  if (!ptr) {
                    return py::none();
                  }
                  return py::array_t<T>(dim, ptr);
                },
                cache.data);
          },
          py::arg("side_index"), py::arg("key"), py::arg("dim"));
  // Budget
  py::class_<Budget>(m, "Budget");
  py::class_<Iterations, Budget>(m, "Iterations").def(py::init<size_t>());
  // Params
  py::class_<BanditParams>(m, "Bandit");
  py::class_<UCB, BanditParams>(m, "UCB").def(py::init<float>(), py::arg("c"));
  py::class_<PUCB, BanditParams>(m, "PUCB").def(py::init<float>(),
                                                py::arg("c"));
  py::class_<UCB1, BanditParams>(m, "UCB1").def(py::init<float>(),
                                                py::arg("c"));
  py::class_<Exp3, BanditParams>(m, "Exp3").def(
      py::init<float, float>(), py::arg("lr"), py::arg("exploration"));
  py::class_<PExp3, BanditParams>(m, "PExp3")
      .def(py::init<float, float>(), py::arg("lr"), py::arg("exploration"));
  py::class_<MatrixUCB, BanditParams>(m, "MatrixUCB")
      .def(py::init<BanditParams, float, uint32_t, uint32_t, uint32_t>(),
           py::arg("bandit"), py::arg("c"), py::arg("delay") = 0,
           py::arg("interval") = 1, py::arg("minimum") = 0);

  py::class_<MCTS::Output::Side>(m, "SideOutput")
      .def(py::init<>())
      .def_readonly("k", &MCTS::Output::Side::k)
      .def_readonly("choices", &MCTS::Output::Side::choices)
      .def_readonly("logit", &MCTS::Output::Side::logit)
      .def_readonly("prior", &MCTS::Output::Side::prior)
      .def_readonly("empirical", &MCTS::Output::Side::empirical)
      .def_readonly("nash", &MCTS::Output::Side::nash);

  py::class_<MCTS::Output>(m, "Output")
      .def(py::init<>())
      .def_readonly("iterations", &MCTS::Output::iterations)
      .def_readonly("duration", &MCTS::Output::duration)
      .def_readonly("initial_value", &MCTS::Output::initial_value)
      .def_readonly("empirical_value", &MCTS::Output::empirical_value)
      .def_readonly("nash_value", &MCTS::Output::nash_value)
      .def_readonly("p1", &MCTS::Output::p1)
      .def_readonly("p2", &MCTS::Output::p2)
      .def_readonly("visit_matrix", &MCTS::Output::visit_matrix)
      .def_readonly("value_matrix", &MCTS::Output::value_matrix)
      .def_property_readonly("empirical_matrix", [](const MCTS::Output &o) {
        auto arr = py::array_t<double>({9, 9});
        auto r = arr.mutable_unchecked<2>();
        for (size_t i = 0; i < 9; ++i)
          for (size_t j = 0; j < 9; ++j)
            r(i, j) = (i < o.p1.k && j < o.p2.k)
                          ? (o.visit_matrix[i][j]
                                 ? o.value_matrix[i][j] / o.visit_matrix[i][j]
                                 : 0.5)
                          : 0.0;
        return arr;
      });

  m.def(
      "output_string",
      [](const pkmn_gen1_battle &battle,
         const pkmn_gen1_chance_durations &durations, const pkmn_result result,
         const MCTS::Output &output) {
        return MCTS::output_string(output,
                                   MCTS::Input{battle, durations, result});
      },
      py::arg("battle"), py::arg("durations"), py::arg("result"),
      py::arg("output"));
  m.def(
      "run",
      [](const pkmn_gen1_battle &battle,
         const pkmn_gen1_chance_durations &durations,
         const Search::Budget &budget, const Search::BanditParams &bandit,
         Search::Heap &heap, Search::Eval &eval, MCTS::Output output,
         std::optional<std::reference_wrapper<Search::SideCache>> p1_cache = {},
         std::optional<std::reference_wrapper<Search::SideCache>> p2_cache =
             {}) {
        mt19937 device{std::random_device{}()};
        return RuntimeSearch::run(
            device, battle, durations, budget, bandit, heap, eval, output,
            p1_cache.has_value() ? &p1_cache.value().get() : nullptr,
            p2_cache.has_value() ? &p2_cache.value().get() : nullptr);
      },
      py::arg("battle"), py::arg("durations"), py::arg("budget"),
      py::arg("bandit"), py::arg("heap"), py::arg("eval"),
      py::arg("output") = MCTS::Output{}, py::arg("p1_cache") = std::nullopt,
      py::arg("p2_cache") = std::nullopt);

  m.def(
      "cpp_inference",
      [](const Py::Battle::Frames &battle_frames, Search::Network &network,
         Search::Budget budget, int frames,
         std::optional<std::reference_wrapper<Search::SideCache>> p1_cache = {},
         std::optional<std::reference_wrapper<Search::SideCache>> p2_cache =
             {}) {
        return cpp_inference(
            battle_frames, network,
            p1_cache.has_value() ? &p1_cache.value().get() : nullptr,
            p2_cache.has_value() ? &p2_cache.value().get() : nullptr, budget,
            frames);
      },
      py::arg("battle_frames"), py::arg("network"), py::arg("budget"),
      py::arg("frames"), py::arg("p1_cache") = std::nullopt,
      py::arg("p2_cache") = std::nullopt);
}

} // namespace Search