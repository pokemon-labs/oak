#include <libpkmn/client.h>
#include <libpkmn/log.h>
#include <libpkmn/pkmn.h>
#include <py/battle/frames.h>
#include <py/battle/output-buffer.h>
#include <py/libpkmn/data.h>
#include <util/search.h>
#include <util/strings.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pkmn.h>

#include <py/search/data.h>

#include <fstream>
#include <string_view>

consteval bool check_buckets() {
  using PKMN::Data::Status;
  constexpr std::array<Status, 8> status_array{
      Status::None,      Status::Poison, Status::Burn,  Status::Freeze,
      Status::Paralysis, Status::Rest1,  Status::Rest2, Status::Rest3};
  PKMN::Pokemon pokemon{};
  pokemon.stats.hp = 714;
  // std::unordered_map<int, int> count{};
  std::array<int, 1000> count{};
  // TODO
  const auto get_entry = [&count](const auto &pokemon, auto sleep) {
    const auto key = Encode::Battle::Key::get_key(pokemon, sleep);
    count[key] += 1;
  };

  pokemon.hp = 1;
  // non slept status conditions
  for (const auto status : status_array) {
    pokemon.status = status;
    get_entry(pokemon, 0);
  }
  // slept
  pokemon.status = Status::Sleep1;
  for (auto sleep = 1; sleep <= 7; ++sleep) {
    get_entry(pokemon, sleep);
  }

  for (auto k = 0; k < 15; ++k) {
    if (count[k] != 1) {
      return false;
    }
  }
  return true;
}

static_assert(check_buckets());

namespace {

// --- Zero-copy tensor views over NN::Affine<> layers -----------------------
//
// These build a strided py::array that aliases an Affine layer's live Eigen
// storage directly -- no copy. `self` is the owning Py::Search::Network
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
// Py::Search::Network::read_parameters()'s on-disk layer order.
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

namespace Py::Search {

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

  {
    // Defaults are pulled directly from the C++ struct so the Python kwarg
    // defaults can never drift from cpp/include/search/poke-engine-evaluate.h.
    static const ::PokeEngine::Params poke_engine_defaults{};

    py::class_<PokeEngine, Eval>(m, "PokeEngine")
        .def(py::init([](float pokemon_alive, float pokemon_hp,
                          float pokemon_attack_boost,
                          float pokemon_defense_boost,
                          float pokemon_special_attack_boost,
                          float pokemon_speed_boost,
                          float pokemon_boost_multiplier_6,
                          float pokemon_boost_multiplier_5,
                          float pokemon_boost_multiplier_4,
                          float pokemon_boost_multiplier_3,
                          float pokemon_boost_multiplier_2,
                          float pokemon_boost_multiplier_1,
                          float pokemon_boost_multiplier_0,
                          float pokemon_boost_multiplier_neg_1,
                          float pokemon_boost_multiplier_neg_2,
                          float pokemon_boost_multiplier_neg_3,
                          float pokemon_boost_multiplier_neg_4,
                          float pokemon_boost_multiplier_neg_5,
                          float pokemon_boost_multiplier_neg_6,
                          float pokemon_frozen, float pokemon_asleep,
                          float pokemon_paralyzed, float pokemon_toxic,
                          float pokemon_poisoned, float pokemon_burned,
                          float leech_seed, float substitute,
                          float confusion, float reflect,
                          float light_screen) {
               auto engine = std::make_unique<PokeEngine>();
               auto &params = engine->get();
               params.POKEMON_ALIVE = pokemon_alive;
               params.POKEMON_HP = pokemon_hp;
               params.POKEMON_ATTACK_BOOST = pokemon_attack_boost;
               params.POKEMON_DEFENSE_BOOST = pokemon_defense_boost;
               params.POKEMON_SPECIAL_ATTACK_BOOST =
                   pokemon_special_attack_boost;
               params.POKEMON_SPEED_BOOST = pokemon_speed_boost;
               params.POKEMON_BOOST_MULTIPLIER_6 = pokemon_boost_multiplier_6;
               params.POKEMON_BOOST_MULTIPLIER_5 = pokemon_boost_multiplier_5;
               params.POKEMON_BOOST_MULTIPLIER_4 = pokemon_boost_multiplier_4;
               params.POKEMON_BOOST_MULTIPLIER_3 = pokemon_boost_multiplier_3;
               params.POKEMON_BOOST_MULTIPLIER_2 = pokemon_boost_multiplier_2;
               params.POKEMON_BOOST_MULTIPLIER_1 = pokemon_boost_multiplier_1;
               params.POKEMON_BOOST_MULTIPLIER_0 = pokemon_boost_multiplier_0;
               params.POKEMON_BOOST_MULTIPLIER_NEG_1 =
                   pokemon_boost_multiplier_neg_1;
               params.POKEMON_BOOST_MULTIPLIER_NEG_2 =
                   pokemon_boost_multiplier_neg_2;
               params.POKEMON_BOOST_MULTIPLIER_NEG_3 =
                   pokemon_boost_multiplier_neg_3;
               params.POKEMON_BOOST_MULTIPLIER_NEG_4 =
                   pokemon_boost_multiplier_neg_4;
               params.POKEMON_BOOST_MULTIPLIER_NEG_5 =
                   pokemon_boost_multiplier_neg_5;
               params.POKEMON_BOOST_MULTIPLIER_NEG_6 =
                   pokemon_boost_multiplier_neg_6;
               params.POKEMON_FROZEN = pokemon_frozen;
               params.POKEMON_ASLEEP = pokemon_asleep;
               params.POKEMON_PARALYZED = pokemon_paralyzed;
               params.POKEMON_TOXIC = pokemon_toxic;
               params.POKEMON_POISONED = pokemon_poisoned;
               params.POKEMON_BURNED = pokemon_burned;
               params.LEECH_SEED = leech_seed;
               params.SUBSTITUTE = substitute;
               params.CONFUSION = confusion;
               params.REFLECT = reflect;
               params.LIGHT_SCREEN = light_screen;
               return engine;
             }),
             py::arg("pokemon_alive") = poke_engine_defaults.POKEMON_ALIVE,
             py::arg("pokemon_hp") = poke_engine_defaults.POKEMON_HP,
             py::arg("pokemon_attack_boost") =
                 poke_engine_defaults.POKEMON_ATTACK_BOOST,
             py::arg("pokemon_defense_boost") =
                 poke_engine_defaults.POKEMON_DEFENSE_BOOST,
             py::arg("pokemon_special_attack_boost") =
                 poke_engine_defaults.POKEMON_SPECIAL_ATTACK_BOOST,
             py::arg("pokemon_speed_boost") =
                 poke_engine_defaults.POKEMON_SPEED_BOOST,
             py::arg("pokemon_boost_multiplier_6") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_6,
             py::arg("pokemon_boost_multiplier_5") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_5,
             py::arg("pokemon_boost_multiplier_4") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_4,
             py::arg("pokemon_boost_multiplier_3") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_3,
             py::arg("pokemon_boost_multiplier_2") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_2,
             py::arg("pokemon_boost_multiplier_1") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_1,
             py::arg("pokemon_boost_multiplier_0") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_0,
             py::arg("pokemon_boost_multiplier_neg_1") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_NEG_1,
             py::arg("pokemon_boost_multiplier_neg_2") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_NEG_2,
             py::arg("pokemon_boost_multiplier_neg_3") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_NEG_3,
             py::arg("pokemon_boost_multiplier_neg_4") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_NEG_4,
             py::arg("pokemon_boost_multiplier_neg_5") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_NEG_5,
             py::arg("pokemon_boost_multiplier_neg_6") =
                 poke_engine_defaults.POKEMON_BOOST_MULTIPLIER_NEG_6,
             py::arg("pokemon_frozen") = poke_engine_defaults.POKEMON_FROZEN,
             py::arg("pokemon_asleep") = poke_engine_defaults.POKEMON_ASLEEP,
             py::arg("pokemon_paralyzed") =
                 poke_engine_defaults.POKEMON_PARALYZED,
             py::arg("pokemon_toxic") = poke_engine_defaults.POKEMON_TOXIC,
             py::arg("pokemon_poisoned") =
                 poke_engine_defaults.POKEMON_POISONED,
             py::arg("pokemon_burned") = poke_engine_defaults.POKEMON_BURNED,
             py::arg("leech_seed") = poke_engine_defaults.LEECH_SEED,
             py::arg("substitute") = poke_engine_defaults.SUBSTITUTE,
             py::arg("confusion") = poke_engine_defaults.CONFUSION,
             py::arg("reflect") = poke_engine_defaults.REFLECT,
             py::arg("light_screen") = poke_engine_defaults.LIGHT_SCREEN)
        .def_property(
            "pokemon_alive",
            [](const PokeEngine &self) { return self.get().POKEMON_ALIVE; },
            [](PokeEngine &self, float v) { self.get().POKEMON_ALIVE = v; })
        .def_property(
            "pokemon_hp",
            [](const PokeEngine &self) { return self.get().POKEMON_HP; },
            [](PokeEngine &self, float v) { self.get().POKEMON_HP = v; })
        .def_property(
            "pokemon_attack_boost",
            [](const PokeEngine &self) {
              return self.get().POKEMON_ATTACK_BOOST;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_ATTACK_BOOST = v;
            })
        .def_property(
            "pokemon_defense_boost",
            [](const PokeEngine &self) {
              return self.get().POKEMON_DEFENSE_BOOST;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_DEFENSE_BOOST = v;
            })
        .def_property(
            "pokemon_special_attack_boost",
            [](const PokeEngine &self) {
              return self.get().POKEMON_SPECIAL_ATTACK_BOOST;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_SPECIAL_ATTACK_BOOST = v;
            })
        .def_property(
            "pokemon_speed_boost",
            [](const PokeEngine &self) {
              return self.get().POKEMON_SPEED_BOOST;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_SPEED_BOOST = v;
            })
        .def_property(
            "pokemon_boost_multiplier_6",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_6;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_6 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_5",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_5;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_5 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_4",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_4;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_4 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_3",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_3;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_3 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_2",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_2;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_2 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_1",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_1;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_1 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_0",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_0;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_0 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_neg_1",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_NEG_1;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_NEG_1 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_neg_2",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_NEG_2;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_NEG_2 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_neg_3",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_NEG_3;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_NEG_3 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_neg_4",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_NEG_4;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_NEG_4 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_neg_5",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_NEG_5;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_NEG_5 = v;
            })
        .def_property(
            "pokemon_boost_multiplier_neg_6",
            [](const PokeEngine &self) {
              return self.get().POKEMON_BOOST_MULTIPLIER_NEG_6;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_BOOST_MULTIPLIER_NEG_6 = v;
            })
        .def_property(
            "pokemon_frozen",
            [](const PokeEngine &self) { return self.get().POKEMON_FROZEN; },
            [](PokeEngine &self, float v) { self.get().POKEMON_FROZEN = v; })
        .def_property(
            "pokemon_asleep",
            [](const PokeEngine &self) { return self.get().POKEMON_ASLEEP; },
            [](PokeEngine &self, float v) { self.get().POKEMON_ASLEEP = v; })
        .def_property(
            "pokemon_paralyzed",
            [](const PokeEngine &self) {
              return self.get().POKEMON_PARALYZED;
            },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_PARALYZED = v;
            })
        .def_property(
            "pokemon_toxic",
            [](const PokeEngine &self) { return self.get().POKEMON_TOXIC; },
            [](PokeEngine &self, float v) { self.get().POKEMON_TOXIC = v; })
        .def_property(
            "pokemon_poisoned",
            [](const PokeEngine &self) { return self.get().POKEMON_POISONED; },
            [](PokeEngine &self, float v) {
              self.get().POKEMON_POISONED = v;
            })
        .def_property(
            "pokemon_burned",
            [](const PokeEngine &self) { return self.get().POKEMON_BURNED; },
            [](PokeEngine &self, float v) { self.get().POKEMON_BURNED = v; })
        .def_property(
            "leech_seed",
            [](const PokeEngine &self) { return self.get().LEECH_SEED; },
            [](PokeEngine &self, float v) { self.get().LEECH_SEED = v; })
        .def_property(
            "substitute",
            [](const PokeEngine &self) { return self.get().SUBSTITUTE; },
            [](PokeEngine &self, float v) { self.get().SUBSTITUTE = v; })
        .def_property(
            "confusion",
            [](const PokeEngine &self) { return self.get().CONFUSION; },
            [](PokeEngine &self, float v) { self.get().CONFUSION = v; })
        .def_property(
            "reflect",
            [](const PokeEngine &self) { return self.get().REFLECT; },
            [](PokeEngine &self, float v) { self.get().REFLECT = v; })
        .def_property(
            "light_screen",
            [](const PokeEngine &self) { return self.get().LIGHT_SCREEN; },
            [](PokeEngine &self, float v) { self.get().LIGHT_SCREEN = v; });
  }

  {
    // Defaults pulled from MCTS::MonteCarlo's own constructor defaults.
    static const ::MCTS::MonteCarlo monte_carlo_defaults{};

    py::class_<MonteCarlo, Eval>(m, "MonteCarlo")
        .def(py::init([](bool forbid_switches, bool forbid_status) {
               auto engine = std::make_unique<MonteCarlo>();
               engine->forbid_switches() = forbid_switches;
               engine->forbid_status() = forbid_status;
               return engine;
             }),
             py::arg("forbid_switches") =
                 monte_carlo_defaults.forbid_switches,
             py::arg("forbid_status") = monte_carlo_defaults.forbid_status)
        .def_property(
            "forbid_switches",
            [](const MonteCarlo &self) { return self.forbid_switches(); },
            [](MonteCarlo &self, bool v) { self.forbid_switches() = v; })
        .def_property(
            "forbid_status",
            [](const MonteCarlo &self) { return self.forbid_status(); },
            [](MonteCarlo &self, bool v) { self.forbid_status() = v; });
  }
  py::class_<Network, Eval>(m, "Network")
      .def(py::init<>())
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
      .def(
          "forward_side",
          [](Network &net, const Py::PKMN::SideProxy &side,
             const Py::PKMN::DurationProxy &duration,
             std::optional<std::reference_wrapper<Py::Search::SideCache>> cache)
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
             const Py::PKMN::SideProxy &side, int index) {
            const auto precompute = [&](auto &net) {
              using Network = typename std::remove_cvref_t<decltype(net)>;
              constexpr auto activation = Network::act;
              if (std::holds_alternative<NN::Battle::SideCache<float>>(
                      cache.data)) {
                auto &c = std::get<NN::Battle::SideCache<float>>(cache.data);
                c.precompute<activation>(net, *side.p, index);
              } else {
                auto &c = std::get<NN::Battle::SideCache<uint8_t>>(cache.data);
                c.precompute<activation>(net, *side.p, index);
              }
            };
            NN::Battle::visit_network(network.get(), precompute);
          },
          py::arg("network"), py::arg("side"), py::arg("index"))
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
  py::class_<BanditParams>(m, "BanditParams");
  py::class_<UCB, BanditParams>(m, "UCB").def(py::init<float>(), py::arg("c"));

  py::class_<MCTS::Output>(m, "Output")
      .def(py::init<>())
      .def_readonly("iterations", &MCTS::Output::iterations)
      .def_readonly("empirical_value", &MCTS::Output::empirical_value)
      .def_readonly("nash_value", &MCTS::Output::nash_value)
      .def_property_readonly("m", [](const MCTS::Output &o) { return o.p1.k; })
      .def_property_readonly("n", [](const MCTS::Output &o) { return o.p2.k; })
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
      .def_property_readonly(
          "empirical_matrix",
          [](const MCTS::Output &o) {
            auto arr = py::array_t<double>({9, 9});
            auto r = arr.mutable_unchecked<2>();
            for (size_t i = 0; i < 9; ++i)
              for (size_t j = 0; j < 9; ++j)
                r(i, j) = (i < o.p1.k && j < o.p2.k)
                              ? (o.visit_matrix[i][j] ? o.value_matrix[i][j] /
                                                            o.visit_matrix[i][j]
                                                      : 0.5)
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
  m.def(
      "output_string",
      [](const BattleView &battle, const DurationsView &durations,
         const pkmn_result result, const MCTS::Output &output) {
        return MCTS::output_string(
            output, MCTS::Input{battle.raw, durations.raw, result});
      },
      py::arg("battle"), py::arg("durations"), py::arg("result"),
      py::arg("output"));
  m.def(
      "search",
      [](const BattleView &battle, const DurationsView &durations,
         const Py::Search::Budget &budget,
         const Py::Search::BanditParams &params, Py::Search::Heap &heap,
         Py::Search::Eval &eval, MCTS::Output output,
         std::optional<std::reference_wrapper<Py::Search::SideCache>> p1_cache =
             {},
         std::optional<std::reference_wrapper<Py::Search::SideCache>> p2_cache =
             {}) {
        mt19937 device{std::random_device{}()};
        return RuntimeSearch::run(
            device, battle, durations, budget, params, heap, eval, output,
            p1_cache.has_value() ? &p1_cache.value().get() : nullptr,
            p2_cache.has_value() ? &p2_cache.value().get() : nullptr);
      },
      py::arg("battle"), py::arg("durations"), py::arg("budget"),
      py::arg("params"), py::arg("heap"), py::arg("eval"),
      py::arg("output") = MCTS::Output{}, py::arg("p1_cache") = std::nullopt,
      py::arg("p2_cache") = std::nullopt);
}

} // namespace Py::Search