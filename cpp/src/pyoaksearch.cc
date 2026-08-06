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

namespace Py::Search {

namespace py = pybind11;
using namespace ::PKMN::Data;
using namespace Py::PKMN;

// Every field of PokeEngine::Params, in (python_kwarg_name, C++ field) pairs.
// Used to generate both the PokeEngine keyword-argument constructor and the
// read/write properties below, so the two stay in sync.
#define POKE_ENGINE_PARAMS(X)                                                \
  X(pokemon_alive, POKEMON_ALIVE)                                            \
  X(pokemon_hp, POKEMON_HP)                                                  \
  X(pokemon_attack_boost, POKEMON_ATTACK_BOOST)                              \
  X(pokemon_defense_boost, POKEMON_DEFENSE_BOOST)                            \
  X(pokemon_special_attack_boost, POKEMON_SPECIAL_ATTACK_BOOST)              \
  X(pokemon_speed_boost, POKEMON_SPEED_BOOST)                                \
  X(pokemon_boost_multiplier_6, POKEMON_BOOST_MULTIPLIER_6)                  \
  X(pokemon_boost_multiplier_5, POKEMON_BOOST_MULTIPLIER_5)                  \
  X(pokemon_boost_multiplier_4, POKEMON_BOOST_MULTIPLIER_4)                  \
  X(pokemon_boost_multiplier_3, POKEMON_BOOST_MULTIPLIER_3)                  \
  X(pokemon_boost_multiplier_2, POKEMON_BOOST_MULTIPLIER_2)                  \
  X(pokemon_boost_multiplier_1, POKEMON_BOOST_MULTIPLIER_1)                  \
  X(pokemon_boost_multiplier_0, POKEMON_BOOST_MULTIPLIER_0)                  \
  X(pokemon_boost_multiplier_neg_1, POKEMON_BOOST_MULTIPLIER_NEG_1)          \
  X(pokemon_boost_multiplier_neg_2, POKEMON_BOOST_MULTIPLIER_NEG_2)          \
  X(pokemon_boost_multiplier_neg_3, POKEMON_BOOST_MULTIPLIER_NEG_3)          \
  X(pokemon_boost_multiplier_neg_4, POKEMON_BOOST_MULTIPLIER_NEG_4)          \
  X(pokemon_boost_multiplier_neg_5, POKEMON_BOOST_MULTIPLIER_NEG_5)          \
  X(pokemon_boost_multiplier_neg_6, POKEMON_BOOST_MULTIPLIER_NEG_6)          \
  X(pokemon_frozen, POKEMON_FROZEN)                                          \
  X(pokemon_asleep, POKEMON_ASLEEP)                                          \
  X(pokemon_paralyzed, POKEMON_PARALYZED)                                    \
  X(pokemon_toxic, POKEMON_TOXIC)                                            \
  X(pokemon_poisoned, POKEMON_POISONED)                                      \
  X(pokemon_burned, POKEMON_BURNED)                                          \
  X(leech_seed, LEECH_SEED)                                                  \
  X(substitute, SUBSTITUTE)                                                  \
  X(confusion, CONFUSION)                                                    \
  X(reflect, REFLECT)                                                        \
  X(light_screen, LIGHT_SCREEN)

PYBIND11_MODULE(pyoaksearch, m) {
  py::module_::import("oak");

  // Heap
  py::class_<Heap>(m, "Heap").def("reset", &Heap::reset);
  py::class_<Node, Heap>(m, "Node").def(py::init<>());
  py::class_<Table, Heap>(m, "Table").def(py::init<>());

  // Eval
  py::class_<Eval>(m, "Eval");
  {
    // Defaults are pulled directly from the C++ struct so that the Python
    // kwarg defaults can never drift from cpp/include/search/poke-engine-evaluate.h.
    static const ::PokeEngine::Params poke_engine_defaults{};

    auto poke_engine_cls =
        py::class_<PokeEngine, Eval>(m, "PokeEngine")
            .def(py::init(
                     [](float pokemon_alive, float pokemon_hp,
                        float pokemon_attack_boost, float pokemon_defense_boost,
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
                        float leech_seed, float substitute, float confusion,
                        float reflect, float light_screen) {
                       auto engine = std::make_unique<PokeEngine>();
                       auto &params = engine->get();
#define X(pyname, cppname) params.cppname = pyname;
                       POKE_ENGINE_PARAMS(X)
#undef X
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
                 py::arg("light_screen") = poke_engine_defaults.LIGHT_SCREEN);

#define X(pyname, cppname)                                                   \
  poke_engine_cls.def_property(                                              \
      #pyname, [](const PokeEngine &self) { return self.get().cppname; },    \
      [](PokeEngine &self, float value) { self.get().cppname = value; });
    POKE_ENGINE_PARAMS(X)
#undef X
  }
  py::class_<MonteCarlo, Eval>(m, "MonteCarlo").def(py::init<>());
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
          py::arg("side"), py::arg("duration"),
          py::arg("cache") = std::nullopt);

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