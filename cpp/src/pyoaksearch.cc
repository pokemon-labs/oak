#include <libpkmn/client.h>
#include <libpkmn/log.h>
#include <libpkmn/pkmn.h>
#include <py/battle/frames.h>
#include <py/battle/output-buffer.h>
#include <py/libpkmn/data.h>
// #include <util/search.h>
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

// Py::Battle::OutputBuffer cpp_inference(const Py::Battle::Frames
// &battle_frames,
//                                        std::string network_path,
//                                        bool discrete = false,
//                                        std::string budget = "0") {

//   RuntimeSearch::AgentParams agent_params{};
//   agent_params.eval = network_path;
//   agent_params.bandit = "pucb-1.0";
//   agent_params.discrete = discrete;
//   agent_params.budget = budget;
//   RuntimeSearch::Agent agent{agent_params};
//   auto battle =
//       *reinterpret_cast<const pkmn_gen1_battle
//       *>(battle_frames.battle.data());
//   auto options = ::PKMN::options();
//   auto result = ::PKMN::result();
//   agent.initialize_network(battle);
//   mt19937 device{std::random_device{}()};

//   Py::Battle::OutputBuffer buffer{battle_frames.size};
//   auto value = buffer.value.mutable_data();
//   auto p1_logit = buffer.policy_logit.mutable_data();
//   auto p2_logit = buffer.policy_logit.mutable_data() + 9;
//   auto p1_policy = buffer.policy.mutable_data();
//   auto p2_policy = buffer.policy.mutable_data() + 9;
//   auto battle_ptr = battle_frames.battle.data();
//   auto durations_ptr = battle_frames.durations.data();
//   auto k = battle_frames.k.data();
//   auto choice = battle_frames.choice.data();
//   auto p1_choices = battle_frames.choices.data();
//   auto p2_choices = battle_frames.choices.data() + 9;

//   for (auto i = 0; i < battle_frames.size; ++i) {
//     MCTS::Input input{
//         .battle = battle,
//         .durations = ::PKMN::durations(options),
//         .result = result,
//     };
//     RuntimeSearch::Heap heap{};
//     const auto output = RuntimeSearch::run(device, input, heap, agent);
//     *value = output.initial_value;
//     std::copy_n(output.p1.logit.data(), output.p1.k, p1_logit);
//     std::copy_n(output.p2.logit.data(), output.p2.k, p2_logit);
//     std::copy_n(output.p1.prior.data(), output.p1.k, p1_policy);
//     std::copy_n(output.p2.prior.data(), output.p2.k, p2_policy);
//     result = ::PKMN::update(battle, choice[0], choice[1], options);
//     // out
//     value += 1;
//     p1_logit += 18;
//     p2_logit += 18;
//     p1_policy += 18;
//     p2_policy += 18;
//     // in
//     battle_ptr += sizeof(pkmn_gen1_battle);
//     durations_ptr += sizeof(pkmn_gen1_chance_durations);
//     k += 2;
//     choice += 2;
//     p1_choices += 18;
//     p2_choices += 18;
//   }

//   return buffer;
// }

PYBIND11_MODULE(pyoaksearch, m) {
  py::module_::import("oak");

  // Heap
  py::class_<Heap>(m, "Heap").def("reset", &Heap::reset);
  py::class_<Node>(m, "Node").def(py::init<>());
  py::class_<Table>(m, "Table").def(py::init<>());

  // Eval
  py::class_<Eval>(m, "Eval");
  py::class_<PokeEngine>(m, "PokeEngine").def(py::init<>());
  py::class_<MonteCarlo>(m, "MonteCarlo").def(py::init<>());
  py::class_<Network>(m, "Network")
      .def(py::init<>())
      .def("read_parameters", &Network::read_parameters, py::arg("path"))
      .def("zero_initialize", &Network::zero_initialize)
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
          py::arg("seed"));
  py::class_<SideCache>(m, "SideCache")
      .def(py::init<>())
      .def(
          "precompute",
          [](SideCache &cache, Network &network,
             const Py::PKMN::SideProxy &side, int index) {
            const auto foo = [&](auto &net) {
              if (std::holds_alternative<NN::Battle::SideCache<float>>(
                      cache.data)) {
                auto &c = std::get<NN::Battle::SideCache<float>>(cache.data);
                c.precompute(net, *side.p, index);
              } else {
              }
            };
            NN::Battle::visit_network(network.get(), foo);
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
          py::arg("side_index"), py::arg("key"), py::arg("dim"));
  // Budget
  py::class_<Budget>(m, "Budget");
  // .def(py::init([](py::object obj) {
  //     if (py::isinstance<py::int_>(obj)) {
  //         return Budget{
  //             Budget::Variant{obj.cast<std::size_t>()}
  //         };
  //     }
  //     if (py::isinstance(obj,
  //     py::module_::import("datetime").attr("timedelta"))) {
  //         // Convert timedelta -> milliseconds
  //         auto total_seconds =
  //             obj.attr("total_seconds")().cast<double>();

  //         return Budget{
  //             Budget::Variant{
  //                 std::chrono::milliseconds(
  //                     static_cast<int64_t>(total_seconds * 1000))
  //             }
  //         };
  //     }

  //     throw py::type_error(
  //         "Budget must be an int or datetime.timedelta");
  // }));

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

  // m.def("cpp_inference", &cpp_inference, py::arg("battle_frames"),
  //       py::arg("network_path"), py::arg("discrete"), py::arg("budget"));

  m.def(
      "output_string",
      [](const BattleView &battle, const DurationsView &durations,
         const pkmn_result result, const MCTS::Output &output) {
        return MCTS::output_string(
            output, MCTS::Input{battle.raw, durations.raw, result});
      },
      py::arg("battle"), py::arg("durations"), py::arg("result"),
      py::arg("output"));
}

} // namespace Py::Search