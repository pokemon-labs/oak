#include <libpkmn/client.h>
#include <libpkmn/log.h>
#include <libpkmn/pkmn.h>
#include <py/libpkmn/data.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pkmn.h>

namespace Py::PKMN {

namespace py = pybind11;
using namespace PKMN;
using namespace PKMN::Data;

PYBIND11_MODULE(pyoaklog, m) {
  py::module_::import("oak");

  m.def(
      "update",
      [](BattleView &battle, DurationsView &durations, uint8_t c1, uint8_t c2,
         int view = 0) -> std::pair<int, std::vector<std::string>> {
        std::array<uint8_t, 512> buffer{};
        pkmn_gen1_log_options log_options{buffer.data(), buffer.size()};
        auto options = PKMN::options(log_options);
        pkmn_gen1_chance_options chance_options{};
        chance_options.durations = durations.raw;
        PKMN::set(options, chance_options);
        auto result = PKMN::update(battle.raw, c1, c2, options);
        durations.raw = PKMN::durations(options);
        const auto finish =
            [&](auto &parser) -> std::pair<int, std::vector<std::string>> {
          parser.battle = battle.raw;
          parser.parse();
          return {result, parser.log};
        };
        if (view == 0) {
          PKMN::Log::Parser<PKMN::Log::View::omniscient> parser{buffer.data()};
          return finish(parser);
        } else if (view == 1) {
          PKMN::Log::Parser<PKMN::Log::View::p1> parser{buffer.data()};
          return finish(parser);
        } else if (view == 2) {
          PKMN::Log::Parser<PKMN::Log::View::p2> parser{buffer.data()};
          return finish(parser);
        } else {
          throw std::runtime_error{
              "Invalid view, expected 0 (omniscient), 1 (p1), or 2 (p2)"};
          return {};
        }
      },
      py::arg("battle"), py::arg("durations"), py::arg("c1"), py::arg("c2"),
      py::arg("view"));

  m.def(
      "compare_battles",
      [](const BattleView &public_battle, const DurationsView &public_durations,
         const BattleView &truth_battle,
         const DurationsView &truth_durations) -> std::pair<bool, std::string> {
        std::string reason = "";
        bool matches = PKMN::Client::compare_battles(
            PKMN::view(public_battle.raw), PKMN::view(public_durations.raw),
            PKMN::view(truth_battle.raw), PKMN::view(truth_durations.raw),
            reason);
        return {matches, reason};
      });
}

} // namespace Py::PKMN