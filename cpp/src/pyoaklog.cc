#include <libpkmn/data.h>
#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/data/status.h>
#include <libpkmn/data/strings.h>
#include <libpkmn/init.h>
#include <libpkmn/layout.h>
#include <libpkmn/log.h>
#include <libpkmn/pkmn.h>
#include <libpkmn/strings.h>

#include <py/libpkmn/data.h>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pkmn.h>

namespace {

namespace py = pybind11;
using namespace PKMN;
using namespace PKMN::Data;

PYBIND11_MODULE(pyoaklog, m) {
  py::module_::import("oak");

  m.def(
      "update_",
      [](BattleView &battle, DurationsView &durations, uint8_t c1,
         uint8_t c2) -> std::pair<int, std::vector<std::string>> {
        std::array<uint8_t, 512> buffer{};
        pkmn_gen1_log_options log_options{buffer.data(), buffer.size()};
        auto options = PKMN::options(log_options);
        pkmn_gen1_chance_options chance_options{};
        chance_options.durations = durations.raw;
        PKMN::set(options, chance_options);
        auto result = PKMN::update(battle.raw, c1, c2, options);
        durations.raw = PKMN::durations(options);
        PKMN::Log::Parser parser{buffer.data()};
        parser.battle = battle.raw;
        parser.parse();
        return {result, parser.log};
        // return result;
      },
      py::arg("battle"), py::arg("durations"), py::arg("c1"), py::arg("c2"));
}

} // namespace