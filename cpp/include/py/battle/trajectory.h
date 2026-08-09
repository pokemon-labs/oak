#include <pkmn.h>
#include <search/mcts.h>

namespace Py::Battle::Trajectory {
struct Header {
  bool empirical_matrix;
};

struct Update {};

Header header;
pkmn_gen1_battle battle;
}; // namespace Py::Battle::Trajectory