#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>

namespace NN {

namespace Battle::Default {

static constexpr int pokemon_hidden_dim = 256;
static constexpr int pokemon_out_dim = 30;
static constexpr int active_hidden_dim = 256;
static constexpr int active_out_dim = 54;
static constexpr int moves_hidden_dim = 256;
static constexpr int moves_out_dim = 25;
static constexpr int side_out_dim =
    active_out_dim + 6 * (pokemon_out_dim + moves_out_dim);
static constexpr int hidden_dim = 64;
static constexpr int value_hidden_dim = 32;
static constexpr int policy_hidden_dim = 64;
} // namespace Battle::Default

namespace Build::Default {
static constexpr int policy_hidden_dim = 128;
static constexpr int value_hidden_dim = 128;
} // namespace Build::Default

} // namespace NN