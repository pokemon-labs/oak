# Warning: Incomplete/Incorrect

This readme is probably at least 50% done and is intended to aid developers. It is also out-of-date; Some parts of the code have been refactored since this document was last updated.

# Project Structure

The core of Oak is a C++ header library that fleshes out the barebones C API and closely resembles the Zig source-code interface. This code is all contained in 

# libpkmn

We use `libpkmn` since it matches Showdown's behaviour and was designed with search in mind.

There are various comptime options when building libpkmn that need to be mentioned.

* `chance`/`calc`

Theese are the most important options by far since MCTS is not really possible without them. They solve the problem of 'randomizing' the state at the start of the iteration and allow for damage clamping. Because the cart and Showdown fundamentally rely on *hidden* variables (not *private* but rather unknown to either player) we cannot simply change the battle seed to sample transitions in the Markov sense, we must also sample the hidden variables based on public observations of sleep turns, wrap turns, etc. The damage roll clamping is necessary because there are 39 possible rolls in gen 1. 

* `log`

This enables log output that closely matches Showdown's omniscient log. We do not use this information for search although this data can also solve the problem of matching the battle updates with tree traversal. Instead we use `pkmn_gen1_chance_actions` per the the previous options.

The following options are minor in comparison:

* `ebc=false`

Competitive teams are not in danger of endless battles, so disabling improves performance

* `miss=false`

The infamous "255" is patched for no reason other that than its too unlikely to meaningfully impact search behaviour or output, and disabling this mechanic might improve learning by decreasing variance in game outcomes.

* `advance=false`

This option is only needed to exactly rng-match Showdown, but disabling it improves performance and the sampling behaviour is 'equilvalent'.

* `key=true`

Automatically masks *secret* (not private) information in the `pkmnn_gen1_chance_actions` so that it can be used as a key for the `std::map` that defines `Node` trees.

From now until stated otherwise, the sections refer to directories in the `include` folder and subsections will refer to files.

# libpkmn/data/

These headers just deplicate the Move, Speices, Types, and Status enums and some tables like boosts and the type chart. Necessary quality of life.

## data.h

Here we redefine the real data: `Pokemon`, `ActivePokemon`, `Side`, `Battle`. These structs are not actually used for battle updates, which is the real logic and difficulty of a simulator. We use libpkmn's opaque `pkmn_gen1_battle_update` function for that. Instead the structs are used to fill in the functionality not provided by libpkmn, like initializing a battle or randomizing the hidden variables.

Regarding C++, we never actually initalize a `PKMN::Pokemon` etc object, instead we use `reinterpret_cast` to view a `pkmn_gen1_battle` and `pkmn_gen1_chance_durations` as `PKMN::Battle` and `PKMN::Durations`, resp. This spares us a lot of unreadable 'bit poking` and probably does not have a performance penalty.

## layout.h

More names for magic constants.

## init.h

Constrcting a battle requires little work on our end, we only need to compute base stats. However the use case of an analysis engine requires us to construct battles that are in progress and with as little tiresome specification as possible (e.g. "level is 100").

This motivates the `Init::Pokemon` struct

```cpp
struct Pokemon {
  Species species;
  std::array<Move, 4> moves;
  std::array<uint8_t, 4> pp = {64, 64, 64, 64};
  // actual hp value, ignored if negative
  int hp = -1;
  // percent
  uint percent = 100;
  Status status = Status::None;
  uint8_t sleeps = 0;
  uint8_t level = 100;
};
```

This data is more of a default description of a pokemon. It only requires species and moves, all other values are default and effectively describe a fresh slot.

> `percent` is the expected input since its rare for a use to have an exact hp value. The `hp` is only used when its value is non negative and it will override `percent` in that case

> `sleeps` is where the sleep *duration* (the number of turns slept as reported by the Showdown cliet) are entered


## strings.h

Debug and quality of life functions.

The most important function is `battle_data_to_string` produces a string representation of the battle and durations:

```
Alakazam: 100% (313/313) Psychic:16 SeismicToss:32 ThunderWave:31 Recover:32 
  Starmie: 100% (323/323) Surf:24 Thunderbolt:24 ThunderWave:32 Recover:32 
  Rhydon: 100% (413/413) Earthquake:16 BodySlam:24 Substitute:16 RockSlide:16 
  Chansey: 100% (703/703) Sing:24 IceBeam:16 ThunderWave:32 SoftBoiled:16 
  Snorlax: 100% (523/523) BodySlam:24 Reflect:32 IceBeam:16 Rest:16 
  Tauros: 100% (353/353) BodySlam:24 HyperBeam:8 Blizzard:8 Earthquake:16 
--- --- --- 2 --- --- ---
(spe 158>>39) 
Snorlax: 100% (523/523) PAR BodySlam:24 Reflect:32 HyperBeam:8 Rest:16 
  Cloyster: 100% (303/303) Blizzard:8 Clamp:16 Explosion:8 Rest:16 
  Alakazam: 100% (313/313) Psychic:16 SeismicToss:32 ThunderWave:32 Recover:32 
  Chansey: 100% (703/703) IceBeam:16 Thunderbolt:24 ThunderWave:32 SoftBoiled:16 
  Jynx: 100% (333/333) LovelyKiss:16 Blizzard:8 Psychic:16 Rest:16 
  Tauros: 100% (353/353) BodySlam:24 HyperBeam:8 Blizzard:8 Earthquake:16 
```

TODO better example, go over how volatiles and durations are formatted.

## pkmn.h

This top level header provides C++ style wrappers for libpkmn functions. In particular, the function (templates) take their parameters by reference instead of by pointer.

Importantly it provides a turn 0 initializer for battles using two 'teams' `p1` and `p2`

```cpp
const auto p1 = Teams::benchmark_teams[0];
const auto p2 = Teams::benchmark_teams[1];
auto battle = PKMN::battle(p1, p2, seed);
```

This function is templated for its team parameters and it is expected that a range of `PKMN::Set` is provided.  A 'set' must simply have species and a range of moves.

Constructing in-progress battles is handled in `util/parse.h` (TODO link to section).

The `pkmn_gen1_battle_update` function is now simply `update`, and the result is marked `[[no_discard]]`. The helper calls `pkmn_gen1_battle_options_set(&options, NULL, NULL, NULL)` automatically.

```cpp
const auto _ = PKMN::update(battle, c1, c2, options);
```

This update wrapper is actually a template that can take `Species` and `Move` enum values for the choices

```cpp
using PKMN::Data::Species;
// replace fainted active with Alakazam (0 is interpreted as a pkmn_choice aka char which makes it a pass)
const auto result = PKMN::update(battle, Species::Alakazam, 0, options);
```

There are some factory functions for default constructed (zero'd) pkmn data structures. This is mostly for appearances.

```cpp
auto options = PKMN::options();
// instead of 
pkmn_gen1_battle_options options{};
```

The `pkmn_gen1_battle_choices` wrapper takes just a battle and result and returns a pair of vectors of `pkmn_choice` with no padding.

```cpp
const auto [p1_choices, p2_choices] = PKMN::choices(battle, result);
if ((p1_choices.size() * p2_choices.size()) == 1) {
  assert(battle.turn == 0);
}
```

A `pkmn_result` can now be constructed/recovered from the battle state, which is necessary for parsing user input:

```cpp
auto result = PKMN::result(battle);
```

Or it can be constructed by specifying its bit-packed data, the result type (win/loss/draw/error/none) and the requests (move/switch/pass) for each side.

```cpp
auto p2_switch_in = PKMN::result(PKMN::Result::None, PKMN::Choice::Pass, PKMN::Choice::Switch);
```

The `PKMN::score` function takes a `pkmn_result` and returns the 1-sum terminal value of player 1. In other words it returns 0.0 for a p1 loss, 1.0 for a p1 win, and 0.5 otherwise. `PKMN::score2` does the same but it returns an integral value equal to 2 times the former (e.g. a value of 1 for a draw.)
If the `pkmn_result` is not terminal or it reports an error, the `score` function will `assert(false)`.

The following is a "clone" (PRNG bevahior diverges) of libpkmn's `example.c` (TODO hyperlink). The reader should compare the readability of the two programs.

```cpp
#include <libpkmn/pkmn.h>

#include <random>

const auto sample(const auto &v) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, v.size() - 1);
  return v[dist(gen)];
}

int main(int argc, char **argv) {

  using enum PKMN::Data::Move;
  using PKMN::Set;
  using PKMN::Team;
  using PKMN::Data::Species;

  const auto p1 = Team{
      Set{Species::Bulbasaur, {SleepPowder, SwordsDance, RazorLeaf, BodySlam}},
      Set{Species::Charmander, {FireBlast, FireSpin, Slash, Counter}},
      Set{Species::Squirtle, {Surf, Blizzard, BodySlam, Rest}},
      Set{Species::Pikachu, {Thunderbolt, ThunderWave, Surf, SeismicToss}},
      Set{Species::Rattata, {SuperFang, BodySlam, Blizzard, Thunderbolt}},
      Set{Species::Pidgey, {DoubleEdge, QuickAttack, WingAttack, MirrorMove}},
  };

  const auto p2 = Team{
      Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
      Set{Species::Chansey, {Reflect, SeismicToss, SoftBoiled, ThunderWave}},
      Set{Species::Snorlax, {BodySlam, Reflect, Rest, IceBeam}},
      Set{Species::Exeggutor, {SleepPowder, Psychic, Explosion, DoubleEdge}},
      Set{Species::Starmie, {Recover, ThunderWave, Blizzard, Thunderbolt}},
      Set{Species::Alakazam, {Psychic, SeismicToss, ThunderWave, Recover}},
  };

  auto battle = PKMN::battle(p1, p2);

  std::vector<uint8_t> buffer;
  buffer.resize(PKMN_LOGS_SIZE);
  auto options = PKMN::options(PKMN::log_options(buffer));
  // if log is not enabled
  // auto options = PKMN::options();
  auto result = PKMN::update(battle, 0, 0, options);
  while (!PKMN::result_type(result)) {
    const auto [p1_choices, p2_choices] = PKMN::choices(battle, result);
    // automatically resets options prior to update
    result =
        PKMN::update(battle, sample(p1_choices), sample(p2_choices), options);
  }

  const auto turns = PKMN::view(battle).turn;

  switch (PKMN::result_type(result)) {
  case PKMN_RESULT_WIN: {
    std::cout << "Battle won by Player A after " << turns << " turns\n";
    break;
  }
  case PKMN_RESULT_LOSE: {
    std::cout << "Battle won by Player B after " << turns << " turns\n";
    break;
  }
  case PKMN_RESULT_TIE: {
    std::cout << "Battle ended in a tie after " << turns << " turns\n";
    break;
  }
  case PKMN_RESULT_ERROR: {
    std::cout << "Battle encountered an error after " << turns << " turns\n";
    break;
  }
  default:
    assert(false);
  }
}
```

# Teams

No functions here, only the current Smogon sample teams with some legacy benchmark teams.

Users cannot add teams this way since these headers are hard coded at compile time. Instead they are intended to add them via a text file:

```bash
user@pc:~$ head -3 my-teams.txt # print first 3 lines
jynx blizzard lovelykiss psychic rest; chansey icebeam sing softboiled thunderbolt; cloyster blizzard clamp explosion hyperbeam; rhydon bodyslam earthquake rockslide substitute; starmie blizzard recover thunderbolt thunderwave; tauros blizzard bodyslam earthquake hyperbeam
alakazam psychic recover seismictoss thunderwave; chansey reflect seismictoss softboiled thunderwave; exeggutor explosion psychic sleeppowder stunspore; lapras blizzard hyperbeam sing thunderbolt; snorlax bodyslam earthquake hyperbeam selfdestruct; tauros blizzard bodyslam earthquake hyperbeam
alakazam psychic recover seismictoss thunderwave; chansey counter icebeam softboiled thunderwave; exeggutor explosion psychic sleeppowder stunspore; lapras blizzard hyperbeam sing thunderbolt; snorlax bodyslam earthquake hyperbeam selfdestruct; tauros blizzard bodyslam earthquake hyperbeam
```

# Search

This project supports an 'Information Set Monte Carlo Tree Search' approach where imperfect information is handled in a two step process

* Determinization

Here the private information (the enemy players species and moves) is filled in. Ideally, the entire team is 



* Perfect Info MCTS

The previous step effectively converts the game into perfect information. Now we can use MCTS as our lookahead method over something much more complicated and expensive like Counterfactual Regret Minimization.

In normal (alternating move) MCTS, each node stores data for a bandit algorithm, typically UCB/PUCB. This data is used to choose a policy for the forward phase of the iteration and then updated with the leaf-node value in the backward phase. Our approach for the simultaneous move case is termed "joint bandits": at each node we store separate bandit data for each player. Each player samples and updates their data simulateously and independently.

This is the essense of Oak's search. All the variations are basically this, they for the most part only vary the kind of bandit and leaf value estimator. The only real departure from this is the `MatrixUCB` feature which is explained here TODO.

The following is a terse primer on the theory of simutaneous move, stochastic, but otherwise perfect info games:

```
Nash, exploitability
UCB performance
TODO
```

## durations.h

This header defines the `randomize_hidden_variables` function which must be executed at the start of each MCTS iteration. It randomizes the party pokemons sleep turns and the wrap/confusion/disable/thrash durations the active pokemon.

The values stored in the `pkmn_gen1_chance_durations` struct require some explanation, and they disagree with the Pokemon Showdown Client regarding sleep (other conditions are not tracked by the client.)

```
TODO 
```




# search/bandit/

* ucb

This is the standard Upper Confidence Bound algorithm (aka Upper Confidence Trees when appiled to tree search). This algorithm is the standard for MCTS for good reason. It is fast and effective even in contexts (adversarial) where it loses theoretical guarantees.

There are various definitions for the exploration term. Oak uses 

$ u_i = \frac{c}{k} \frac{\sqrt{N}}{n_i}$.

Here $k$ is the number of actions, $c$ is the standard UCB parameterg, $n_i$ is the number of visits for action $i$, and $N$ is the total number of visits for all actions. Other formulizations of the exploration term are significantly more expensive because they also call `std::log`.

Each of the $n_i$ is initialized to 1 for simplicity. Also note that `c` is divided by $k$. This has the effect of making the vanially UCB algorithm equivalent to $PUCB$ (below) when the policy inference is unifrom.

* exp3


Each of the bandit algorithms above is a *namespace* that defines a `Bandit` struct. The `Bandit` stores


## joint.h


## mcts.h

The MCTS code in Oak is difficult to parse at first because of its heavy use of compile time logic.

However, all variations are essentially the standard recursive formulation of mcts:

```cpp
float run_iteration(auto& node, auto &state) {
  if (node.is_expanded()) {
    // select and commit moves for both players
    auto obs = state.transition(p1_choice, p2_choice);
    auto& child = node.get_child(obs);
    // get corresponding child node
    auto value = run_iteration(child, state);
    // update node stats using leaf value
    return value;
  } else {
    if (state.is_terminal()) {
      return state.value();
    }
    // if state is not terminal, exapand node
    // and get value estimate from rollout/network inference/etc
    auto value = get_estimate();
    return value;
  }
}
```

During this forward the phase the state will be mutated with no way to return to the state it was originally at the root node. For this reason we must copy the state at the start of each iteration and perform the iteration on the copy instead.


### MatrixUCB

```cpp
template <typename BanditParams> struct MatrixUCBParams {
  BanditParams bandit_params;
  uint32_t delay;
  uint32_t interval;
  float c;
};
```

MatrixUCB is an alternative to the joint bandit approach that utilizes the matrix structure to achieve theoretically superior results. It reduces to normal UCB in the case that one player has only 1 action.

> Matrix games with bandit feedback. O'Donoghue, B., Lattimore, T. and Osband, I. (2020) Conference on Uncertainty in Artificial Intelligence.

Despite the theoretical advantages, it is not feasible to use MatrixUCB at each node in the tree because of the high cost of computing Nash equilibrium.

Instead, we implement a compromise where (when enabled), MatrixUCB is only performed at the root (handled by `run_root_iteration`) and subsequent nodes are searched with the normal `run_iteration`.

The `MatrixUCBParams` wraps normal joint bandit params and stores additional params for the root. Even with MatrixUCB enabled, the root will use normal joint bandit search until `output.iterations >= delay`, and while it is active Nash equilbrium will only be solved when `output.iterations % interval == 0` and the cached solution used otherwise.

### State

```cpp
struct Input {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
};
```

The so-called "state" refers of course to the libpkmn battle struct but also to the durations. This is explained here TODO.

The `pkmn_result` is just a single byte that tells us what is requested from either player at that point in the battle: whether they may move/switch, switch, or pass. Strictly speaking it is not necessary because that information can be deduced from the battle, but it is an argument to `pkmn_gen1_choices` and it is more effecient to store and update it (`result = pkmn_gen1_battle_update(...)`) than to construct it every update.

### Search Results

```cpp
struct Output {
  uint8_t m;
  uint8_t n;
  std::array<pkmn_choice, 9> p1_choices;
  std::array<pkmn_choice, 9> p2_choices;

  std::array<double, 9> p1_prior;
  std::array<double, 9> p2_prior;

  std::array<std::array<size_t, 9>, 9> visit_matrix;
  std::array<std::array<double, 9>, 9> value_matrix;
  double total_value;

  double empirical_value;
  double nash_value;
  std::array<double, 9> p1_empirical;
  std::array<double, 9> p2_empirical;
  std::array<double, 9> p1_nash;
  std::array<double, 9> p2_nash;

  size_t iterations;
  std::chrono::milliseconds duration;
};
```

All variations of the MCTS produce the exact same kind of output. `Output` gives the empirical value of the state and the empirical policies (which converge to Nash with adversarial bandits like Exp3) and the best move for either player. It also stores some information that is used during the search, so it is both a parameter and return type.

* `m`, `n`, `p1_choices`, `p2_choices`

`m` and `n` are the number of actions of player 1 and 2, resp. Not that all of the data in `Output` is padded to the max number of actions (9).

* `p1_prior`, `p2_priot`

If the search parameters require (i.e. network with "P" bandits) a policy inference at the root, that is stored here. Otherwise the data is zero. MCTS does not require value estimation at the root so that is omitted.

* `visit_matrix`, `value_matrix`

The `value_matrix` stores *cumulative* values, so that empirical scores are calculated like

`visit_matrix[i][j] > 0 ? value_matrix[i][j] / visit_matrix[i][j] : 0.5`



* `iterations`, `duration`

The upcoming `run()` function signature allows searches to be stopped and resumed easily.

```cpp
const auto output_1 = search.run(...);
const auto output_2 = search.run(..., output_1);
```

Here `output_2` will include the iterations and cumulative data of `output_1`.

### run()

The main mcts function template has the signature

```cpp
  Output run(auto &device, const auto budget, const auto &params, auto &heap,
             auto &model, const Input &input, Output output = {})
```



#### Budget

There are 3 modes for the search budget

* iteration count

The search will run until this many iterations have been performed

* `std::chrono` duration

The search will run for at least this length of time

* pointer to `bool`-like

This acts like an run/halt flag. The search will run until the value of the flag is observed once as `false`. 

#### Model

There are 3 kinds of value (and policy) estimators

* `MonteCarlo`

The struct holds no data and simply specifies that monte carlo rollouts should be used at leaf nodes. It is an unbiased value estimator but generally it produces very weak search. It can be competent in positions where the optimal strategy is to attack and the game will end soon. However it is hopeless in a full OU 6v6,

* `PokeEngine`

This is a more or less exact copy of PokeEngine's, the perfect info MCTS component of `FoulPlay`, gen 1 eval. Loosely speaking the eval more or less sums the HP. In the kwargs, this eval is referred to as "fp".

PokeEngine does not try to estimate the winrate. Instead it computes score, a raw sum of pokemon alive, status modiefiers, etc, then it subtracts the score at the root node to compute an advantage. Then it uses the sigmoid of the advantage as its final value.

* `NN::Battle::Network`

The Oak battle network. It always provides a value but may also return policy inference in the form of logits. Whether it does so is determined by the bandit algorithm used.

#### Params

The parameters ("c" for UCB, "lr" and "exploration" for Exp3) of the bandit algorithms.

Each of these structs (e.g. `UCB::Params`, `PExp3::Params`) may be wrapped in a `MatrixUCB<typename T>` template that specifies that the MatrixUCB algorithm should be used at the root node. More on that below

#### Heap

The type of data structure used to store the search stats

* Node

* Table


## hash.h

## poke-engine-evaluate.h

# format/

This directory contains

1. Team-building specific data and code for determining what changes can be made to an incomplete team

2. A clone of Pokemon-Showdown's random-battles team generation

As team building is still a mostly unproven feature, it is expected that most users won't use it and instead supply their own team pools for self play data generation. This section can be skipped in that case.

This acts almost like the Pokemon Showdown team validator. The only difference is that the validators logic is typcically much more complicated because, generally, a legal moveset is not equivalent to a choice of up to 4 legal moves. In most formats there are combinations of moves on a species that are not legal, typically because of breeding incompatilibty.

This is not the case in RBY because there is no breeding. Nevertheless, there are a few illegal combinations of moves.

⚠️ These de facto illegal move combinations are completely legal in Oak ⚠️

This is not a priority to fix because the illegal combinations are not competitively relevent in OU. More information can be found here TODO smogon article about this.

***This code is not used for validation and user provided teams are never checked for legality.***

Only the team-building networks use this code

# encode/

Everything related to converting `libpkmn` battle and durations into tensors for Eigen/Torch lives here.

This folder and `nn` are both mostly divided into `battle/` and `build/` subdirectories. These pertain to the problems of battling and team-building, and they are treated independetly. The learning for team-building is done using values generated by network self-play, but the connection ends there.

Most effort has gone towards battling and it is probably an easier problem than team-building (and also possibly a prerequisite.) As a result, the battle network architecture is domain-specific and optimized. In contrast, the build networks are basic two layer MLPs that take a simple one-hot representation of an (in)complete team.

The battle network is Oak's reason for being while the build network is mostly a proof of concept and a principled way to add variety to the team pool during data generation.

# encode/battle

## battle.h

This file principally defines 

* `Encode::Battle::Pokemon::write(..., float *)`

* `Encode::Battle::ActivePokemon::write(..., float *)`

which write the encoding onto the underlying array of the tensor. The tensor is assumed to be zero'd prior to writing - it does not reset previous writes.

The dimensions of the two encodings are determined at compile-time and are have no adjustable parameters.

## policy.h

This defines functions that map a players actions (meaning their `pkmn_choice`s and `PKMN::Side` data) onto the policy network output. The policy dimesions are basically a union of the `PKMN::Data::Move` and `PKMN::Data::Species` enums, minus both enums `None` entry and `Move::Struggle`.

Each `pkmn_choice` corresponds to either a move, switch, or pass. The first two cases map to the policy output in the obvious way. A pass is only possible when the side only has one action (and therefor the logit value does not matter), so it maps to dimension 0, and its value is not used during the bandit selection. 

# encode/build

TODO

# nn/

Both battling and team-building networks share some code.

## affine.h

`Affine<>` defines a basic linear layer. It has dynamic size which allows Oak networks to have adjustable hyperparameters. The template bools determine if and what non-linearity is applied to the output. The class only has a default constructor. The weights and bias data is resized when `Affine::read_paramaters(std::ifstream& )` is called, not when the object is constructed.

The stream first reads a two `u32`s for the in and out dimensions. Then it attempts to read the biases then the weights as a stream of `floats`. Both the read and write functions return a bool indicated if the operation was successful.

## embedding-net.h

An embedding net is a two layer MLP with ReLu activation by default on both layer.

`template <bool relu_0 = true, bool relu_1 = true> struct EmbeddingNet;`

Also stores a buffer that holds the output of the first layer. This buffer is resized when parameters are read.

The default parameters are used for the Pokemon and ActivePokemon embedding nets.

The team-building network is just an actor/critic network pair, each of which is an `EmbeddingNet<true, false>` and the final output is used for policy logits or pre-sigmoid value output.

## default-hyperparameters

The final `NN::Battle::Network` only has two hyper-parameters which are fixed, the input dims to the Pokemon and ActivePokemon embedding nets.

Therefore these values just the defaults used in the release pre-trained networks. These value are used by the 


# nn/battle/

# nn/build/

