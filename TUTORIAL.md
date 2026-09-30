# Installation and Activation

Oak is available on the python package index under [oak-lab](https://pypi.org/project/oak-lab/).

It is recommended that you install Oak in a virtual environment:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install oak-lab
```

If the installation fails, it is likely that you are using either an unsupported Python version/OS/CPU architecture. Currently, **Oak is only available for Python versions 3.10 - 3.14 on the Linux operating system with x86-64 architecture.** Windows users can use [WSL](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux).

To install from a source build (see README), run `pip install -e .` in the repo root after `dev/libpkmn` and `dev/release`. Rebuilt binaries are then picked up without reinstalling.

Setting the environment variable `OAK_DEBUG` to anything other than `0` makes both the binaries and the Python modules use the Debug build (requires `dev/debug`).

The installation can quickly be checked with the `benchmark` program, which will run ~1M iterations of pure MCTS on turn 1 of a 6v6 game

```bash
(.venv) $ benchmark
thread 0
12811ms. 1048576 iterations.
```

Below 10ms the time is printed in microseconds (`µs`) instead.

While a Oak-installed virtual environment is active, the following binaries will be available from command line:

* `benchmark`
* `oak-search-test`
* `generate`
* `vs`

`chall` is also installed but is defunct (its binary is no longer built).

The Python scripts are not installed as commands; run them as modules:

* `python -m oak.scripts.lab`
* `python -m oak.scripts.battle`
* `python -m oak.scripts.build`
* `python -m oak.scripts.rl`

The usage of all these programs will be covered in this tutorial.

Oak is also a traditional Python library:

```python
>>> import oak, oak.search
>>> battle, durations, result = oak.parse_battle("snorlax bodyslam rest | starmie psychic thunderwave recover")
>>> output = oak.search.run(battle, durations,
...     oak.search.parse_budget("20000"),
...     oak.search.parse_bandit("ucb-1.0"),
...     oak.search.Node(),
...     oak.search.parse_eval("fp", False))
>>> print(oak.search.output_string(battle, durations, result, output))
Iterations: 20000, Time: 22.87 ms
Value: 0.529

Player 1:
   BodySlam  Rest      
e: 0.972     0.029     
n: 1.000     0.000     
p: 0.000     0.000     
Player 2:
   Psychic   ThunderWave  Recover   
e: 0.215     0.766     0.019     
n: 0.000     1.000     0.000     
p: 0.000     0.000     0.000     

EV Matrix:
         Psychic  Thunder  Recover  
BodySla  0.547    0.525    0.644    
Rest     0.339    0.433    0.531    

Visits:
         Psychic  Thunder  Recover  
BodySla  4156     14897    377      
Rest     137      426      7        
```

# Training a Battle Network

The example above was chosen to illustrate that Oak is capable of replacing the search used in IS-MCTS projects like [Foul Play](https://github.com/pmariglia/foul-play). But the evaluation used in the example is PokeEngine, not a (much stronger) trained Oak battle network. Let's begin with data generation and network training.

The general plan:

* Fast self-play using `generate`

* Train value/policy network using `python -m oak.scripts.battle`

* Compare strength (relative to FoulPlay and Monte-Carlo) using `vs`

## Data generation

The `generate` program accepts many keyword arguments but only a few are required: `--budget`, `--bandit`, `--eval` and `--policy-mode`. They reflect basic considerations that we will discuss now:

* `--eval=fp`

This is value estimator that the search will use in self-play games. To start with, we only have PokeEngine evaluation ("fp") and Monte-Carlo ("mc").

Despite being a simple hand crafted score function, "fp" is much stronger and faster than Monte-Carlo.

```
(.venv) $ vs --budget=4096 --bandit=ucb-1.0 --policy-mode=x --p1-eval=fp --p2-eval=mc
...
W D L:
186 1 31
```

Above we've used the `vs` command with sensible arguments to compare the strength of these two evals, and we can see "fp" scored 186 wins to "mc"s 31.

Therefore we will use the "fp" eval function our first self-play data generation run.

* `--budget=4096`

2^12 is a reasonable iteration count for a few reasons. AlphaZero used 1000 ~ 2^10 iterations, and the branching factor for a simultaneous move game is the product of the number of actions for either player. Its totally possible RBY has a higher average branching factor after including RNG. Therefore we probably should use at least as many iterations as alternating-move configs.

```bash
(.venv) $ benchmark --eval=fp --budget=4096
thread 0
3548µs. 4096 iterations.
```

On my machine this gives us 3.5 milliseconds per step.

Budgets are either an iteration count (`4096`) or a time (`100ms`, `8s`).

* `--bandit=ucb-1.0`

The bandit algorithms available are:

* `ucb`
* `pucb`
* `exp3`
* `pexp3`
* `rm` (regret matching)

Each of these has a float parameter that comes afterwards separated by a '-', e.g. `ucb-1.0`. For the 'ucb' variants this is the exploration weight "c" and for 'exp3' variants it is the update weight "lr". The exp3 variants have a second optional parameter which is the weight of the uniform policy noise in the forecast (default 0.05) e.g. `pexp3-1.0-0.1`. `pexp3` takes a third optional parameter, the prior temperature.

`pucb` and `pexp3` use the network prior, so they require a network eval ("Contextual bandits must use network eval").

`ucb1` (the Foul Play clone) is compiled out of the current build (`#define NO_UCB1` in `cpp/src/search.cc`) and fails with "Invalid bandit".

Currently all evidence points to ucb being the strongest variant, despite exp3's [theoretical guarantees](https://arxiv.org/abs/1804.09045). It is probably also better suited towards low iteration searches.

* `--policy-mode=`

The search will produce multiple strategies or policies for either player. These are:

* `e` Empirical
* `x` Argmax
* `n` Nash equilibrium of empirical value matrix
* `p` Network prior
* `u` Uniform noise

Weighted combinations are separated by '-', e.g. `x0.9-e0.1`. Without the separator (`x0.9e0.1`) the whole string is read as a single `x` term with weight `0.9e0.1`, i.e. pure argmax.

Caveats:

* `p` is all zeros without a network eval, so it fails with "zero policy".
* Any mode containing `u` currently crashes `vs`.
* The root Nash solve treats unvisited action pairs as value 0 (a loss for P1). At low budgets this skews `n` badly: identical `fp` agents at 64 iterations score ~86% for P1.

The following arguments are optional but important:

* `--teams=`

Any time teams are required Oak will default to the 16 Smogon sample teams. More teams is certainly beneficial however. The programs expect a simple plaintext format that can be explained hopefully with just an example (2 lines/teams):

```
jynx blizzard lovelykiss psychic rest; chansey counter icebeam softboiled thunderbolt; jolteon doublekick rest thunderbolt thunderwave; snorlax bodyslam icebeam reflect rest; starmie blizzard psychic recover thunderwave; tauros blizzard bodyslam earthquake hyperbeam
gengar confuseray explosion hypnosis thunderbolt; chansey icebeam softboiled thunderbolt thunderwave; cloyster blizzard clamp explosion rest; exeggutor explosion psychic rest sleeppowder; golem earthquake explosion rest rockslide; slowbro amnesia rest surf thunderwave
```

* `--fast-search-prob=`
* `--fast-budget=`

With this, the full search budget is only used for some of the steps, otherwise we use a reduced budget. This is an innovation of [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565); It reduces cost of data gen while balancing value and policy learning.

* `--threads=`

By default this program will use the max number of threads minus one.

* `--dir=fp-data`

The name of the directory where all work will be saved. If this is not provided, the programs will all use a data-time string with the program name as the prefix. It is advised to use short code names.

```bash
(.venv) $ generate --budget=1024 --bandit=ucb-1.0 --policy-mode=x --eval=fp --dir=fp-data
Created directory fp-data
279.767 battle frames/sec.
keep node ratio: 0
progress: 0.000781659%
Game Lengths: 27 33 32 20 2 2 1 
```

You should see something like this. A line confirming the work directory was created and some periodic 

After some time has passed we enter `Ctrl + C` to send a SIGINT signal. This terminates the program and triggers a save; all the Oak programs save their arguments to disk:

```bash
(.venv) $ ls fp-data
0.battle.data  2.battle.data  4.battle.data  6.battle.data  matchup-matrix
1.battle.data  3.battle.data  5.battle.data  args
(.venv) $ head -10 fp-data/args
   --team-modify-prob : 0
--pokemon-delete-prob : 0
   --move-delete-prob : 0
 --build-network-path : 
        --max-pokemon : 6
      --budget : 1024
             --bandit : ucb-1.0
         --matrix-ucb : 
               --eval : fp
```

## Training

Let's start training by running a quick check with `lab`. This script is intended to be a multi-utility for RL. It is also written entirely with torch and the `oak` Python library so it serves as a good example for adding new functionality.

The `battle-frame-stats` utility will recursively scan the provided `--data-path` for all files with the `.battle.data` extension. It then parses them to print statistics.

```bash
(.venv) $ python -m oak.scripts.lab battle-frame-stats --data-path=fp-data
Found 7 data files
Total battle frames: 19373207
Average battle length: 80.1695282078021
Iteration counts:
1024: 19373207
```

### Architecture

We should first discuss the Oak network design, from battle encoding to value and policy outputs.


The information of a generation 1 battle can be split into two sides (no field conditions exist yet), and each side can be partitioned (save for `last_damage` and `last_selected`, which we ignore) into five `Pokemon` and `ActivePokemon`. The latter is the combination of the underlying `Pokemon` data for that slot and the active pokemon information (volatiles, stats, etc.)

We use distinct 2-layer MLPs (`pokemon_net`, `active_net`, `moves_net`) to encode each side. The data is encoded in a mostly one-hot format, with continuous fields (e.g. `Stats`, HP percentage) normalized to [0, 1].

The outputs of these networks, called 'embeddings', are concatenated into two side embeddings (`oak.train.side_out_dim` = 384 each) and these are concatenated into the battle embedding. This final embedding is what serves as the input into an AlphaZero style value/policy trunk. Using the default sizes for clarity, these are a shared trunk:

> 768 x 64 -> 64 x 64

The output of the trunk is the shared input for the value head:

> 64 x 32 - > 32 x 1

and the two policy heads, one for each player:

> 64 x 64 -> 64 x 315

In the policy output, we mask for legal moves. The logits (`oak.train.policy_out_dim` = 315) correspond to switching to a species or using a move. Recharge/Struggle do not have logits, which is fine because they are only selectable when there is only 1 legal action for that player.

### Quantization

Any program that uses a network has the option (`--quantize`) to use an `i8` quantized version of that network's value/policy trunk. This process introduce some error to the output, but the increase in speed is typically enough to result in a stronger search.

*Warning*: Not all networks are quantizable. Networks trained with `--discrete` are. Floating point versions of networks (Python and C++) have total liberty with regards to hyper-parameters. However, the quantized trunks are only available when:

1. The battle embedding dim is 768

2. `HIDDEN`, `VALUE_HIDDEN`, `POLICY_HIDDEN` are in {32, 64, 128}

3. `VALUE_HIDDEN`, `POLICY_HIDDEN` <= `HIDDEN`

### Args


All Oak scripts will list their arguments if the `--help` flag is provided:

```bash
python -m oak.scripts.battle --help
```

Some of these listed are self explanitory, so we will focus on a few

* `--data-dir=fp-data`
The battle script will look for `.battle.data` files *recursively* in the provided dir (default=".").

* `--batch-size=4096`
Pokemon scores have higher variance because of the stochastic nature of the game. Large batches should mitigate this.

* `--lr=.001`
The default for the Pytorch Adam implementation.

* `--threads=`
Unlike `generate`, this script uses only one thread by default. This kwarg limits the max number of threads that Torch/CUDA can use and the number of data reading threads.

* `--min-iterations=`
Frames from searches with fewer iterations are skipped. Set it above `--fast-budget` to train only on full searches.

* `--discrete`
Clamped activations/parameters, required for `--quantize`.

The following arguments are default and were not explicitly entered, but deserve mention anyway.

* `--value-nash-weight=0.0`
* `--value-empirical-weight=0.0`
* `--value-score-weight=1.0`
The final value target is a weighted sum of 3 different value estimates.
The empirical value is just the average leaf value that is back propagated to the root. The Nash value is the (unique) value corresponding to Nash equilibria on the root empirical value matrix (it is normally very close the the empirical value.)

Most RL setups use only the score as we have. Additionally, using the PokeEngine eval we used for self play is not intended to be a value estimator. This means that the empirical and Nash values are less meaningful than if we used Monte-Carlo or a Network.

* `--policy-nash-weight=0.0`
There are two targets for the policy learning, the empirical and Nash strategies; the empirical weight is 1 minus this. The Nash stratagies produced by UCB bandit varaints tend to be low quality since these algorithms tend to leave some move pairs very unexplored. This results in a high-variance estimate of in that entry of the empirical value matrix.

* `--pokemon-hidden-dim=256`
* `--active-hidden-dim=256`
* `--moves-hidden-dim=256`
* `--pokemon-out-dim=30`
* `--active-out-dim=54`
* `--moves-out-dim=25`
* `--hidden-dim=64`
* `--value-hidden-dim=32`
* `--policy-hidden-dim=64` 

There are the default hyperparameters. The emphasis is on speed and minimizing the number of FLOPs per inference.

### Run

```bash
(.venv) $ python -m oak.scripts.battle --dir=first-net --data-dir=fp-data --batch-size=4096 --lr=.001 --threads=8
Using device: cpu
Saved initial network in output directory.
Initial network hash: 12608495754081121817
tensor([[0.5209, 1.0000, 0.8011, 0.7875, 1.0000],
        [0.5206, 0.0000, 0.4727, 0.4726, 0.0000],
        [0.5208, 0.0000, 0.4635, 0.4803, 0.0000],
        [0.5214, 1.0000, 0.5804, 0.3524, 1.0000],
        [0.5209, 1.0000, 0.5161, 0.6316, 1.0000]], grad_fn=<CatBackward0>)
P1 policy inference/target
...
loss: p1:0.25835880637168884, p2:0.26018473505973816
loss: v:0.2493373304605484
```

The program prints compare targets/predictions and display loss values. They are likely to change and won't be discussed further.

A network is saved every `--checkpoint` (default 50) steps as `{step}.battle.net`, alongside a `.train_state` file with the optimizer state.

We allow the training to go for 1000 steps. In this training regime (non-Network eval, low iteration, small data set) the networks seems to plateau after a few hundred steps with `lr=.001`.


## Evaluation

The `vs` program requires the following information for both players:

* `budget`
* `eval`
* `bandit`
* `policy-mode`

These information can be entered with no prefix so that it applies to both players (e.g. `--budget=8s`)
or with the prefix `p1-`/`p2-` (e.g. `--p1-eval=fp`.)
A prefixed argument will override a non-prefixed argument.

Notes:

* `--max-games` counts matches, not games. Each match plays two games with the teams swapped (one with `--mirror-match`).
* W/D/L is from P1's perspective.
* `--threads` defaults to all hardware threads. With time budgets, concurrent searches compete for CPU.
* With iteration budgets, `--seed` makes runs reproducible.

Lets first compare the trained network with the PokeEngine eval using a think time of 1 second.
This first test does not use the networks policy inference since is using the same bandit as PokeEngine (for initial comparison's sake.)

The examples below used `ucb1`, which is compiled out of the current build; substitute `ucb`.

```bash
(.venv) $ vs --budget=1000ms --p1-eval=apple/500.battle.net --p2-eval=fp --bandit=ucb1-2.0 --policy-mode=x --threads=8 --mirror-match
```

Every `--print-interval` seconds (default 15) the program prints the score, W/D/L, per-player search stats (iterations and duration mean/stdev), and a per-thread line of the format

```
  {thread}: {updates}, ({p1_output.empirical_value}/{p1_output.nash_value} {p1_output.iterations}), ({p2_output.empirical_value}/{p2_output.nash_value} {p2_output.iterations})
```

On exit it prints

```
score: 0.45 over 80 games.
W D L:
36 0 44
```

### Args

The Pokemon-Showdown timer affords a 10 second increment, so that is probably the best search budget to use. This however makes getting (low variance) results agonizingly slow, so we start with 1 second think time. Skill disparities seem to grow with more time.

The `ucb1` bandit is clone of FoulPlay's. It does not use policy priors since the hand crafted eval does not provide them.

### Results

```bash
score: 0.111111 over 9 games; Elo diff: -361.236
1 0 8
^Cscore: 0.2 over 10 games.
2 0 8
```

10 games is a *very* small sample size but the picture is still clear: our network is outmatched with these settings. However this can easily be explained and fixed.

1. Using the exact same bandit means the network cannot use its trained policy inference

2. The network is slower than the simpler eval

Indeed, the `benchmark` tool shows that the network is about 3x slower:

```bash
(.venv) $ benchmark --eval=fp --budget=1000ms
thread 0
1000ms. 253551 iterations.
(.venv) $ benchmark --eval=apple/500.battle.net --budget=1000ms
thread 0
1000ms. 85005 iterations.
```

The speed penalty could be greatly mitigated if it was allowed to use policy inference. Let's try that:

```bash
(.venv) $ vs --budget=1000ms --p1-eval=apple/500.battle.net --p2-eval=fp --p1-bandit=pucb-1.0 --bandit=ucb1-2.0 --policy-mode=x --threads=8 --mirror-match
# ...
score: 0.639344 over 61 games; Elo diff: 99.4568
39 0 22
^Cscore: 0.639344 over 61 games.
```

With this change, the network is now 2:1 vs 'fp'.

# Python Scripting

The primary goal of this program is to train a strong network for IS-MCTS like Foul-Play. We claimed that the Oak Python API is sufficient to reproduce all of the C++ binaries.

### Search Objects

A search takes a battle, its durations, and four objects from `oak.search`:

* Budget: `parse_budget("4096")`, `parse_budget("8s")`, or `Iterations(n)` / `Duration(ms)`
* Bandit: `parse_bandit("ucb-1.0")`, or `UCB(c)`, `PUCB(c)`, `Exp3(lr, exploration)`, `PExp3(lr, exploration, temp)`, `MatrixUCB(bandit, c, interval, grow)`
* Heap: `Node()`, or `MatrixUCBNode()` for `MatrixUCB`. `Table()` (transposition table) is compiled out of the current build.
* Eval: `parse_eval(string, quantize)`, or `MonteCarlo()`, `PokeEngine()`, `Network()`

```python
import oak, oak.search as s
battle, durations, result = oak.parse_battle("snorlax bodyslam rest | starmie psychic thunderwave recover")
output = s.run(battle, durations, s.parse_budget("8s"), s.parse_bandit("ucb-1.0"), s.Node(), s.parse_eval("mc", False))
policy = s.get_policy_from_side(output.p1, "x")   # same mode strings as --policy-mode
```

The Heap is almost totally opaque. It is meant to be created, passed into the search function, and destroyed.

`parse_battle` takes a battle string (`|` separates the sides, `;` the pokemon) and an optional seed, and returns `(battle, durations, result)`. `oak.choices(battle, result)` gives the legal choices for both players and `oak.update(battle, durations, c1, c2)` advances the battle in place.

`output` exposes `iterations`, `empirical_value`, `nash_value`, `initial_value`, `visit_matrix`, `value_matrix`, `empirical_matrix` (unvisited entries = 0.5) and per-player `p1`/`p2` (`k`, `choices`, `empirical`, `nash`, `prior`, `logit`).

Caveats:

* `output.duration` raises a `TypeError` (chrono type not registered); time the call yourself.
* `run` seeds its RNG from `std::random_device`, so Python searches are not reproducible.

`src/oak/__init__.pyi` is out of date; `help()` on the live modules is authoritative.

# Training a Team-Building Network

The team building network architechture and infrastructure is much simpler than that for battling.

TODO

# RL

The `rl` program is very simple. It runs `generate` and the training scripts `battle`/`build` at the same time but with the `--in-place` flag for the learners.

This means that, in addition to saving the updated network parameters in the usual way (i.e. "working-dir/step.battle.net"), it will save the latest parameters the path that `generate` reads from. Each self-play worker reads the parameters again at the start of each battle

`rl` launches `generate` from `PATH`, so run it with the Oak environment active. The battle learner is given `--min-iterations=fast-budget+1`, so it only trains on full searches. `--discrete` also passes `--quantize` to `generate`.

```bash
(.venv) $ python -m oak.scripts.rl --budget=2048 --fast-budget=128 --fast-search-prob=.935 --bandit=pucb-0.25 --policy-mode=e0.9-x0.1 --fast-policy-mode=x --batch-size=2048 --lr=.0001 --build-batch-size=0 --build-lr=0 --build-trajectories-per-step=0 --build-keep-prob=0 --sleep=4 --data-window=16 --delete-window=32
```

The above is an example of a fast run with no team-building. The latter only takes place when `--team-modify-prob` is non-zero, among other conditions. The lack of `--discrete` flag means the net will use ReLU activations and won't be quantizable. The battle learner will use only the most recently generated files, set by `--data-window`. Any file outside of `--delete-window` will be automatically deleted (so always set it larget than data window.)

`--sleep` sets a mandatory wait period (in seconds) between each step of the `battle` learner. This is because RL is almost always bottle-necked by the speed of data generation. We slow the learner down so it isn't seeing the same data all the time.

```bash
(.venv) $ python -m oak.scripts.rl --budget=1024 --fast-budget=256 --fast-search-prob=.75 --bandit=pexp3-1.0-0.1 --policy-mode=n --fast-policy-mode=x --batch-size=8192 --lr=.001 --lr-decay=.99 --lr-decay-interval=100 --build-batch-size=2048 --build-lr=.01 --build-trajectories-per-step=2048 --build-keep-prob=.5 --max-pokemon=1 --team-modify-prob=1 --pokemon-delete-prob=1 --sleep=4
```

Proof of concept `rl` run with team-building but only for 1v1.

### Args

The arguments for `rl` contain the arguments for both `battle` and `build`, where the latter's arguments are prefixed with "build-" for disambiguation. For example

* `--lr` sets the learning rate build `battle`

* `--build-lr` sets the learning rate for `build`

`--budget`/`--fast-budget` are iteration counts only (no time budgets). `--generate-path`, `--battle-path` and `--build-path` are accepted but unused.
