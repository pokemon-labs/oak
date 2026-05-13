# Installation and Activation

Oak is available on the python package index under [oak-lab](https://pypi.org/project/oak-lab/).

It is recommended that you install Oak in a virtual environment:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install oak-lab
```

If the installation fails, it is likely that you are using either an unsupported Python version/OS/CPU architecture. Currently, **Oak is only available for Python versions 3.10 - 3.14 on the Linux operating system with x86-64 architecture.** Windows users can use [WSL](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux).

The installation can quickly be checked with the `benchmark` program, which will run ~1M iterations of pure MCTS on turn 1 of a 6v6 game

```bash
(.venv) $ benchmark
12811ms.
1048576 iterations.
```

While a Oak-installed virtual environment is active, the following binaries will be available from command line:

* `benchmark`
* `oak-search-test`
* `generate`
* `chall`
* `vs`

 and the following Python scripts:

* `lab`
* `battle`
* `build`
* `evo`

The usage of all these programs will be covered in this tutorial.

Oak is also a traditional Python library:

```bash
(.venv) $ python
Python 3.13.3 (main, Jan  8 2026, 12:03:54) [GCC 14.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import oak
>>> input = oak.parse_battle("snorlax bodyslam rest | starmie psychic thunderwave recover")
>>> agent = oak.Agent()
>>> agent.budget = "8s"
>>> agent.bandit = "ucb-1.0"
>>> heap = oak.Heap()
>>> output = oak.search(input, heap, agent)
>>> print(oak.format(input, output))
Iterations: 847615, Time: 8000.02 sec
Value: 0.448

P1
BodySlam  Rest      
0.989     0.011     
1.000     0.000     
P2
Psychic   ThunderWave  Recover   
0.028     0.970     0.001     
0.000     1.000     0.000     

Matrix:
         Psychic  Thunder  Recover  
BodySla  0.602    0.443    0.767    
Rest     0.395    0.398    0.485    

Visits:
         Psychic  Thunder  Recover  
BodySla  23811    813826   943      
Rest     329      8673     33       
```

# Training a Battle Network

The example above was chosen to illustrate that Oak is capable of replacing the search used in IS-MCTS projects like [Foul Play](https://github.com/pmariglia/foul-play). But the evaluation used in the example is Monte Carlo, not a (much stronger) trained Oak battle network. Let's begin with data generation and network training.

The general plan:

* Fast self-play using `generate`

* Train value/policy network using `battle`

* Compare strength (relative to FoulPlay and Monte-Carlo) using `vs`

## Data generation

The `generate` program accepts many keyword arguments but only a few are required. Those that are reflect basic considerations that we will discuss now:

* `--eval=fp`

This is value estimator that the search will use in self-play games. To start with, we only have PokeEngine evaluation ("fp") and Monte-Carlo ("mc", default).

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
3548µs.
4096 iterations.
```

On my machine this gives us 3.5 milliseconds per step.

* `--bandit=ucb-1.0`

There are 5 bandit algorithms available:

* `ucb`
* `pucb`
* `ucb1`
* `exp3`
* `pexp3`

Each of these has a float parameter that comes afterwards separated by a '-', e.g. `ucb-1.0`. For the 'ucb' variants this is the exploration weight "c" and for 'exp3' variants it is the update weight "gamma". The exp3 variants have a second optional parameter which is the weight of the uniform policy noise in the forecast e.g. `pexp3-1.0-0.1`.

Currently all evidence points to ucb being the strongest variant, despite exp3's [theoretical guarantees](https://arxiv.org/abs/1804.09045). It is probably also better suited towards low iteration searches.

* `--policy-mode=`

The search will produce multiple strategies or policies for either player. These are:

* `e` Empirical
* `x` Argmax
* `n` Nash equilibrium of empirical value matrix
* `p` Network prior
* `u` Uniform noise

All of these characters are valid arguments. Additionally, weighted combinations may be passed e.g. `x0.9e0.1`.

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

The `battle-frame-stats` argument will recursively scan the provided `dir` for all files with the '.battle.data` extension. It then parses them to print statistics.

```bash
(.venv) $ lab battle-frame-stats --dir=fp-data
Total battle frames: 19373207
Average battle length: 80.1695282078021
(.venv) $ python
Python 3.13.3 (main, Jan  8 2026, 12:03:54) [GCC 14.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 19373207 / 80 # number of games
242165.0875
>>> 
```

### Architecture

We should first discuss the Oak network design, from battle encoding to value and policy outputs.


The information of a generation 1 battle can be split into two sides (no field conditions exist yet), and each side can be partitioned (save for `last_damage` and `last_selected`, which we ignore) into five `Pokemon` and `ActivePokemon`. The latter is the combination of the underlying `Pokemon` data for that slot and the active pokemon information (volatiles, stats, etc.)

We use two distict 2-layer MLPs to encode each sides `Pokemon` and `ActivePokemon`. The data of these two is encoded in a mostly one-hot format, with continuous fields (e.g. `Stats`, HP percentage) normalized to [0, 1].

The outputs of these networks, called 'embeddings', is concatenated into two side embeddings and these are concatenated into the battle embedding. This final embedding is what serves as the input into an AlphaZero style value/policy trunk. Using the default sizes for clarity, these are a shared trunk:

> 768 x 64 -> 64 x 64

The output of the trunk is the shared input for the value head:

> 64 x 32 - > 32 x 1

and the two policy heads, one for each player:

> 64 x 64 -> 64 x (SPECIES + MOVES)

In the policy output, we mask for legal moves. The 151 SPECIES logits correspond to switching to that pokemon, and the 165 MOVE logits correspond to using that move. Recharge/Struggle do not have logits, which is fine because they are only selectable when there is only 1 legal action for that player.

### Quantization

Any program that uses a network has the option to `i8` quantized version of that network's value/policy trunk. This process introduce some error to the output, but the increase in speed is typically enough to result in a stronger search.

*Warning*: Not all networks are quantizable. Floating point versions of networks (Python and C++) have total liberty with regards to hyper-parameters. However, the quantized trunks are only available when:

1. `BATTLE_EMBEDDING_DIM` = 10 * `POKEMON_EMBEDDING_DIM` + 2 * `ACTIVE_POKEMON_EMBEDDING_DIM` = 768

2. `HIDDEN`, `VALUE_HIDDEN`, `POLICY_HIDDEN` are in {32, 64, 128}

3. `VALUE_HIDDEN`, `POLICY_HIDDEN` <= `HIDDEN`

### Args


All Oak scripts will list their arguments if the `--help` flag is provided:

```bash
battle --help
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

The following arguments are default and were not explicitly entered, but deserve mention anyway.

* `--value_nash_weight=0.0`
* `--value_empirical_weight=0.0`
* `--value_score_weight=1.0`
The final value target is a weighted sum of 3 different value estimates.
The empirical value is just the average leaf value that is back propagated to the root. The Nash value is the (unique) value corresponding to Nash equilibria on the root empirical value matrix (it is normally very close the the empirical value.)

Most RL setups use only the score as we have. Additionally, using the PokeEngine eval we used for self play is not intended to be a value estimator. This means that the empirical and Nash values are less meaningful than if we used Monte-Carlo or a Network.

* `--p_empirical_weight=1.0`
* `--p_nash_weight=0.0`
There are two targets for the policy learning, the empirical and Nash strategies. The Nash stratagies produced by UCB bandit varaints tend to be low quality since these algorithms tend to leave some move pairs very unexplored. This results in a high-variance estimate of in that entry of the empirical value matrix.

* `--pokemon_hidden_dim=128`
* `--active_hidden_dim=128`
* `--pokemon_out_dim=59`
* `--active_out_dim=83`
* `--hidden_dim=64`
* `--value_hidden_dim=32`
* `--policy_hidden_dim=64` 

There are the default hyperparameters. The emphasis is on speed and minimizing the number of FLOPs per inference.

It 

### Run

```bash
(.venv) $ battle --dir=first-net --data-dir=fp-data --batch-size=4096 --lr=.001 --threads=8
Using device: cpu
Saved initial network in output directory.
Initial network hash: 12608495754081121817
tensor([[0.5209, 1.0000, 0.8011, 0.7875, 1.0000],
        [0.5206, 0.0000, 0.4727, 0.4726, 0.0000],
        [0.5208, 0.0000, 0.4635, 0.4803, 0.0000],
        [0.5214, 1.0000, 0.5804, 0.3524, 1.0000],
        [0.5209, 1.0000, 0.5161, 0.6316, 1.0000]], grad_fn=<CatBackward0>)
P1 policy inference/target
tensor([[[1.4237e-01, 1.3655e-01, 1.3181e-01, 1.4931e-01, 1.4837e-01,
          1.3983e-01, 1.5175e-01, 0.0000e+00, 0.0000e+00],
         [4.8676e-03, 4.1733e-02, 1.7075e-02, 3.6469e-03, 2.2889e-04,
          9.3213e-01, 2.2889e-04, 0.0000e+00, 0.0000e+00]],

        [[1.0072e-01, 1.1029e-01, 1.2581e-01, 1.0011e-01, 1.0813e-01,
          1.1339e-01, 1.0655e-01, 1.1912e-01, 1.1589e-01],
         [2.1958e-02, 2.6352e-02, 3.2456e-02, 3.0747e-02, 7.7623e-02,
          5.6626e-02, 5.8837e-01, 1.4134e-01, 2.4399e-02]],

        [[1.0464e-01, 1.1460e-01, 1.0297e-01, 1.0395e-01, 1.1237e-01,
          1.1721e-01, 1.1067e-01, 1.1333e-01, 1.2025e-01],
         [4.7303e-04, 9.5064e-03, 4.7303e-04, 9.1138e-01, 3.7095e-02,
          4.8676e-03, 1.3413e-02, 2.2202e-02, 4.7303e-04]],

        [[1.1258e-01, 1.2462e-01, 1.1549e-01, 1.2205e-01, 1.2577e-01,
          1.3473e-01, 1.3427e-01, 1.3048e-01, 0.0000e+00],
         [1.9028e-02, 5.4321e-01, 2.4399e-02, 4.7303e-04, 4.7303e-04,
          2.5864e-02, 2.1820e-03, 3.8427e-01, 0.0000e+00]],

        [[1.0664e-01, 1.1665e-01, 1.0602e-01, 1.1697e-01, 1.1450e-01,
          9.6538e-02, 9.8539e-02, 1.1822e-01, 1.2590e-01],
         [2.9145e-03, 9.6132e-04, 6.5766e-03, 9.5064e-03, 4.7303e-04,
          2.4262e-03, 1.4745e-01, 1.8296e-02, 8.1128e-01]]],
       grad_fn=<CatBackward0>)
loss: p1:0.25835880637168884, p2:0.26018473505973816
loss: v:0.2493373304605484
```

The program prints compare targets/predictions and display loss values. They are likely to change and won't be discussed further.

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

Lets first compare the trained network with the PokeEngine eval using a think time of 1 second.
This first test does not use the networks policy inference since is using the same bandit as PokeEngine (for initial comparison's sake.)

```bash
(.venv) $ vs --budget=1000ms --p1-eval=apple/500.battle.net --p2-eval=fp --bandit=ucb1-2.0 --policy-mode=x --threads=8 --mirror-match
score: -nan over 0 games; Elo diff: -nan
0 0 0
info: 
        0: 7, (0.50466/0.507031), (0.548622/0.550781)
        1: 8, (0.534188/0.550781), (0/0)
        2: 7, (0.508277/0.515625), (0.529301/0.53125)
        3: 7, (0/0), (0.438785/0.429688)
        4: 7, (0/0), (0.44198/0.433594)
        5: 7, (0.64514/0.648438), (0.400198/0.405924)
        6: 9, (0/0), (0.501222/0.488281)
        7: 7, (0.595607/0.592076), (0.52585/0.53125)
score: -nan over 0 games; Elo diff: -nan
```

The data display is of the format

```
  {thread}: {update}, ({p1_output.empirical_value}/{p1_output.nash_value}), ({p1_output.empirical_value}/{p1_output.nash_value})
```

### Args

The Pokemon-Showdown timer affords a 10 second increment, so that is probably the best search budget to use. This however makes getting (low variance) results agonizingly slow, so we start with 1 second think time. Skill disparities seem to grow with more time.

The `ucb1` bandit is clone of FoulPlay's. It does not use policy priors since the hand crafted eval does not provide them.

### Results

```bash
score: 0.111111 over 9 games; Elo diff: -361.236
1 0 8
info: 
        0: 12, (0.389936/0.394531), (0.65818/0.664062)
        1: 141, (0.935303/0.941406), (0.54506/0.546875)
        2: 4, (0.722264/0.71875), (0.524766/0.525746)
        3: 45, (0.856075/0.854868), (0.584453/0.579086)
        4: 65, (0.662711/0.662109), (0.442586/0.433594)
        5: 146, (0.93344/0.9375), (0.909666/0.910156)
        6: 142, (0.733726/0.734375), (0.574636/0.574219)
        7: 43, (0.686821/0.685948), (0.502774/0.511719)
^Cscore: 0.2 over 10 games.
2 0 8
```

10 games is a *very* small sample size but the picture is still clear: our network is outmatched with these settings. However this can easily be explained and fixed.

1. Using the exact same bandit means the network cannot use its trained policy inference

2. The network is slower than the simpler eval

Indeed, the `benchmark` tool shows that the network is about 3x slower:

```bash
(.venv) $ benchmark --eval=fp --budget=1000ms
1000001 ms.
253551 iterations.
(.venv) $ benchmark --eval=apple/500.battle.net --budget=1000ms
1000008 ms.
85005 iterations.
```

The speed penalty could be greatly mitigated if it was allowed to use policy inference. Let's try that:

```bash
(.venv) $ vs --budget=1000ms --p1-eval=apple/500.battle.net --p2-eval=fp --p1-bandit=pucb-1.0 --bandit=ucb1-2.0 --policy-mode=x --threads=8 --mirror-match
# ...
score: 0.639344 over 61 games; Elo diff: 99.4568
39 0 22
info: 
        0: 93, (0.47058/0.467969), (0.566559/0.5625)
        1: 56, (0.373886/0.371094), (0.532504/0.527344)
        2: 151, (0.888732/0.886719), (0.431848/0.421875)
        3: 121, (0.778775/0.773438), (0.595778/0.584635)
        4: 53, (0.98679/0.984375), (0.768821/0.777344)
        5: 49, (0.890679/0.886719), (0.494158/0.49486)
        6: 44, (0.964383/0.964844), (0.535601/0.542969)
        7: 202, (0.973888/0.972656), (0.520967/0.527344)
^Cscore: 0.639344 over 61 games.
```

With this change, the network is now 2:1 vs 'fp'.

# Python Scripting

The primary goal of this program is to train a strong network for IS-MCTS like Foul-Play. We claimed that the Oak Python API is sufficient to reproduce all of the C++ binaries.

This section will focus on reproducing the core functionality of `chall`, since this is closest to the task of integrating Oak with Foul-Play.

### chall

Let's first go over the program. We will cover the 

### Search Objects

All the now-familiar search parameters are contained in the oak.Agent class:

```bash
(.venv) user@laptop:~$ python
Python 3.13.3 (main, Jan  8 2026, 12:03:54) [GCC 14.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import oak
>>> agent = oak.Agent()
>>> for field in [agent.budget, agent.bandit, agent.eval, agent.matrix_ucb, agent.table]:
...     if type(field) == str:
...         print(f"'{field}', length:{len(field)}")
...     else:
...         print(field)
...         
'', length:0
'', length:0
'', length:0
'', length:0
False
False
>>> 
```

The above shows that all fields are strings, except for `discrete` and `table`.

An empty `eval` string will default to Monte-Carlo, but `bandit` and `budget` must be provided. The `table` flag has not previously been discussed, but it is alternative (and WIP) data structure for the search: a transposition table instead of a tree. 

The tree/table is exposed as the Heap class

```bash
>>> heap = oak.Heap()
```

The Heap class is almost totally opaque. It is meant to be created, passed into the search function, and destored.

The last piece is the Input class, which encodedes the information of a determinized or perfect-information battle. The underlying data of this is the `pkmn_gen1_battle` and `pkmn_gen1_chance_durations`. However, we do not expose the fields of these structs, and the Input class is opaque save for print functions

A search Input is defined via a battle string, in the same format as the `chall` program inputs

TODO

# Training a Team-Building Network

The team building network architechture and infrastructure is much simpler than that for battling.

TODO

# RL

The `rl` program is very simple. It runs `generate` and the training scripts `battle`/`build` at the same time but with the `--in-place` flag for the learners.

This means that, in addition to saving the updated network parameters in the usual way (i.e. "working-dir/step.battle.net"), it will save the latest parameters the path that `generate` reads from. Each self-play worker reads the parameters again at the start of each battle

```bash
(.venv) $ rl --budget=2048 --fast-budget=128 --fast-search-prob=.935 --bandit=pucb-0.25 --policy-mode=e0.9x0.1 --fast-policy-mode=x --batch-size=2048 --lr=.0001 --build-batch-size=0 --build-lr=0 --build-trajectories-per-step=0 --build-keep-prob=0 --sleep=4 --data-window=16 --delete-window=32
```

The above is an example of a fast run with no team-building. The latter only takes place when `--team-modify-prob` is non-zero, among other conditions. The lack of `--discrete` flag means the net will use ReLU activations and won't be quantizable. The battle learner will use only the most recently generated files, set by `--data-window`. Any file outside of `--delete-window` will be automatically deleted (so always set it larget than data window.)

`--sleep` sets a mandatory wait period (in seconds) between each step of the `battle` learner. This is because RL is almost always bottle-necked by the speed of data generation. We slow the learner down so it isn't seeing the same data all the time.

```bash
(.venv) $ rl --budget=1024 --fast-budget=256 --fast-search-prob=.75 --bandit=pexp3-1.0-0.1 --policy-mode=n --fast-policy-mode=x --batch-size=8192 --lr=.001 --lr-decay=.99 --lr-decay-interval=100 --build-batch-size=2048 --build-lr=.01 --build-trajectories-per-step=2048 --build-keep-prob=.5 --sleep=2 --max-pokemon=1 --team-modify-prob=1 --pokemon-delete-prob=1 --sleep=4
```

Proof of concept `rl` run with team-building but only for 1v1.

### Args

The arguments for `rl` contain the arguments for both `battle` and `build`, where the latter's arguments are prefixed with "build-" for disambiguation. For example

* `--lr` sets the learning rate build `battle`

* `--build-lr` sets the learning rate for `build`

