# Ataraxos

Self-play reinforcement learning for Gen 1 OU on Pokémon Showdown, on
libpkmn (oak's engine). It's birch (Gen 3, on hoenn) ported to Gen 1: the
same causal net, training loop, Exp3 over the team pool and evaluation.
Gen 1 has no items, abilities or natures, and every set is assumed to max
its DVs and stat experience, so a set is its species, level and moves.

The agent sees only what a Showdown player sees: its own view of the
battle log, and the `|request|`. In training, both come from a true libpkmn
battle (libpkmn writes only the omniscient log, so `src/view.hpp` makes
each player's view).

## Layout

| Directory | What |
| --- | --- |
| `src/` | The pipeline from the log and request to the net: `obs.hpp` (libpkmn's binary messages), `view.hpp` (a player's view of them), `request_data.hpp`, `tokens.hpp` (the net's input), `teams.hpp` (Showdown export teams, random teams), `battle_data.hpp` and `env.hpp` (lockstep self-play) |
| `ataraxos/` | The Python package: `nets/` (`policy.py`, `causal_net.py`, `causal_actor.py`), `rl/` (`train.py`, `buffer.py`, `metrics.py`, `teams.py`), `eval/` (`agents.py`, `h2h.py`, `pairs.py`), `_ext/` (the binding `ataraxos_env.cpp`). Run as `python -m ataraxos.rl.train` from this directory |
| `tests/cpp/` | Checkers, each a standalone program (`check_*.cpp`) |
| `tests/py/` | The Python tests (`test_*.py`) |
| `teams/selfplay/` | Training pools in Showdown's export format, converted from oak's one-line teams by `codegen/oak_teams.cjs` |
| `box/` | Scripts for a rented GPU box: `push.sh` copies this directory with the built libpkmn and runs `setup.sh` there |

## Build

Needs oak's libpkmn "client" build (`dev/libpkmn` from oak's root), then:

```
cmake -S . -B build && make -C build
```

builds `ataraxos/_ext/ataraxos_env*.so` and the checkers in `build/`.

## Status

- Phase 1 (this): env, view, tokens, request, the causal net and training.
- Phase 2: the Gen 1 bookkeeping and the state tokens (`src/bookkeeping.hpp`
  and `src/state_tokens.hpp` are stubs; the net's `state` is off).
- Phase 3: Showdown text to the binary log, the client, the ladder driver,
  and their round-trip tests.
