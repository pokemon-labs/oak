# Ataraxos: birch's Gen 3 RL, ported to Gen 1 on libpkmn

This is `~/oak/ataraxos/`, on oak's branch `main2` (pushed to origin). It
recreates `~/birch` (Gen 3 on hoenn: read its `CLAUDE.md` and `README.md`)
for Gen 1 OU on libpkmn, oak's engine (`cpp/include/libpkmn/`,
`cpp/extern/engine/`). Keep birch's design and code; drop only the Gen 3
parts. Reuse oak's code where possible (`client.h`'s `compare_battles`,
`log.h`'s binary-to-text parser, `init.h`).

## Working with the user

- **Stop and confirm decisions** before going far with them: lay out the
  determinations and open choices, recommend, wait.
- **Tests run on the box**, not the laptop (the laptop only compiles).
- **Laptop network: only `git push` and moving team files.** Everything else
  (downloads, builds, pip) uses the box's internet. Code reaches the box by
  committing on `main2`, pushing, and `git pull` on the box.
- Commits: short lowercase messages, the user's git identity, no Claude
  trailers. Commit only `ataraxos/`; the user's uncommitted oak edits
  (`cpp/include/search/mcts.h`, `cpp/src/sprt.cc`, `prior.h`, `test.py`,
  `RL/`) stay out.

## The box

Vast instance, `ssh -p 62252 root@180.75.78.58` (BatchMode works with the
laptop's key). It also runs the user's birch training (`runs/axstall`, on the
GPU) and safari/glegle ladder bots: don't touch `~/oak`, `~/birch`, `~/safari`
there or their processes; run tests with `nice -n 10` and few threads. Our
clone is `~/oak-ataraxos` (main2, submodules engine and pybind11). libpkmn
was built there with `/opt/zig-0.16` (zig master is too new) into
`~/oak-ataraxos/cpp/libpkmn/ReleaseFast/client/`. Python is
`/venv/main/bin/python` (torch 2.14 cu132). Build: `cd ataraxos && cmake -S .
-B build && make -C build`. `box/setup.sh` runs every test
(`PY=/venv/main/bin/python`).

## Decisions (the user's unless marked)

- Every set maxes DVs and stat experience; no items, abilities, natures,
  EVs/IVs. A set is species, level, moves. Teams are Showdown export files
  (`src/teams.hpp`); `teams/selfplay/oak1500.txt` and `oak150.txt` were
  converted from `~/teams1500`, `~/teams150` (`codegen/oak_teams.cjs`).
- The view (`src/view.hpp`, libpkmn writes only the omniscient log): the
  foe's HP as Showdown's percent (ceil, 99 when not full), sleep shown as
  status 1, the foe's Pokémon renumbered by first appearance (birch does the
  same). Note: oak's `log.h` lacks the 99 cap.
- Tokens: 11 columns (KIND, SUB, ACTOR, OTHER, MOVE, MOVE2, SPECIES, NUM,
  HP, MAX_HP, STATUS). Request: 107 fields (`src/tokens.hpp`).
- Choices are libpkmn's; Gen 1 has no hidden-illegal choice (Disable is
  announced, partial trapping only stops the move), so birch's error /
  re-request path is gone. `maybe_locked` isn't filled yet.
- Nets: only the causal net's infrastructure; MoveNet and HistoryNet merged
  away into `ataraxos/nets/policy.py`.
- State tokens (approved, not written yet): the turn uncapped to 1000 (int16;
  libpkmn's turn is 16-bit), no field token; two actives (species, types,
  boosts, stats, moves, disabled move, the duration counters, the volatile
  flags); both sides' 6 Pokémon with PP (the request has no bench PP).
- Bookkeeping = safari's `PSBattle` + `Mechanics` (`~/safari/src/battle.py`,
  `mechanics.py`, text-based Python) ported to the binary log, including
  Gen 1's stat modification glitch. Tested like safari's `tests/client.py`
  with `compare_battles` and safari's key, plus (Claude's, from the test)
  the foe's `last_moves.index` not compared: a full paralysis hides the
  chosen slot, as safari's key already skips the foe's selected move.

## Status (2026-10-31)

Phase 1 (env, view, tokens, request, teams, binding, the Python package,
box scripts): every test passes on the box.

Phase 2 (bookkeeping, state tokens):

- Bookkeeping (`src/bookkeeping.hpp`) is done for now (the user: move on).
  `build/check_bookkeeping 2000 SEED` mismatches 16, 17, 21, 21 of 2000
  games on seeds 1-4 (from 1756). `check_bookkeeping GAMES SEED - REASON`
  prints the mismatches whose reason contains REASON with the log and both
  battles. Each fix's commit names the rule (`git log src/bookkeeping.hpp`).
- Test key additions (the user's calls): the foe's recharging is skipped
  after a flinch move hit it (a flinch clears the recharge, no message); a
  substitute's HP is skipped once a Psywave hit it (damage never shown).
  The substitute's HP must stay an upper bound: compare_battles fails one
  below the truth whatever the key, so Psywave into a sub counts 1 (as
  safari). Our own recharging comes from the request's `recharge`.
- Left open: (1) binding selections, most of the non-Mimic PP / moveset
  mismatches: in Showdown mode a binding side's selected move is logged
  and becomes last_used_move but the bind just continues (no PP, the same
  damage; libpkmn `mechanics.zig` 154-162, 723), and the bind's end is
  silent, so a continuation and a real move after it look alike. `a30cadb`
  ("a different move ends a bind") treats a new move as a new bind: right
  for last_used_move, wrong for PP. (2) Mimic's PP, on hold (the user's
  call). (3) a few last_damage off 20-40% (the other side's HP is the
  middle of its percent; big-HP Pokémon). (4) one thrashing, one
  counterable. Tried and reverted: a bind's later turns keeping
  last_damage (libpkmn reuses it, capped by a KO) made every seed worse.
- **Next**: `src/state_tokens.hpp` and the net's `embed_state` for the
  approved layout (Decisions), turn `state` back on (`CausalConfig.state`),
  rerun `tests/py/test_causal.py` on the box. Two questions were put to the
  user and not yet answered, ask again before writing the columns: the
  foe's unseen Pokémon as "unseen" tokens (species None) so each side has
  6 (Claude's recommendation), and the foe's PP known only for revealed
  moves, unrevealed moves None with PP 0 (Claude's recommendation).

Phase 3: Showdown text to the binary log (birch's `text_parser.hpp`), the
client (`client.hpp`), the ladder driver (`birch/play/showdown.py`,
`botctl.py`), and their round-trip tests (`check_text_parser`,
`check_client`). The user wants whatever scope birch has.

`box/hf_sync.sh` targets `opencover/ataraxos-runs` (the user's OK).
