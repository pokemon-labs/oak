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
- Training follows Showdown's request (the user's call, 2026-10-31): the
  request (`request_data`) is the JSON Showdown would send (a hard lock;
  a semi-lock for Bide and a bind's second turn; "Fight" while asleep,
  frozen or bound by a bind's first turn; Struggle; `disabled` for no PP;
  `maybeLocked` on a bind's later turns and the turn after it ends on its
  own unless it lasted 5, from Showdown's gen1 conditions.ts), the env's
  choices are Showdown's (`showdown_choices`), each played as libpkmn's
  (`engine_choice`: the same, else the first legal move). `Binds` tracks
  the bind lengths libpkmn doesn't keep. The bookkeeping hears each
  choice as the player knows it (`Bookkeeping::selected`), the same in the
  env and the client. Gen 1 has no hidden-illegal choice, so birch's
  error / re-request path is gone.
- Nets: only the causal net's infrastructure; MoveNet and HistoryNet merged
  away into `ataraxos/nets/policy.py`.
- State tokens (`src/state_tokens.hpp`, 265 int16s, 14 tokens): the turn
  clamped to 1000 (int16; libpkmn's turn is 16-bit), no field token; two
  actives (species, types, boosts, stats, moves with PP, disabled move,
  the Durations, sub HP, toxic counter, Bide's damage, 16 volatile flags;
  libpkmn's hidden counters left out); both sides' 6 Pokémon with PP (the
  request has no bench PP), ours by party position, the foe's by first
  appearance, each with an active flag. The foe's unseen Pokémon are
  "unseen" rows (species 0, all zeros), so each side has 6; the foe's
  unrevealed moves are 0 with PP 0 (both Claude's recommendations, the
  user's OK). The turn rides on both active tokens.
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
- State tokens and `embed_state` are done (`a5e2a04`), `CausalConfig.state`
  is on by default; `tests/py/test_causal.py` passes on the box on CPU
  (`check_state`'s invariants rewritten for the layout), the cs net is
  4.40M params. Not run: test_causal's CUDA graphed-actor check (the GPU
  is the user's birch run).

Phase 3 (done but the open items): Showdown text to the binary log (birch's `text_parser.hpp`), the
client (`client.hpp`), the ladder driver (`birch/play/showdown.py`,
`botctl.py`), and their round-trip tests (`check_text_parser`,
`check_client`). The user wants whatever scope birch has.

- The user's calls (2026-10-31): the formatter is ataraxos's own
  (`src/format.hpp`, a port of libpkmn's `src/pkg/protocol.ts`, which
  libpkmn's integration test checks against Showdown), not oak's `log.h`
  (approximate, and other oak modules use it); and a check against real
  Showdown text from Showdown's simulator on the box.
- Done: `src/names.hpp` (Showdown's Gen 1 names by libpkmn ID, from
  `codegen/names.cjs` on the laptop's Showdown, 3de1c70), `src/format.hpp`
  (Showdown's spellings where protocol.ts differs: Mist is -activate, no
  trailing empty field), `src/text_parser.hpp` (a view's parser numbers the
  foe by first switch-in, so the ladder needs only our nicknames), and
  `check_text_parser`: 0 mismatches on 20000 games (seed 2); `forms` lists
  the text forms covered.
- Real Showdown text (`check_showdown`): Node 22.11 and Showdown (5b01be8)
  are on the box in `~/oak-ataraxos/node` and `pokemon-showdown`
  (git-excluded; the user installed them). `tests/showdown/play.cjs` plays
  Gen 1 OU in Showdown's simulator (our teams, "No Ability" as the
  validator would set, uniform choices) and dumps every stream with the
  simulator's HP and status; `check_showdown DUMP` parses the raw log and
  both player streams and compares what they show with the truth: 0 parse
  errors and 0 mismatches on 5000 battles (seed 2). Found and fixed:
  Showdown writes no [of] on Recoil (the parser takes the other active, as
  libpkmn does). Run: `../node/bin/node tests/showdown/play.cjs 1000 1
  teams/selfplay/oak1500.txt > dump.txt && build/check_showdown dump.txt`.
- Open (for the user): Showdown's Sleep Clause and Freeze Clause, which
  libpkmn doesn't have. On Showdown a blocked sleep move is the move line,
  a -message and a -hint (read past), so the parsed bytes show a move that
  did nothing; self-play training has no clause at all.
- The client (`src/client.hpp`, birch's; `src/json.hpp` is hoenn's JSON
  reader): our side's Bookkeeping from the first request's sets through
  libpkmn's Init (the user's call), so a rejoin's state is off.
  `check_client 2000 1` (birch's: formatter text and this file's Showdown
  JSON vs BattleData's tokens, fields, choices, state): 0 failures, also
  on `teams/selfplay/oak1500.txt` (3rd argument). `check_client showdown
  DUMP` (play.cjs now dumps each request, the choices Showdown takes and
  the one made): all 188k real requests of 1000 battles parse and give
  Showdown's choices exactly. Request shapes per move request, env vs
  Showdown on oak1500: Fight 11.5/11.0%, hard lock 1.10/1.00%, semi-lock
  0.22/0.25%, maybeLocked 89/106 per 1000 games (noise at ~100 counts,
  the user's call; every Showdown maybeLocked is a bind and the cases
  traced match the rule), Struggle 0.15/0.03% (the env's games run
  longer: no clauses).
- `check_view` checks Showdown's choices against libpkmn's (same
  switches, every move maps to a legal move).
- `check_bookkeeping` now chooses as training does (Showdown's choices,
  `selected`): 76, 65, 66, 80 of 2000 on seeds 1-4 (16-21 with libpkmn's
  choices, which leaked whether a bind goes on: the new numbers are the
  baseline, the user's call). Most of what's left is a pick on a hidden
  bind turn that the log never shows (`last_moves.index` after a bind,
  `last_selected_move`): hidden state.
- The ladder driver (`ataraxos/play/showdown.py`, `botctl.py`, birch's
  with gen1ou, `Client.choose` for the choice sent, `--server local`
  with `--name`), the binding `ataraxos/_ext/ataraxos_client.cpp`, the pool
  packer `codegen/pack_teams.cjs`, and `tests/showdown/local_server.cjs`
  (Showdown's server on 127.0.0.1:8123, noguestsecurity, config.js
  untouched). Tested end to end there: two bots with an untrained cs_tiny
  net, a schedule of 3 games, chat redacted, botctl state / games /
  propose / choose: 3 games each, both quit, no client failure or unknown
  tag. Not done: anything on the real server (outward-facing: ask first).
- Training: `runs/rby1` on the box (started 2026-10-31, `box/run_rby1.sh`):
  cs2b's flags (cs2, 8.35M params, 8192 battles) on oak1500, evals vs
  random and maxdamage every 25 iterations (512 battles), synced to
  `opencover/ataraxos-runs`. ~48 s an iteration, ~10 GB GPU. The user
  stopped birch's cs2b for it. Then: the real ladder when the user says.

`box/hf_sync.sh` targets `opencover/ataraxos-runs` (the user's OK).

## Phase 4: search (started 2026-11-01)

The history sampler (`src/history_sampler.hpp`, the user's design): a
world for an Ataraxos rollout, no replay. The user's calls:

- The world is our Bookkeeping with the foe's unrevealed species and moves
  filled in (DVs and stat experience maxed, so a set is species, level,
  moves). Party order doesn't matter.
- The guess, safari's TeamPredictor: pool teams that fit what's revealed
  (Kuhn matching, birch's), drawn by the prior; else safari's SetDict (set
  weights from the prior, Species Clause); else random OU moves (oak's
  move pool) for a Pokémon no pool set fits. The guesses assume a strong
  foe. Prior: rank decay over teams1500's peak-rating order (`Pool(teams,
  ratio)`, 0.1 by default, Claude's pick), or `set_prior` (Exp3's
  `q_avg` once `rby1` has run long: it's near uniform early).
- The foe's history is divorced from Showdown: our view's bytes tokenized
  with viewer = the foe. HP must be consistent (the user): the foe's own
  HP is a running estimate from our percent (`bk::hp_from_percent`,
  bookkeeping's rule, factored out) out of its max HP, ours a percent; a
  Faint line zeroes it (Explosion's user has no Damage line). Its request
  after each update is `request_data` on our bookkeeping after that update
  (`Client::after()`) with the same guess and those HPs; a fainted foe
  Pokémon has no status (libpkmn clears it, bookkeeping keeps it).
- Our numbering (ours by party, the foe's by appearance) on both sides of
  the history and the world; a rollout's Views start as `identity_view`.
  The world's hidden counters are oak's `randomize_hidden_variables` on
  bookkeeping's Durations, a fresh seed each.

`check_history_sampler GAMES SEED POOL DRAWS EVERY [half|random]`
(self-play, every decision): with the foe's true sets the rebuilt history
equals the real one (summaries, tokens, requests; the foe's own HP to its
percent), its lines, requests and world agree on the foe's HP (0
inconsistent, the user's worry), and pool draws give our real legal
choices and play out without an engine error. `half`: the pool is the
file's first half, games from the second; `random`: random teams
(distinct species). 1000 games, seed 2 (~90 s): 0 inconsistent in every
mode; the summaries, tokens and requests all equal but `locked` in 423 of
193k requests (the foe's Bind, maybeLocked / Fight, is left default:
`Binds` reads libpkmn's hidden bind counter) and ~40 PP (bookkeeping's
known PP); our choices differ in 124 of 773k draws (bookkeeping's
mismatches); the true team is always among the matches. `half` draws:
37% a team, 51% sets, 12% random moves.

Next: the rollouts (both sides' nets from their histories, our forced
first action, depth in turns, the value head at the leaf) and the closed-
form update `[e^q ρ^α π^β]^{1/(α+β)}`.
