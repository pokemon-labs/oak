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
- Sleep Clause and Freeze Clause are in libpkmn: `-Dshowdown` (dev/libpkmn)
  sets `options.mod` (`src/lib/common/options.zig`), which gates both
  (`gen1/mechanics.zig`, "Sleep Clause Mod", "Freeze Clause Mod"), so
  self-play has them. On Showdown a blocked sleep move is the move line, a
  -message and a -hint (read past), so the parsed bytes show a move that
  did nothing.
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

- `runs/rby2` (2026-11-01 19:39): rby1's flags but `--minibatch 12288`.
  Per battle it trails rby1 vs maxdamage early (5-9 points to 0.6M
  battles), level by ~0.8M; ~1.4x rby1's games/s. H2h, 2048 games, EMA:
  matched by battles (rby2_250 vs rby1_250) 0.382 [0.361, 0.403]; by
  gradient steps (rby2_300, 5195 steps, vs rby1_150, 5631) 0.629 [0.608,
  0.649] (`runs/h2h/`). Stopped (SIGTERM) at it 313 for rby3; resume from
  ckpt300 (copied to `/root/anchors/rby2_{250,300}.pt`).
- `runs/rby3` (started 2026-11-01 ~22:50, the user's calls, for RBY's
  variance): rby2's command line (8192 games an iteration, minibatch
  12288: the user kept rby2's) with the changes below. The lower filter
  rate keeps twice the decisions, so ~2x rby2's gradient steps an
  iteration (rby1's). Not yet: `--games_per_iter 16384 --minibatch 6144`.
  - `--td_lambda 0.6` (from 0.8: `value_ev_outcome` ~0.30, the outcome is
    mostly luck, so less of it in the targets);
  - `--ema_decay 0.98` (from 0.961; rby1's EMA beat its net late);
  - `--adv_filt_rate 0.5` (from 0.75; worth a try: the top-|A| quarter is
    likely the crit / full para / freeze turns, since A carries the next
    turn's TD error. The user isn't sure a lower threshold is the right
    fix: a chance-aware filter or advantage may be);
  - `--lr_coef 0.003 --lr_decay 0.6` (Claude's; in rby3 on "start 3".
    rby1's lr sat at the 3e-4 ceiling all run: `0.5 / (step+1)^1.1`, step
    = decisions / 5e6, falls under it only at ~4.2B decisions. This one
    (`power_schedule`): 3e-4 to ~0.23B decisions, 2.5e-4 at 0.3B, 1.2e-4
    at 1B, 8e-5 at 2B, 5e-5 at 5B. The schedule counts decisions, so at
    16384 games an iteration (~2M decisions) it moves twice as fast per
    iteration as rby1's).
  - Evals (the user's, 2026-11-01): only against rby1 at the same
    iteration, `--anchors /root/anchors/rby1_{it}.pt` (no random or
    maxdamage), so only at its 150, 200, 250, 500, 1000, 1300, 1450, 1500,
    the rby1 checkpoints left (hf_sync keeps N % 500 == 0). Resumed from
    ckpt50 with it.

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

The rollouts (`src/rollouts.hpp`, bound as `ataraxos_env.Rollouts`;
`ataraxos/search.py`), the user's calls (2026-11-01):

- `worlds` history samples per root and `reps` rollouts per (world, our
  choice), worlds * reps * n in all (Pokémon is stochastic and the foe's
  choices vary, so a world gets several); `budget` (128 to start) sets
  worlds = budget / (reps * n) when `worlds` is 0. Defaults (Claude's): reps
  2, depth 10 turns. Every rollout samples the foe's net and its own battle
  seed independently (the user: just sample the foe's policy).
- A rollout is `BattleData::resume` on the world (identity views, the
  world's Durations as libpkmn's chance durations on its first update, as
  oak's MCTS), our Bookkeeping and the sample's foe one. Our first choice is
  forced (the root's actor already ran on the root summary, so our cache is
  the real actor's, copied); the foe's cache is its rebuilt history run once
  per world (one-choice summaries joined to the next, as training; the last
  window - 1, at their true positions) and copied to the world's rollouts.
  The leaf is our first decision at turn >= root turn + depth (asked even
  with one choice); its value is P(win) - P(loss); a finished game +1/0/-1.
- The update: `[e^q rho^alpha pi^beta]^(1/(alpha+beta))`, alpha 0.002, beta
  0.02 (the paper's), rho the attack_switch magnet, pi the root's net.
- `check_rollouts GAMES SEED POOL WORLDS REPS DEPTH EVERY` (uniform self-
  play, the pool the file's first half): 100 games seed 2, 197k rollouts in
  5 s, 0 leaves early, 0 undone, 0.17% `differs` (the world's choices for us
  aren't the root's: bookkeeping's mismatches).
- `python -m ataraxos.eval.search_h2h CKPT` (one checkpoint both sides, one
  searching, the pool the first half of oak1500 and the games the second):
  16 battles at budget 128 do ~4 searches/s next to rby1's training, nearly
  all of it the rollouts' net steps (~185 ms a step of ~1000 rows, eager).
  The user: the real eval is the ladder, not h2h (stopped).
- `tests/py/check_search.py CKPT`: our copied cache rerun on the root
  gives the root's log-probs exactly (0.0000); with the foe's true sets,
  its first rollout decision vs the real foe's policy (choices matched by
  meaning: the world numbers the foe by appearance): KL mean 0.0065,
  argmax agrees 97% (the HP estimates).
- The ladder driver searches with `--search-pool FILE` (`Rollouts.add_client`;
  the Client has no Binds: `Binds{}`); the jsonl keeps q and the net's
  probs. `beanseggyum` on pkmn.bot since 2026-11-01: ckpt1100 with search
  (budget 128, reps 2, depth 10, GPU), ~0.5-1 s a decision
  (`~/arena/beanseggyum/run.sh` on the box).

Not done: CUDA graphs for the rollouts' actor.

### Joint search (2026-11-02: built, checked, not on the ladder)

Why: the vanilla search best-responds to the foe's net policy (q(a) is a
mean over rollouts where the foe samples its net), so it can exploit a
predicted switch that doesn't come (beanseggyum spamming Sing into a
paralyzed Tauros). The idea is from the ~/agent Oak fork's "Ataxaros" (see
the agent-fork-ataxaros memory: `~/agent/oak-positions/cpp/include/search/
mcts.h`, `~/agent/hypotheses.md` H19/H19i/H19j/H21). Its single update is
one-sided like ours; only its iterated mode lets the foe react, and with no
magnet that over-sharpened with budget and collapsed the foe to one action.
So we use MMD (with the magnet) on both players. Its last iterate converges
to the KL-regularized equilibrium; our closed form is the one-player case.

The user's calls (2026-11-02):

- A Bayesian game over types: 3 guessed teams per side ("3x3"). Our types:
  A0 = our real team plus A1, A2 guessed from what the foe has seen of us.
  The foe's types: B0-B2 from the existing sampler. The type weights are
  uniform for now (ideally the foe's belief over our team).
- Two modes behind an option:
  - `per_type` (default, Claude's proposal): one policy per player per own
    type. q1_A(i) = sum_B P(B) sum_j Q_AB(i,j) p2_B(j), and
    q2_B(j) = -sum_A P(A) sum_i Q_AB(i,j) p1_A(i). A rollout in world
    (A, B) feeds p1_A and p2_B (the "plus sign" in the type matrix). We
    play p1_A0, with no mixing step.
  - `per_pair` (the user's first idea): an independent joint MMD per
    (A, B), then play sum_B P(B) p1_{A0,B}.
- Each type's magnet is its net policy (pi_A or pi_B) plus rho, as now.
  The foe gets its own beta (option), defaulting to ours.
- Both sides' root choices are forced in the rollouts (the user: OK). Store
  the cell means Q_AB(i,j) and recompute q at every solve, never
  accumulate q, because the other side's policy is non-stationary.
- The budget allocation is to be experimented with. Start: rounds of
  sample, then MMD to the fixed point on the current Q, then reallocate by
  the current policies with a floor so thin cells get samples. Budget ~512
  to start (128 is thin for 9 * m * n cells); check the time per decision.
- Keep vanilla search as is (beanseggyum runs it; the ladder needs a
  baseline). Joint is a new mode in `search.py`, on main2. Not the ladder
  yet (the user: don't worry about it). No Sing replay or offline eval (the
  user: skip it).

Implementation notes (Claude's reading of the code):

- Our-side guesses: `Sampler` guesses for `1 - now.player`, and
  `Sample::foe` is the foe's Bookkeeping (our side cut to what it has
  seen). So a second `Sampler` on that Bookkeeping fills our unrevealed
  Pokémon and moves (`guess`, `fill`). This part is reuse.
- Our history for type A is new work. Our requests all game list our team,
  so for A1/A2 they must show A's sets. Rebuild it with our view's tokens
  unchanged (no HP rewrite) and the requests regenerated from A, as a
  variant of `sample()`'s foe-history rebuild; `prefill` takes it from
  there. A0 keeps the real actor's cache copy.
- Caches are per type, not per pair. The foe's history depends only on B
  (what it has seen of us is the same in every A), ours only on A. So
  prefill 3 + 3, and world (A, B) pairs our cache A with the foe's cache B.

Confirmed (the user, 2026-11-02, Claude's proposals):

- A1/A2 keep our real exact HP and everything the foe has seen; only our
  unseen Pokémon and our seen Pokémon's unused moves (the active's too, so
  our root choices can differ by type) are swapped (`Sampler::own_side`).
- Our history as A1/A2: our view's tokens as they are, the requests from
  our bookkeeping after each update with A's sets, `Bind{}` as the foe's
  (`Sampler::own`). A0 keeps the real actor's cache.
- The foe's root choice is forced, so each type's prefill runs its root
  decision too: that gives pi_A / pi_B, the magnets. A type that waits or
  has one choice at the root has m = 1 (its tokens join its first rollout
  decision).
- Types are independent sampler draws, no dedupe (a repeat counts double).
- C++ shape: `add_joint` once per root (3 + 3 samples, 5 prefills),
  `schedule(root, cells)` per round then the usual start/step, each
  rollout carries (A, B, i, j), Python keeps the cell sums; type caches
  prefilled once and copied to the rollouts each round; world (A, B) is
  B's world with our side as A's, hidden counters and seed fresh per
  rollout.
- `Sampler` takes seen slots with gaps (our side keeps party slots).

Plan:
1. C++ (`history_sampler.hpp`, `rollouts.hpp`, binding): our-side guesses,
   our history for type A, worlds from (A, B), both root choices forced,
   cell Q per (A, B, i, j).
2. `search.py`: `joint` mode with `per_type` / `per_pair`, the rounds
   schedule, uniform type weights, the foe's beta.
3. Checks, kept quick: with A = A0 and B = the true foe team, the caches
   equal today's; MMD converges on random matrices; with the foe fixed it
   gives our closed form; per_type == per_pair with one type a side.

Status (2026-11-02): steps 1-3 done (`SearchConfig(joint=True)`,
`joint_mode`, `types_a/b`, `joint_budget`, `rounds`, `floor`, `shrink`,
`foe_beta`; `Rollouts.add_joint_selfplay/add_joint_client`, `schedule`,
`joint_layout`, `joint_values`; `search.run_client` takes it too).

- MMD is extragradient (mirror-prox) by default (`extra`, Claude's): plain
  MMD cycles in the zero-sum game at any step that converges in reasonable
  time (it needs eta <= lam / L^2, lam = 0.022: >20k iterations);
  extragradient with eta 1 reaches the same fixed point in ~430 iterations
  (~40 ms numpy, 3x3 types, 9 choices), ~330 warm-started in the search.
- A cell with no rollout reads the mean of its row's and column's means in
  its (A, B) block; every cell is shrunk to that by `shrink` (1)
  pseudo-counts (Claude's). Round sampling: P(A) P(B) ((1 - floor) p1 p2 +
  floor uniform), floor 0.25, 4 rounds.
- `check_history_sampler` now also checks our side as the foe sees it:
  with our true sets `own_side` gives our real side back and `own` our real
  summaries. 1000 games seed 2: our true team always among the matches;
  side differs 20 of 195k decisions (meme moves; Mimic's slot fixed, the
  rest left: the user's call, no crash); locked / PP as the foe's.
- `tests/py/check_joint.py [CKPT SEARCHES BATTLES BUDGET MODE]`: MMD
  converges on random games; with the foe fixed it is the closed form
  (1e-6); per_type == per_pair with one type a side. With ckpt1100, the
  foe types' root policies with the true sets vs the real foe actor's: KL
  0.0007-0.01, argmax 100% (its root request has the real binds now); 0
  `differs`. Budget 512: ~1.1 s a root (per_type and per_pair alike, 2
  roots batched, next to rby3), mostly the 4 rounds' rollouts; MMD ~0.2 s.

Not done: the ladder (the user: later), budget-allocation experiments,
type weights from the foe's belief, MMD on the GPU or in C++ if 0.2 s
matters.
