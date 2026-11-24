# pi4: a strong perfect-info teacher on rby4's inputs

2026-11-24. The user's plan: train a strong perfect-info net, then distill
it into a Showdown-view student. pi4 replicates rby4 (the strong net) as
closely as possible, with perfect-info input, no history encoders, two
policy outputs and Expected SARSA. Nothing here is built yet.

## Where things stand

- **pi3** (`ataraxos/pi/net.py` size `pi3`, 1.42M params, oak's battle
  encoding): ckpt1925 vs rby4 ckpt1750 0.247 [0.234, 0.261], vs pi2
  ckpt2675 0.646, over 4096 games each. Stopped at ckpt1950 for pi4
  (resume: `tmux new-session -d -s pi3 -c ~/oak/ataraxos "RUN=pi3
  SIZE=pi3 box/run_pi1.sh"`).
- **rby4** (cs3 + Expected SARSA, 8.49M params): the decision block 3.29M
  (`history_encoder`, despite its name: 5 layers, d 256), the history
  4.33M (2.63M causal layers over decisions, 1.32M update encoder, 0.38M
  line encoder), ~0.9M embeddings and heads. Lineage: rby3 1050 iterations, then rby4 1750,
  ~23M games in all (pi2's "14.4M" counted only rby4's own).
- **What pi3 lacks of rby4's input:** boost stages (accuracy and evasion
  are lost entirely), the disabled move, Bide's damage, the turn, the
  matchup features (`src/features.hpp`), exact HP (oak's 50 buckets),
  side summaries and the clause flags, and the last-move state below.
  Neither net has the last-move state as input: rby4 saw it only through
  its 8-decision history of the log.

## The user's calls (2026-11-24)

- **Input looks like rby4's:** `src/state_tokens.hpp` + `src/features.hpp`
  + the request tokens, filled with perfect information. Teacher-only aux
  inputs are fine (the student leaves them out).
- **Perfect info means what someone who knows both teams sees, never a
  hidden value:** exact HP, sets, PP, Durations counting up (no hidden
  counters, as pi1-3), and the last-move fields as tracked from the log,
  never a click that was not shown.
- **Both nets keep last_damage, last_selected_move and last_used_move**
  (Counter is huge in the metagame) plus a "desync possible" bit. No
  Counter-specific matchup features (too complicated).
- **Expected SARSA / a Q head**, as long as it doesn't preclude
  distillation (it doesn't: the student can distill Q as well).
- **Two policy outputs** (one forward pass gives both players' policies).
- **No search in training.**
- **10-15M params** (no history encoders, so a bigger trunk).

## Counter in our env (libpkmn `-Dshowdown`, Desync Clause on)

From `cpp/extern/engine/src/lib/gen1/mechanics.zig` (read, not yet
tested). Counter (`counterDamage`, :1266) succeeds iff `battle.last_damage
!= 0` (one value for the whole battle), the foe's last *executed* move was
counterable (`lastMove(foe).counterable`, set by `setCounterable` :797
whenever the foe runs a move: charge turns and Metronome / Mirror Move's
called move included) and the foe's `last_selected_move` is counterable
(Normal or Fighting, bp > 0, not Counter). Exactly one of the two: the
desync glitch, which Desync Clause turns into a fail. Damage 2 x
last_damage. Counter moves after any non-Counter move (:286).

- `last_damage` is zeroed at the start of each non-Counter move that
  reaches the damage step (:869) and set to its damage (capped at the
  target's HP, uncapped into a substitute, the drained HP for drain moves,
  0 on a miss, 1 after a Jump Kick crash); confusion self-hits and Bide's
  release set it too. Switching, residual damage, turns that can't move,
  onBegin status moves and charge turns leave it.
- The counterable flag changes only when that side executes a move; it
  survives switching. `last_selected_move` is written at selection
  (Showdown saves a click even while asleep or frozen, :146), survives
  switching, and isn't rewritten on forced turns (recharge, Thrash, Rage,
  charge).
- So carried state decides Counter when the foe doesn't run a damaging
  move first this turn: a switch-in, Hyper Beam's recharge, a sleeping /
  frozen / paralyzed foe.
- Asleep or frozen (or bound on a bind's first turn), Showdown's request is
  one "Fight" (src/request_data.hpp, matched against real Showdown by
  check_client), played as libpkmn's first legal move (engine_choice): the
  player can't pick, and the saved selection is that move. (libpkmn's
  comment says Showdown makes a sleeping Pokémon "select a move normally";
  the requests we checked say otherwise.) So a real choice goes unshown only
  on full paralysis, flinch, a confusion self-hit, being bound, or a turn
  cut short by a faint.

**Tracking from the log** (`src/bookkeeping.hpp`, safari's port, already
tracks last_damage, last_selected_move, last_used_move and counterable):
our side exact; the foe's counterable flag exact from its move lines;
last_damage exact on our side, from the percent on the foe's, and the
min-roll damage formula for a substitute hit (safari's `sub_dmg`); the
foe's last_selected_move is stale after a click it never showed.

**`maybe_desync`** (new, per side, in the Bookkeeping):

- set when that side's click went unshown: `cant` for paralysis, flinch,
  sleep, freeze or bound; waking up; a confusion self-hit; nothing shown
  in a turn both actives started (a faint ended the turn before it moved,
  its own or the other's: Self-Destruct, poison after a switch-in; its
  click was a move, since a switch goes first; switchIn doesn't reset
  last_selected_move, so the click carries to the replacement);
- not set on forced turns (recharge, Thrash / Petal Dance, Rage, charge);
- cleared by a move line from a fresh click, or a `cant ... Disable` line
  (it names the move);
- kept through switches;
- the input bit is `maybe_desync && counterable` (with the flag clear
  Counter fails whatever was clicked).

## Design

**Env: lockstep rows on rby4's env** (`src/env.hpp` + `BattleData`, the
libpkmn build with the log, `state` tracking on), not `src/pi_env.hpp`
(no log). A `pi4_obs` beside `pi_obs` groups each gather's decisions by
battle: one row per battle with 1-2 deciding players, the other side n 1.
Why: the tracked fields need the log; each side's tracked last-move fields
come from the *opponent's* Bookkeeping, so the teacher and the student
share one implementation; the choices are Showdown's (the student's action
space, no mapping for distillation); and evals can play rby4 directly (as
`ataraxos/eval/pi_h2h.py` does). Cost: the Bookkeepings update every step,
so the env is slower than pi_env's (the GPU should bound it at this size).

**Input** (P1 as "ours", one forward per row):

- the state tokens (`encode_state`) and matchup features
  (`encode_features`) run on a Bookkeeping holding the true battle:
  P1's side by party position, P2's by P1's first-appearance order with
  the unseen ones after, every Pokémon seen, true sets, PP, exact HP,
  status, the engine's Durations;
- appended fields (zero-initialized in the student, so rby4 still loads):
  last_damage on both active tokens (pi4: the engine's, which it could
  compute from the exact HP and substitute HP it sees; the student: its
  Bookkeeping's), and on each active token that side's last_selected_move,
  last_used_move, counterable and `maybe_desync && counterable`, tracked
  (P1's from P2's Bookkeeping and vice versa; the student's own side from
  its own choices);
- teacher-only: P2's 6 switch-in features (the mirror of the features'
  "our 6 as they'd come in", for P2's policy);
- both players' request tokens (rby4's 11 each: header, 4 moves with
  their features, 6 side slots with switch-in features), from each
  player's real request.

**Net** (`ataraxos/pi4/net.py` or a size in `causal_net.py`): rby4's
decision block with no history: pre-LN layers over [value token, 14 state
tokens, P1's 11 request tokens, P2's 11], a side embedding on each.
d 384, 6 heads, ff 1152, 8 layers (rby4's 64-dim heads and 3x ff;
~11.8M in the layers, ~14M in all). Heads: W/D/L value (P1's), rby4's
Policy (query vs each choice's key) once per side, rby4's Q head per side
(W/D/L per choice, that side's view).

**Trainer:** `ataraxos/pi/train.py` (lockstep rows, both policies, the
speed flags) with:

- the new obs and net behind the existing flags (`--buffer numpy`,
  `--actor graphs`, `--compile_train`, ...), generalized from pi's three
  input arrays to the new ones;
- Expected SARSA ported from `ataraxos/rl/buffer.py`'s `sarsa_targets`:
  each side's Q target is its next row with 2+ choices, Q summed over the
  data policy; rby4's `--expected_sarsa 1 --q_coef 1 --sarsa_adv 0`;
- rby4's settings: `--td_lambda 0.6 --ema_decay 0.98 --adv_filt_rate 0.5
  --lr_coef 0.003 --lr_decay 0.6`, the team flags (`--team_gamma 0.5
  --team_gamma_tau 0 --team_eta 1e-5`), minibatch 12288, 8192 battles;
  fresh schedules (`--lr_ceil 3e-4 --temperature_ceil 0.1`, rby3's start);
  search off;
- evals every 25: maxdamage (saturates ~0.89), pi3 ckpt1950 (on
  `pi_obs`) and rby4 ckpt1750 (its own view), policy only.

## Built (2026-11-24, not yet run on the box)

- `src/bookkeeping.hpp`: `unshown` (`hide_click`, `forced`, `acted`);
  `cant ... Disable` sets the foe's last_selected_move to the move named.
  `check_bookkeeping` also checks `unshown` against the truth (a laptop
  run before the user asked for box-only tests: 2000 games seed 1, 76 games
  mismatched as before; the tracked selection differs in 37672 of 448634
  views, 2 of them unflagged, 1 changing counterability, to look at on the
  box; the flag set in 9.7% of views, with the counterable flag in 2.9%).
- `src/state_tokens.hpp`: `LAST_FIELDS`, `encode_last`. `src/pi4.hpp`:
  `pi4_bookkeeping`, `encode_pi4` (PI4_FIELDS = STATE_FIELDS + 9 per
  viewer). `tests/cpp/check_pi4.cpp` (oak1500: 0 mismatches in 101 games,
  laptop).
- `src/env.hpp`: `set_light`, `tracking`. The binding: `pi4_obs(rows)`,
  `set_light`, `stats`.
- `ataraxos/pi4/`: `net.py` (Pi4Net, sizes pi4 and pi4_tiny), `env.py`
  (Pi4Env), `agents.py` (evals in rby4's env against anything).
- `ataraxos/pi/train.py` + `fast.py`: `Model` (pi or pi4), the actor's Q
  columns, `--expected_sarsa --q_coef --sarsa_adv`.
- `tests/py/check_pi4.py`, `box/run_pi4.sh`.

## Steps

1. **Bookkeeping:** `maybe_desync`; check_bookkeeping compares the
   last-move fields with the truth and checks the flag is sound (set
   whenever the tracked last_selected_move differs from the truth's),
   reporting how often it's set.
2. **The perfect-info Bookkeeping and `pi4_obs`:** a check that its tokens
   match libpkmn's state (sets, PP, HP, status, Durations, last_damage)
   and its tracked fields the opponent's Bookkeeping.
3. **The state layout's new fields** in `state_tokens.hpp` (appended after
   the features), the student's cs3 reading them through zero-initialized
   linears (rby4 ckpt1750 loads unchanged).
4. **The net**, with checks (shapes, graphed actor = eager, learner =
   actor).
5. **The trainer:** obs, SARSA (a check against rl's on the same data),
   evals.
6. **Box:** build, run every check, benchmark a few iterations on the
   3060. Rough guess before measuring: 2-4x rby4's compute per decision;
   rby4 took ~50 s an iteration on a 4090-class card, so on the 3060
   expect minutes an iteration. Then choose the card for the run (~23M
   games to match rby4's lineage).

## Later: the student

Distillation itself is out of scope here. What pi4 fixes for it: the
same encoder and choices; the student is cs3 plus the new state fields,
reading its own Bookkeeping. The imitation gap (the teacher knows the
foe's set; a student copying its policy learns the average of its picks):
distill value and Q as well as the policy, label the states the student
reaches (`pi4_obs` on the student's games), then RL fine-tune.
