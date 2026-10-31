# Ataraxos → ADV OU: context for development

Background for applying the Ataraxos recipe (arXiv 2511.07312, "Superhuman AI for Stratego…")
to Gen 3 Pokémon singles (ADV OU) at a smaller scale. Paper numbers below are from its
Methods and Supplementary Information (S3.3–S3.7, Tables S5/S7/S11/S12). Anything marked
*(ours)* is a project decision or a suggestion, not something from the paper.

## The recipe in one paragraph

Ataraxos is near-on-policy PPO self-play with two additions: (1) a regularizer whose
coefficient is annealed over training ("damping"), and (2) test-time search that is just
one more regularized policy update at the root, computed with rollouts. That is what lets
it skip CFR-style machinery: no policy averaging and no trajectory importance reweighting.
Hidden information is handled by a separately trained belief network, used only at search time.

## Move learning: deltas from vanilla PPO

Loss: `L = L_π + L_v`, where `L_v` is cross-entropy against a categorical W/L/D outcome target, and

```
L_π = −min(r·Â, clip(r, 0.8, 1.2)·Â)  +  0.1·KL(π_θ ‖ π_data)  +  α_t·KL(π_θ ‖ ρ)
```

| Item | Paper value | Note |
|---|---|---|
| Ratio clip ε | 0.2 | standard PPO |
| Reverse KL to data-collection policy | 0.1 | update-size control (on top of clipping) |
| Magnet/regularizer KL coeff α_t | `0.05 / iter^0.3` | the "damping"; annealed |
| Magnet ρ | uniform movable piece, then uniform legal move for it | hierarchical, not uniform over moves |
| LR (Adam) | `clip(0.5 / iter^1.1, 5e-6, 1e-4)` | ≈1e-4 until iter ~2.3k, hits the floor at iter ~35k |
| Grad-norm clip | 0.267 | |
| λ for advantages | 0.5 | separate λs for policy and value |
| λ for value (outcome) targets | 0.8 | |
| Advantage filter | keep a step only if \|Â\| ≥ 0.75-quantile **and** ≥ 0.01 | ~2.5× faster per iter, and *better* sample efficiency + final strength |
| Epochs | 1 per iteration | |
| EMA of weights | 0.999 | |

Things that are easy to get wrong:
- **Rewards are terminal-only** (W/L/D), and regularization goes into the *loss*, not the
  rewards ("myopic" KL). With γ = 1 and sparse rewards, `δ_t = V(s_{t+1}) − V(s_t)`, except
  at the last step, where `δ = r_T − V(s_{T−1})`.
- **Data layout:** persistent vectorized envs (1,536/GPU), each stepped 202 moves per
  iteration. Training then runs over 202 batches grouped by simulator step, one epoch.
  Trajectories are therefore truncated at iteration boundaries, and λ-returns must
  bootstrap there. The clipping and KL-to-data terms only do anything *because* the policy
  drifts across those sequential batches. With one gradient step per fresh batch they are
  no-ops (r = 1, KL = 0).
- **Both KLs are reverse** (`KL(π_θ ‖ ·)`). `KL(π ‖ uniform) = log|A| − H(π)`, so the
  magnet term is an entropy bonus measured against ρ.
- **Annealing is the main tuning knob.** The paper's "energy reserve": anneal α too slowly
  and play stays weak; too fast and entropy collapses, the policy stops being able to
  learn, and it becomes exploitable. Watch entropy, KL, and clip-rate curves.
- Training used a **shorter draw rule (100 moves) than evaluation (200 moves)** because it
  trained better (discourages dawdling). *(ours)* The ADV OU analogue is a training turn
  cap tighter than the real one, to discourage stall loops.

## Setup learning (paper) → team pool + Exp3 *(ours)*

Paper: a decoder-only transformer places 40 pieces in fixed row-major order. It is trained
with **Monte Carlo** returns (the game outcome) plus a max-entropy term
(`α = 0.1 / iter^0.3`) at a higher LR (5e-5), with no advantage filtering. The entropy is
the point: a predictable setup leaks information and is exploitable.

Ours: replace the setup network with a **large static team pool** with a tabular
distribution over it, updated by Exp3 from Monte Carlo game outcomes. Notes:
- Exp3's importance-weighted reward `r / p(team)` is the same 1/p estimator a policy
  gradient uses implicitly (∇log π = ∇π / π). Variance is high for rarely sampled teams,
  so expect many games per team.
- **Last-iterate cycling:** in two-player zero-sum self-play, plain Exp3's *average*
  converges, but its *current* distribution can cycle. This is the problem the paper's
  damping solves. Options: an entropy- or magnet-regularized tabular update with an annealed
  coefficient (the tabular version of what the paper does), or sample from the
  time-averaged distribution for evaluation and belief priors.
- Rewards are non-stationary, since the move network keeps improving. Exp3 is an
  adversarial bandit, so that's fine in principle, but its step size needs a schedule.
- Gen 3 has **no team preview**, so the team is hidden info like a Stratego setup. The
  lead is fixed by team order, so decide whether pool entries include lead variants.

## Belief network

Paper: a separate encoder–decoder transformer (57M params) that predicts the opponent's
hidden piece types from the information state. It is trained with teacher forcing and NLL
on the true hidden types, using all positions from self-play of the *final* networks,
unfiltered. It is used only to sample hidden states for search.

*(ours)* Predict the opponent's team **by component** (species, item, ability, moves,
nature, EVs), the way the paper predicts per-piece types. Do **not** output a categorical
over pool indices: that can only produce pool teams, and test-time opponents bring teams
the pool has never seen (pool-consistency filtering is worse: unusual revealed facts can
rule out every entry and leave an empty posterior). Train on self-play of the final move
net with teams drawn from the **time-averaged** Exp3 distribution. A learned net beats
filtering because observed *behavior* (not just revealed facts) is evidence too. Token
scheme, decode order, masking and sizing: see `docs/BRIDGE.md`.

## Test-time search

At each decision (no search for setups/teams):
1. Sample hidden states from the belief network: ~1,000 / (number of legal actions) samples.
2. For each (legal action, sample) pair, run ≥1 rollout. The first action is forced, the
   next 39 plies use the move network, and the endpoint is scored by the value head. The
   total budget is 1,000 rollouts. `q̂(a)` is the mean.
3. Apply the closed-form regularized update and sample the move from it:
   ```
   π_search = argmax ⟨q̂, π⟩ − α·KL(π ‖ ρ) − β·KL(π ‖ π_θ)
            ∝ [ e^{q̂} · ρ^α · π_θ^β ]^{1/(α+β)}      α = 0.002, β = 0.02
   ```
This is "update equivalence": training applies the same regularized update using a
one-sample (1/p) estimate per state, and search applies it with all-action Q estimates
at a single state, tabularly. It inherits improvement guarantees, but gains are capped at
roughly "one more learning step."

*(ours)* Pokémon is **simultaneous-move and stochastic**. At the root, the opponent's
action comes from the move network on its sampled information state, and chance (damage
rolls, crits, speed ties, secondary effects) comes from the simulator's RNG in each
rollout. The update itself only needs `q̂` for our own actions, so nothing else changes.
Rollout depth should be counted in turns.

## Architecture notes

Paper: the move network is an encoder-only transformer (8 layers, d=384, 14.7M params).
Tokens are the 92 occupiable squares plus a value token, with learned absolute positional
embeddings. The policy is a key–query product over (from, to) square pairs, which "learned
faster than less sophisticated parameterizations." It is an inductive bias, not extra
expressiveness. A per-square move-type output head could represent the same policies.

*(ours)* ADV OU has at most 9 actions (4 moves + 5 switches). A plain head is fine, or
query/key scoring of the active Pokémon against move tokens and bench tokens. Natural
tokens: 12 Pokémon (side + active/bench embeddings; team slots are a set, so avoid
slot-order positional embeddings), field/side-condition tokens, and a value token.
- **Drop steps with exactly one legal action** (forced switches, locked moves) from
  training. They carry no policy signal.
- **Magnet ρ** analogous to the paper's: uniform over {attack, switch}, then uniform within.
  Otherwise a full bench biases the reference toward switching.
- Games are tens of turns, not hundreds to thousands, so the paper's low λs (chosen for
  very long games) may be worth raising. Monte Carlo is more viable here.

## Open questions *(ours)*
- Pool size vs. Exp3 sample cost, and whether to regularize/anneal the team distribution
  or use its time average.
- Belief/setup token-scheme choices (crossing, EV bucketing, decode order): see `docs/BRIDGE.md`.
- Team-selection lead variants.
- Simulator throughput, which sets envs/iteration and rollout budget.
