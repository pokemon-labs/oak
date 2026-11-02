"""Ataraxos's test-time search (the paper's, arXiv 2511.07312; the user's
design for Gen 1, 2026-11-01): at a root decision of ours, `worlds` history
samples (src/history_sampler.hpp) and, in each, `reps` rollouts per legal
choice (src/rollouts.hpp), worlds * reps * n in all; `budget` sets worlds
as budget / (reps * n) when `worlds` is 0. A rollout plays our choice,
then both sides' nets for `depth` turns, and scores our next decision with
the value head, P(win) - P(loss) (a finished game is +1 / 0 / -1). q is
each choice's mean, and the move is drawn from the closed-form update

    pi_search ∝ [e^q rho^alpha pi^beta]^(1 / (alpha + beta))

(rho the training magnet, pi the net's policy at the root).

The nets' caches: ours is the real game's, copied from the actor that
played the root (after it ran on the root's summary, so a rollout's first
decision of ours is forced and not run again); the foe's is its rebuilt
history, run once per world (`prefill`) and copied to that world's
rollouts. Many roots search together: every decision of the rollouts is
one batch per step.

The joint search (`joint=True`; ataraxos/CLAUDE.md "Joint search", the
user's design 2026-11-02): a Bayesian game over `types_a` of our types (A0
our real team, the rest guessed from what the foe has seen of us) and
`types_b` of the foe's, uniform weights. Each type's cache is its own
history run once, its root decision included (that gives its policy pi,
the magnet with rho; A0's is the real actor's). Rollouts force both root
choices, cell (A, B, i, j), and come in `rounds`: sample the cells by the
current policies (with a `floor` of uniform), keep each cell's mean
Q_AB(i, j), and solve by MMD on both sides to its fixed point, the
KL-regularized equilibrium (alpha + beta the regularization, the magnet
rho^alpha pi^beta; the foe's beta `foe_beta`). With the foe fixed it is the
closed form above. `per_type`: one policy a type, q1_A(i) = sum_B P(B)
sum_j Q_AB(i, j) p2_B(j), q2_B(j) = -sum_A P(A) sum_i Q_AB(i, j) p1_A(i);
we play p1_A0. `per_pair`: an independent game per (A, B); we play sum_B
P(B) p1_{A0,B}. A cell with no rollout yet reads its row and column means
(its block's mean, else 0), and a thin cell is shrunk to that by `shrink`
pseudo-counts.
"""

import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from ataraxos._ext import ataraxos_env
from ataraxos.eval.agents import subset
from ataraxos.nets.causal_actor import CausalActor
from ataraxos.nets.causal_net import summarize_grouped
from ataraxos.nets.policy import pad_chunks
from ataraxos.rl.train import magnet


@dataclass
class SearchConfig:
    worlds: int = 0      # history samples per root; 0: from the budget
    reps: int = 2        # rollouts per (world, choice)
    depth: int = 10      # turns before the leaf
    budget: int = 128    # rollouts per root, with worlds 0
    alpha: float = 0.002  # the magnet's coefficient
    beta: float = 0.02    # the net policy's
    magnet: str = "attack_switch"
    # the joint search
    joint: bool = False
    joint_mode: str = "per_type"  # or "per_pair"
    types_a: int = 3       # ours, A0 our real team
    types_b: int = 3       # the foe's
    joint_budget: int = 512  # rollouts per root over all rounds
    rounds: int = 4
    floor: float = 0.25    # the uniform share of each round's cell sampling
    shrink: float = 1.0    # pseudo-counts toward a cell's row and column means
    foe_beta: float = -1.0  # the foe's beta; < 0: ours
    eta: float = 0.0       # MMD's step; 0: 1 (extragradient), else lam / 4
    extra: bool = True     # extragradient MMD (plain MMD cycles at big steps)
    mmd_iters: int = 2000
    mmd_tol: float = 1e-5
    act_rows: int = 1024   # the rollouts' decisions per net call (the activations' memory)


def act_chunked(actor, obs, rows):
    """actor.act on `obs` in chunks of at most `rows` decisions (each key's
    cache is written once, so the result is act's)."""
    D = len(obs["n"])
    if D <= rows:
        return actor.act(obs)
    parts = [actor.act(subset(obs, np.arange(i, min(i + rows, D)))) for i in range(0, D, rows)]
    return (np.concatenate([p[0] for p in parts]), torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]))


def win_minus_loss(value_logits: torch.Tensor) -> torch.Tensor:
    """P(win) - P(loss) from the value head's (loss, draw, win) logits."""
    p = torch.softmax(value_logits.float(), -1)
    return p[:, 2] - p[:, 0]


def update(q, logp, choices, n, cfg: SearchConfig):
    """The closed-form update's log-probs [CHOICES] for one root: q [n], the
    net's log-probs [CHOICES] and the root's choices [CHOICES]."""
    legal = torch.arange(len(choices), device=logp.device) < n
    rho = magnet(choices[None].long(), legal[None], cfg.magnet)[0]
    qq = torch.zeros_like(logp)
    qq[:n] = torch.as_tensor(q, dtype=logp.dtype, device=logp.device)
    x = (qq + cfg.alpha * rho + cfg.beta * logp.masked_fill(~legal, 0)) / (cfg.alpha + cfg.beta)
    return torch.log_softmax(x.masked_fill(~legal, -1e9), -1)


def mmd(Q, w1, w2, m1, m2, lam1, lam2, legal1, legal2, l1=None, l2=None, eta=None, iters=2000, tol=1e-4,
        extra=True):
    """MMD on both sides of G Bayesian games to the fixed point: Q [G, a, b,
    N, M] (ours, means), type weights w1 [a], w2 [b], magnets' log-probs m1
    [G, a, N], m2 [G, b, M] (rho^alpha pi^beta over lam), regularization
    lam1, lam2, legal masks like m1, m2. Starts from log-policies l1, l2
    (default the magnets). Returns (l1, l2, q1, q2, iterations): q1 [G, a,
    N] = sum_B w2 sum_j Q p2, q2 [G, b, M] = -sum_A w1 sum_i Q p1. Each step
    l <- (l + eta q + eta lam m) / (1 + eta lam), normalized. `extra`
    (default): extragradient (mirror-prox) steps, q read at a half step:
    plain MMD needs eta <= lam / L^2 to settle in a zero-sum game (it
    cycles at larger steps), extragradient only eta < 1 / L (L ~ max |Q|),
    with the same fixed point."""
    neg = -1e9

    def norm(l, legal):
        l = np.where(legal, l, neg)
        return l - np.logaddexp.reduce(l, -1, keepdims=True)

    m1, m2 = norm(m1, legal1), norm(m2, legal2)
    l1 = m1.copy() if l1 is None else norm(l1, legal1)
    l2 = m2.copy() if l2 is None else norm(l2, legal2)
    eta = (1.0 if extra else min(lam1, lam2) / 4) if not eta else eta
    Qw1 = Q * w1[None, :, None, None, None]
    Qw2 = Q * w2[None, None, :, None, None]

    def step(a1, a2, at1, at2):
        """MMD's step from (a1, a2) with q read at (at1, at2)."""
        q1 = np.einsum("gabij,gbj->gai", Qw2, np.exp(at2))
        q2 = -np.einsum("gabij,gai->gbj", Qw1, np.exp(at1))
        return (norm((a1 + eta * q1 + eta * lam1 * m1) / (1 + eta * lam1), legal1),
                norm((a2 + eta * q2 + eta * lam2 * m2) / (1 + eta * lam2), legal2))

    it = 0
    for it in range(1, iters + 1):
        p1, p2 = np.exp(l1), np.exp(l2)
        h1, h2 = step(l1, l2, l1, l2)
        n1, n2 = step(l1, l2, h1, h2) if extra else (h1, h2)
        d = max(np.abs(np.exp(n1) - p1).max(), np.abs(np.exp(n2) - p2).max())
        l1, l2 = n1, n2
        if d < tol:
            break
    p1, p2 = np.exp(l1), np.exp(l2)
    q1 = np.einsum("gabij,gbj->gai", Qw2, p2)
    q2 = -np.einsum("gabij,gai->gbj", Qw1, p1)
    return l1, l2, q1, q2, it


def cell_means(total, count, legal, shrink):
    """Q [a, b, N, M] from the cells' value sums and counts: an empty cell
    reads the mean of its row's and its column's means in its (A, B) block
    (one of them if the other is empty, else the block's, else 0), and
    every cell is shrunk toward that by `shrink` pseudo-counts."""
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nanmean of two NaNs
        rs, rc = total.sum(3, keepdims=True), count.sum(3, keepdims=True)
        cs, cc = total.sum(2, keepdims=True), count.sum(2, keepdims=True)
        bs, bc = total.sum((2, 3), keepdims=True), count.sum((2, 3), keepdims=True)
        block = np.where(bc > 0, bs / bc, 0.0)
        row = np.where(rc > 0, rs / rc, np.nan)
        col = np.where(cc > 0, cs / cc, np.nan)
        prior = np.nanmean(np.stack(np.broadcast_arrays(row, col)), 0)
        prior = np.where(np.isnan(prior), block, prior)
        q = (total + shrink * prior) / (count + shrink) if shrink > 0 else np.where(count > 0, total / count, prior)
    return np.where(legal, q, 0.0)


def gather_rows(obs, rows, keys=("request", "state", "choices", "n")):
    """Rows `rows` of a decisions dict (tokens by offsets)."""
    off = obs["offsets"]
    parts = [obs["tokens"][off[r]:off[r + 1]] for r in rows]
    lens = np.array([len(x) for x in parts], np.int64)
    out = {k: obs[k][rows] for k in keys if k in obs}
    out["tokens"] = np.concatenate(parts) if len(parts) else obs["tokens"][:0]
    out["offsets"] = np.r_[0, np.cumsum(lens)].astype(np.int64)
    return out


class Search:
    """`teams`: the history sampler's pool (Showdown export files); `ratio`
    its rank-decay prior."""

    def __init__(self, net, teams, device, cfg: SearchConfig = SearchConfig(), threads=8, amp=True,
                 ratio=0.1, seed=0):
        self.net, self.device, self.cfg, self.amp = net, torch.device(device), cfg, amp
        self.r = ataraxos_env.Rollouts(list(teams), net.c.window - 1, threads=threads, state=net.c.state,
                                       ratio=ratio)
        self.actor = None
        self.rng = np.random.default_rng(seed)
        self.stats = {"roots": 0, "rollouts": 0, "steps": 0, "differs": 0}
        self.times = {"sample": 0.0, "prefill": 0.0, "rollouts": 0.0, "update": 0.0}

    def worlds(self, n):
        c = self.cfg
        return c.worlds or max(1, c.budget // (c.reps * n))

    def _actor(self, rollouts):
        if self.actor is None or self.actor.slots < 2 * rollouts:
            self.actor = None
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
            self.actor = CausalActor(self.net, max(rollouts, 64), self.device, amp=self.amp)
        return self.actor

    @torch.no_grad()
    def prefill(self, actor, keys, lay):
        """Runs the foe's summaries of each world into its cache at `keys`
        [W] (world w's first rollout's foe key): summary j of a world at
        position t0 + j."""
        net, dev, R = self.net, self.device, actor.R
        summaries, off = lay["summaries"], lay["offsets"]
        count = np.diff(summaries)
        t0 = torch.from_numpy(lay["t0"]).to(dev)
        keys_t = torch.from_numpy(keys).to(dev)
        actor.game[keys_t] = 0
        actor.t[keys_t] = t0
        tokens = torch.from_numpy(lay["tokens"])
        for j in range(int(count.max()) if len(count) else 0):
            w = np.flatnonzero(count > j)
            s = summaries[w] + j  # each world's summary j
            k = torch.from_numpy(keys[w]).to(dev)
            o = torch.from_numpy(np.stack([off[s], off[s + 1]], 1).reshape(-1))
            # the chunks of rows s, as pad_chunks reads consecutive offsets
            lens = (o[1::2] - o[0::2])
            flat = torch.cat([tokens[a:b] for a, b in zip(o[0::2].tolist(), o[1::2].tolist())]) \
                if lens.sum() else tokens[:0]
            offs = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lens, 0)])
            chunks = pad_chunks(flat, offs, net.c.max_lines)
            lengths = np.minimum(lens.numpy(), net.c.max_lines)
            request = torch.from_numpy(lay["request"][s]).to(dev)
            t = actor.t[k]
            with torch.autocast(dev.type, dtype=actor.dtype, enabled=actor.amp):
                x = summarize_grouped(net, chunks, lengths, request, actor.k.dtype)
                x = x + net.pos[t.clamp(max=net.c.max_pos - 1)]
                valid = torch.arange(R, device=dev)[None] < t[:, None]
                for l, layer in enumerate(net.layers):
                    x, kl, vl = layer.step(x, actor.k[l, k], actor.v[l, k], valid)
                    actor.k[l, k, t % R] = kl.to(actor.k.dtype)
                    actor.v[l, k, t % R] = vl.to(actor.v.dtype)
            actor.t[k] = t + 1

    def run(self, env, real: CausalActor, obs, logp, seed=None, truth=False, check=None):
        """Searches decisions `obs` (rows of the SelfPlay `env`'s output, all
        ours, the real actor having just run on them; `logp` [D, CHOICES]
        its log-probs) and returns (picks u8 [D], q list, search log-probs
        [D, CHOICES]). `truth`: every world has the foe's true sets. `check`
        (tests, a dict): fills in "ours", the copied cache rerun on the
        root summary against `logp` (max abs), and "foe", per root whose
        rollouts start with a foe decision: (root, the mean of their first
        foe log-probs, that decision's request, choices and n)."""
        c = self.cfg
        keys = obs["battle"].astype(np.int64) * 2 + obs["player"]
        if c.joint:
            def add_joint(r, i, seed):
                r.add_joint_selfplay(env, int(obs["battle"][i]), int(obs["player"][i]), c.types_a, c.types_b,
                                     c.depth, seed, truth)

            return self.search_joint(add_joint, real, keys, obs["player"], obs, logp, seed, check)

        def add(r, i, seed):
            r.add_selfplay(env, int(obs["battle"][i]), int(obs["player"][i]), self.worlds(int(obs["n"][i])),
                           c.reps, c.depth, seed, truth)

        return self.search(add, real, keys, obs["player"], obs, logp, seed, check)

    def run_client(self, client, real: CausalActor, key, obs, logp, seed=None):
        """One search at the ladder Client's request: `obs` its one decision,
        which the real actor ran at cache key `key`."""
        c = self.cfg
        keys, players = np.array([key], np.int64), np.array([client.viewer()])
        if c.joint:
            def add_joint(r, i, seed):
                r.add_joint_client(client, c.types_a, c.types_b, c.depth, seed)

            return self.search_joint(add_joint, real, keys, players, obs, logp, seed)

        def add(r, i, seed):
            r.add_client(client, self.worlds(int(obs["n"][0])), c.reps, c.depth, seed)

        return self.search(add, real, keys, players, obs, logp, seed)

    @torch.no_grad()
    def search(self, add, real, keys, players, obs, logp, seed=None, check=None):
        """`add(rollouts, i, seed)` adds root i; `keys` [D] the roots' keys
        in the real actor's cache, `players` [D] their sides."""
        D = len(obs["n"])
        seed = int(self.rng.integers(1 << 62)) if seed is None else seed
        r = self.r
        tic = time.perf_counter()

        def lap(what):
            nonlocal tic
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            self.times[what] += now - tic
            tic = now

        r.clear()
        for i in range(D):
            add(r, i, seed + i)
        lay = r.layout()
        N = r.rollouts
        lap("sample")
        actor = self._actor(N)
        dev = self.device
        world_of = lay["world"].astype(np.int64)
        root_of = lay["root"].astype(np.int64)[world_of]
        players = np.asarray(players, np.int64)
        us = players[root_of]
        idx = np.arange(N, dtype=np.int64)
        ours = torch.from_numpy(idx * 2 + us).to(dev)
        foes = torch.from_numpy(idx * 2 + 1 - us).to(dev)
        # ours: the real game's cache
        src = torch.from_numpy(np.asarray(keys, np.int64)[root_of]).to(real.device)
        actor.k[:, ours] = real.k[:, src].to(dev, actor.k.dtype)
        actor.v[:, ours] = real.v[:, src].to(dev, actor.v.dtype)
        actor.t[ours] = real.t[src].to(dev)
        actor.game[ours] = 0
        # the foe's: each world's history, then copied to its rollouts
        _, first = np.unique(world_of, return_index=True)
        first_keys = idx[first] * 2 + 1 - us[first]
        self.prefill(actor, first_keys, lay)
        fk = torch.from_numpy(first_keys[world_of]).to(dev)
        actor.k[:, foes] = actor.k[:, fk]
        actor.v[:, foes] = actor.v[:, fk]
        actor.t[foes] = actor.t[fk]
        actor.game[foes] = 0
        if check is not None:
            # the root summary again, on each root's first rollout's copy
            _, root_first = np.unique(root_of, return_index=True)
            again = dict(obs)
            again["battle"] = idx[root_first].astype(np.uint32)
            again["player"] = us[root_first].astype(np.uint8)
            again["game"] = np.zeros(D, np.uint32)
            k = torch.from_numpy(idx[root_first] * 2 + us[root_first]).to(dev)
            t = actor.t[k].clone()
            actor.t[k] = t - 1
            _, lp_again, _ = actor.act(again)
            actor.t[k] = t
            legal = torch.arange(logp.shape[1], device=logp.device)[None] < torch.from_numpy(obs["n"]).to(logp.device)[:, None]
            check["ours"] = float((lp_again.to(logp.device) - logp).abs().masked_fill(~legal, 0).max())
        lap("prefill")

        out = r.start(seed)
        steps = 0
        first = check is not None
        while len(out["n"]):
            picks, lp_r, value = act_chunked(actor, out, self.cfg.act_rows)
            if first:
                first = False
                foe_rows = np.flatnonzero(out["player"] != players[out["root"]])
                check["foe"] = []
                for i in range(D):
                    rows = foe_rows[out["root"][foe_rows] == i]
                    if len(rows):
                        lq = lp_r[torch.from_numpy(rows).to(dev)].mean(0)
                        check["foe"].append((i, lq, out["request"][rows[0]], out["choices"][rows[0]], out["n"][rows[0]]))
            out = r.step(picks, win_minus_loss(value).cpu().numpy().astype(np.float32))
            steps += 1
        assert r.done
        lap("rollouts")

        picks = np.zeros(D, np.uint8)
        qs, lps = [], torch.empty_like(logp)
        choices = torch.from_numpy(np.ascontiguousarray(obs["choices"])).to(logp.device)
        for i in range(D):
            q = r.q(i)
            lp = update(q, logp[i], choices[i], int(obs["n"][i]), self.cfg)
            picks[i] = torch.multinomial(lp.exp(), 1).item()
            qs.append(q)
            lps[i] = lp
        lap("update")
        s = self.stats
        s["roots"] += D
        s["rollouts"] += N
        s["steps"] += steps
        s["differs"] += int(lay["differs"].sum())
        return picks, qs, lps

    def _rollout_loop(self, actor, r, seed):
        """Runs the rollouts started from `r.start(seed)` to their leaves;
        returns the steps."""
        out = r.start(seed)
        steps = 0
        while len(out["n"]):
            picks, _, value = act_chunked(actor, out, self.cfg.act_rows)
            out = r.step(picks, win_minus_loss(value).cpu().numpy().astype(np.float32))
            steps += 1
        assert r.done
        return steps

    @torch.no_grad()
    def search_joint(self, add_joint, real, keys, players, obs, logp, seed=None, check=None):
        """The joint search (the module's docstring) of roots `obs`, as
        `search`: `add_joint(rollouts, i, seed)` adds joint root i. Returns
        (picks, q list (q1 of A0), search log-probs [D, CHOICES]). `check`
        (tests, a dict) gets "types" (each type's root log-probs), "layout"
        (joint_layout's), "Q" and
        "count" per root, "iters" (MMD's, per round) and "pol" ((l1, l2) per
        root)."""
        c = self.cfg
        D = len(obs["n"])
        seed = int(self.rng.integers(1 << 62)) if seed is None else seed
        r, dev = self.r, self.device
        C = logp.shape[1]
        tic = time.perf_counter()

        def lap(what):
            nonlocal tic
            if dev.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            self.times[what] = self.times.get(what, 0.0) + now - tic
            tic = now

        r.clear()
        for i in range(D):
            add_joint(r, i, seed + i)
        lay = r.joint_layout()
        lap("sample")

        Y = len(lay["n"])
        per_round = max(1, c.joint_budget // c.rounds)
        rmax = D * per_round
        # rollout i's keys are 2i, 2i + 1; type y's 2 rmax + y
        actor = self._actor(rmax + (Y + 1) // 2)
        tkeys = 2 * rmax + np.arange(Y, dtype=np.int64)
        players = np.asarray(players, np.int64)
        first, na, nb = lay["first"].astype(np.int64), lay["a"].astype(np.int64), lay["b"].astype(np.int64)

        # the types' caches: their histories, then A0's the real actor's
        self.prefill(actor, tkeys, lay)
        real_y = np.flatnonzero(lay["real"])
        src = torch.from_numpy(np.asarray(keys, np.int64)[lay["root"][real_y].astype(np.int64)]).to(real.device)
        k = torch.from_numpy(tkeys[real_y]).to(dev)
        actor.k[:, k] = real.k[:, src].to(dev, actor.k.dtype)
        actor.v[:, k] = real.v[:, src].to(dev, actor.v.dtype)
        actor.t[k] = real.t[src].to(dev)
        actor.game[k] = 0
        # each type's policy at the root: A0's the real one's, a deciding
        # type's from its root decision, else its one choice
        choices = torch.from_numpy(np.ascontiguousarray(lay["choices"])).to(dev)
        n_t = torch.from_numpy(lay["n"].astype(np.int64)).to(dev)
        legal_t = torch.arange(C, device=dev)[None] < n_t[:, None]
        lp_t = torch.where(legal_t, 0.0, -1e9).to(torch.float32)
        lp_t[:, 0] = 0.0
        lp_t = torch.log_softmax(lp_t, -1)
        dec = np.flatnonzero((lay["decides"] > 0) & (lay["real"] == 0))
        if len(dec):
            sub = {"tokens": lay["dec_tokens"], "offsets": lay["dec_offsets"], "request": lay["dec_request"],
                   "choices": lay["choices"], "n": lay["n"]}
            if "dec_state" in lay:
                sub["state"] = lay["dec_state"]
            sub = gather_rows(sub, dec)
            sub["battle"] = (rmax + dec // 2).astype(np.uint32)
            sub["player"] = (dec % 2).astype(np.uint8)
            sub["game"] = np.zeros(len(dec), np.uint32)
            _, lp, _ = actor.act(sub)
            lp_t[torch.from_numpy(dec).to(dev)] = lp.float()
        lp_t[torch.from_numpy(real_y).to(dev)] = logp[lay["root"][real_y].astype(np.int64)].to(dev).float()
        rho = magnet(choices.long(), legal_t, c.magnet)
        lp_np = lp_t.cpu().numpy().astype(np.float64)
        rho_np = rho.float().cpu().numpy().astype(np.float64)
        legal_np = legal_t.cpu().numpy()
        lap("prefill")

        beta2 = c.beta if c.foe_beta < 0 else c.foe_beta
        lam1, lam2 = c.alpha + c.beta, c.alpha + beta2
        # per root: the types' magnets and legal masks, the cells' sums
        roots = []
        for i in range(D):
            A, B = int(na[i]), int(nb[i])
            ya, yb = first[i] + np.arange(A), first[i] + A + np.arange(B)
            mag1 = (c.alpha * rho_np[ya] + c.beta * lp_np[ya]) / lam1
            mag2 = (c.alpha * rho_np[yb] + beta2 * lp_np[yb]) / lam2
            roots.append(dict(A=A, B=B, ya=ya, yb=yb, m1=mag1, m2=mag2, g1=legal_np[ya], g2=legal_np[yb],
                              total=np.zeros((A, B, C, C)), count=np.zeros((A, B, C, C)),
                              l1=None, l2=None))
        iters = []
        steps = 0
        rng = np.random.default_rng(seed)
        for rd in range(c.rounds):
            # sample this round's cells by the current policies
            cells = []
            for i, x in enumerate(roots):
                p1 = np.exp(self._marginal(x, "l1", "m1", "g1"))  # [A, C]
                p2 = np.exp(self._marginal(x, "l2", "m2", "g2"))  # [B, C]
                u1 = x["g1"] / x["g1"].sum(-1, keepdims=True)
                u2 = x["g2"] / x["g2"].sum(-1, keepdims=True)
                s1 = (1 - c.floor) * p1 + c.floor * u1
                s2 = (1 - c.floor) * p2 + c.floor * u2
                prob = (s1[:, None, :, None] * s2[None, :, None, :]) / (x["A"] * x["B"])
                counts = rng.multinomial(per_round, prob.reshape(-1) / prob.sum()).reshape(prob.shape)
                for a, b, ci, cj in zip(*np.nonzero(counts)):
                    cells.append((i, a, b, ci, cj, counts[a, b, ci, cj]))
            r.clear_rollouts()
            r.schedule(np.array(cells, np.int64).reshape(-1, 6))
            vals = r.joint_values()
            N = len(vals["root"])
            # the rollouts' caches from their types'
            root_of = vals["root"].astype(np.int64)
            us = players[root_of]
            idx = np.arange(N, dtype=np.int64)
            ours = torch.from_numpy(idx * 2 + us).to(dev)
            foes = torch.from_numpy(idx * 2 + 1 - us).to(dev)
            ka = torch.from_numpy(tkeys[first[root_of] + vals["a"]]).to(dev)
            kb = torch.from_numpy(tkeys[first[root_of] + na[root_of] + vals["b"]]).to(dev)
            for dst, sk in ((ours, ka), (foes, kb)):
                actor.k[:, dst] = actor.k[:, sk]
                actor.v[:, dst] = actor.v[:, sk]
                actor.t[dst] = actor.t[sk]
                actor.game[dst] = 0
            lap("prefill")
            steps += self._rollout_loop(actor, r, seed + 1 + rd)
            lap("rollouts")
            vals = r.joint_values()
            self.stats["differs"] += int(vals["differs"].sum())
            self.stats["rollouts"] += N
            for i, x in enumerate(roots):
                m = vals["root"] == i
                at = (vals["a"][m], vals["b"][m], vals["action"][m], vals["foe_action"][m])
                np.add.at(x["total"], at, vals["value"][m])
                np.add.at(x["count"], at, 1)
                it = self._solve(x, lam1, lam2)
                iters.append(it)
            lap("update")

        picks = np.zeros(D, np.uint8)
        qs, lps = [], torch.empty_like(logp)
        for i, x in enumerate(roots):
            l1 = self._marginal(x, "l1", "m1", "g1")[0]  # A0
            n = int(obs["n"][i])
            lp = torch.from_numpy(l1).to(logp.device, logp.dtype)
            lp = torch.log_softmax(lp.masked_fill(torch.arange(C, device=logp.device) >= n, -1e9), -1)
            picks[i] = torch.multinomial(lp.exp(), 1).item()
            qs.append(x["q1"][:n])
            lps[i] = lp
        self.stats["roots"] += D
        self.stats["steps"] += steps
        if check is not None:
            check["types"] = lp_np
            check["layout"] = lay
            check["Q"] = [cell_means(x["total"], x["count"], x["g1"][:, None, :, None] & x["g2"][None, :, None, :],
                                     c.shrink) for x in roots]
            check["count"] = [x["count"] for x in roots]
            check["iters"] = iters
            check["pol"] = [(x["l1"], x["l2"]) for x in roots]
        lap("update")
        return picks, qs, lps

    def _marginal(self, x, l, m, g):
        """A root's current log-policy a type [types, C] (per_pair: our A's
        is sum_B P(B) p1_AB, the foe's B's sum_A P(A) p2_AB); the magnet
        before any solve."""
        if x[l] is None:
            return np.where(x[g], x[m] - np.logaddexp.reduce(np.where(x[g], x[m], -1e9), -1, keepdims=True), -1e9)
        if self.cfg.joint_mode == "per_type":
            return x[l][0]
        A, B = x["A"], x["B"]
        p = np.exp(x[l]).reshape(A, B, -1)  # [A * B, 1, C] per pair
        p = p.mean(1) if l == "l1" else p.mean(0)
        return np.log(np.maximum(p, 1e-300))

    def _solve(self, x, lam1, lam2):
        """MMD on a root's current Q (warm-started); keeps l1, l2 and q1
        (A0's: per_pair, sum_B P(B) of its pairs')."""
        c = self.cfg
        A, B = x["A"], x["B"]
        legal = x["g1"][:, None, :, None] & x["g2"][None, :, None, :]
        Q = cell_means(x["total"], x["count"], legal, c.shrink)
        if c.joint_mode == "per_type":
            l1, l2, q1, _, it = mmd(Q[None], np.full(A, 1 / A), np.full(B, 1 / B), x["m1"][None], x["m2"][None],
                                   lam1, lam2, x["g1"][None], x["g2"][None], x["l1"], x["l2"], c.eta, c.mmd_iters,
                                   c.mmd_tol, c.extra)
            x["q1"] = q1[0, 0]
        elif c.joint_mode == "per_pair":
            G = A * B
            m1 = np.repeat(x["m1"], B, 0)[:, None]  # pair (a, b) = a * B + b
            m2 = np.tile(x["m2"], (A, 1))[:, None]
            g1 = np.repeat(x["g1"], B, 0)[:, None]
            g2 = np.tile(x["g2"], (A, 1))[:, None]
            l1, l2, q1, _, it = mmd(Q.reshape(G, 1, 1, *Q.shape[2:]), np.ones(1), np.ones(1), m1, m2, lam1, lam2,
                                   g1, g2, x["l1"], x["l2"], c.eta, c.mmd_iters, c.mmd_tol, c.extra)
            x["q1"] = q1[:B, 0].mean(0)
        else:
            raise ValueError(c.joint_mode)
        x["l1"], x["l2"] = l1, l2
        return it
