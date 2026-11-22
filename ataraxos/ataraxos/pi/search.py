"""Search in training for the perfect-info trainer (2026-11-22, the user's
design; settings from tests/py/search_gain.py; docs/pi_speed.md).

KataGo-style: each env step, a share p of the rows where either side has 2+
choices is searched (p annealed by the trainer). A searched row (a root):

  - `budget` depth-1 rollouts (SelfPlay.rollouts: a fresh RNG seed and
    hidden counters each, one (P1, P2) choice pair, the forced steps, then
    the leaf), the pairs drawn from the net's priors, with `floor` of them
    uniform over the legal pairs (search_gain.py's 0.25);
  - the leaves' value: the net's value head (P1's W/D/L), or the result's
    one-hot where the game ended; a leaf's Q is P(win) - P(loss);
  - cell means (search_np.cell_means, `shrink` pseudo-counts toward the
    row / column means), then one round of joint MMD to the fixed point
    (ataraxos_pi.mmd, extragradient) with the prior as both sides' magnet
    (alpha 0) and lambda `beta`;
  - both sides play a pick drawn from their posterior; the targets kept are
    both posteriors and the W/D/L under them (sum p1_i p2_j W_ij).

The trainer stores them per row (`searched`, `post`, `swdl`); the learner
uses the posteriors as cross-entropy targets, masks searched decisions out
of the PPO loss and mixes `swdl` into their value targets.

Speed options (each default the original path; the trainer's --search_*):
  - `leaf`: "eager" (numpy to the device per chunk), "async" (pinned slots,
    non-blocking copies), "graphs" (async + a CUDA graph per power-of-2
    leaf count, padded);
  - `compile`: torch.compile of the leaf forward (dynamic shapes);
  - `value_only`: PiNet.value (no policy heads) for the leaves;
  - `direct` (leaf async|graphs): rollouts write straight into the pinned
    slot (SelfPlay.rollouts_into);
  - begin / finish: `run` is finish(begin(...)); the trainer's --search_async
    calls them apart so one env's leaf forward runs while the other steps;
  - `plan` "cpp": the cells drawn in C++ (ataraxos_pi.plan; its own RNG, so
    not the numpy draws);
  - `solve` "cpp": cell means, MMD and the W/D/L in C++ per root
    (ataraxos_pi.solve; numpy's cell_means and the einsum otherwise).
"""

import time

import numpy as np
import torch

from ataraxos._ext import ataraxos_pi
from ataraxos.pi.net import CHOICES
from ataraxos.search_np import cell_means

C = CHOICES
LEAF_KEYS = ("mon", "hp", "active")


def sample(rng, probs, n):
    """One index per row of probs [..., C] (inverse CDF), clipped below n."""
    cdf = np.cumsum(probs, -1)
    u = rng.random(probs.shape[:-1])[..., None] * cdf[..., -1:]
    i = (u > cdf).sum(-1)
    return np.minimum(i, np.maximum(n, 1) - 1)


def done_onehot(wdl, done, result):
    d = np.flatnonzero(done)
    wdl[d] = 0.0
    wdl[d, result[d].astype(np.int64) + 1] = 1.0
    return wdl


class LeafEval:
    """The leaves' P1 (loss, draw, win) probabilities. `launch(slot, K,
    leaves=None)` queues slot's K leaves (from `leaves`, or already written
    into `inputs(slot, K)` with direct), `collect(handle, done, result)`
    waits and returns [K, 3] with the finished games' one-hots."""

    def __init__(self, net, device, amp, mode="eager", compile=False, value_only=False, slots=2, chunk=65536,
                 min_bucket=4096):
        if mode not in ("eager", "async", "graphs"):
            raise ValueError(f"leaf mode {mode!r}: eager, async or graphs")
        self.net, self.dev, self.amp, self.mode = net, torch.device(device), amp, mode
        self.value_only, self.chunk, self.min_bucket = value_only, chunk, min_bucket
        self.cuda = self.dev.type == "cuda"
        if mode == "graphs" and not self.cuda:
            raise ValueError("leaf graphs needs CUDA")
        self._fwd = self._fwd_eager if not compile else torch.compile(self._fwd_eager, dynamic=True)
        self.cap = 0
        self.pin = [None] * slots      # per slot: dict of pinned inputs [cap, ...]
        self.pout = [None] * slots     # per slot: pinned output [cap, 3]
        self.events = [torch.cuda.Event() if self.cuda else None for _ in range(slots)]
        self.din = None                # device inputs [cap, ...]
        self.dout = None               # device output [cap, 3]
        self.graphs = {}               # bucket -> CUDA graph

    def _fwd_eager(self, mon, hp, active):
        with torch.autocast(self.dev.type, torch.bfloat16, enabled=self.amp and self.cuda, cache_enabled=False):
            value = self.net.value(mon.long(), hp, active) if self.value_only else \
                self.net(mon.long(), hp, active)[0]
        return torch.softmax(value.float(), -1)

    def _grow(self, K):
        """Buffers for at least K rows (a power of 2); a regrowth drops the graphs."""
        if K <= self.cap:
            return
        cap = max(self.min_bucket, 1 << (K - 1).bit_length())
        shapes = dict(mon=((2, 6, ataraxos_pi.MON_FIELDS), torch.int16), hp=((2, 6), torch.float32),
                      active=((2, ataraxos_pi.ACTIVE_DIM), torch.float32))
        mk = lambda dev, pin: {k: torch.zeros((cap,) + s, dtype=dt, device=dev, pin_memory=pin)
                               for k, (s, dt) in shapes.items()}
        self.pin = [mk("cpu", self.cuda) for _ in self.pin]
        self.pout = [torch.zeros((cap, 3), dtype=torch.float32, pin_memory=self.cuda) for _ in self.pout]
        self.din = mk(self.dev, False)
        self.dout = torch.zeros((cap, 3), dtype=torch.float32, device=self.dev)
        self.graphs = {}
        self.cap = cap

    def inputs(self, slot, K):
        """Numpy views of slot's pinned inputs, K rows (for direct writes)."""
        self._grow(K)
        return {k: self.pin[slot][k][:K].numpy() for k in LEAF_KEYS}

    def _graph(self, bucket):
        g = self.graphs.get(bucket)
        if g is None:
            x = {k: v[:bucket] for k, v in self.din.items()}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    self._fwd(x["mon"], x["hp"], x["active"])
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                self.dout[:bucket].copy_(self._fwd(x["mon"], x["hp"], x["active"]))
            self.graphs[bucket] = g
        return g

    @torch.no_grad()
    def launch(self, slot, K, leaves=None):
        if self.mode == "eager":
            return slot, K, leaves
        self._grow(K)
        pin = self.pin[slot]
        if leaves is not None:
            for k in LEAF_KEYS:
                pin[k][:K].numpy()[...] = leaves[k]
        if K == 0:
            return slot, K, None
        if self.mode == "graphs":
            bucket = max(self.min_bucket, 1 << (K - 1).bit_length())
            for k in LEAF_KEYS:  # rows past K stay as they were: valid inputs, outputs unread
                self.din[k][:K].copy_(pin[k][:K], non_blocking=True)
            self._graph(bucket).replay()
            src = self.dout[:K]
        else:
            for k in LEAF_KEYS:
                self.din[k][:K].copy_(pin[k][:K], non_blocking=True)
            src = self._fwd(self.din["mon"][:K], self.din["hp"][:K], self.din["active"][:K])
        self.pout[slot][:K].copy_(src, non_blocking=True)
        if self.events[slot] is not None:
            self.events[slot].record()
        return slot, K, None

    @torch.no_grad()
    def collect(self, handle, done, result):
        slot, K, leaves = handle
        if self.mode == "eager":
            out = np.zeros((K, 3), np.float32)
            live = np.flatnonzero(~done.astype(bool))
            for s in range(0, len(live), self.chunk):
                r = live[s:s + self.chunk]
                x = [torch.from_numpy(leaves[k][r]).to(self.dev, non_blocking=True) for k in LEAF_KEYS]
                out[r] = self._fwd(*x).cpu().numpy()
        else:
            if K and self.events[slot] is not None:
                self.events[slot].synchronize()
            out = self.pout[slot][:K].numpy().copy()
        return done_onehot(out, done.astype(bool), result)


class TrainSearch:
    def __init__(self, net, device, amp, budget=64, beta=0.03, floor=0.25, shrink=1.0, iters=2000, tol=1e-4,
                 threads=8, seed=0, compile=False, leaf="eager", value_only=False, direct=False, slots=2,
                 chunk=65536, plan="numpy", solve="numpy"):
        if direct and leaf == "eager":
            raise ValueError("direct needs leaf async or graphs")
        if plan not in ("numpy", "cpp") or solve not in ("numpy", "cpp"):
            raise ValueError("plan and solve: numpy or cpp")
        self.plan, self.solve = plan, solve
        self.budget, self.beta, self.floor, self.shrink = budget, beta, floor, shrink
        self.iters, self.tol, self.threads, self.direct = iters, tol, threads, direct
        self.rng = np.random.default_rng(seed)
        self.leaves = LeafEval(net, device, amp, leaf, compile, value_only, slots, chunk)

    def begin(self, env, obs, picks, logp, p, offset=0, slot=0):
        """Picks the roots, runs the rollouts and queues the leaves' forward.
        Returns a token for `finish`."""
        t0 = time.perf_counter()
        R = len(obs["n"])
        n = obs["n"].astype(np.int64)
        roots = np.flatnonzero(((n >= 2).any(1)) & (self.rng.random(R) < p)) if p > 0 else np.zeros(0, np.int64)
        tok = dict(R=R, roots=roots, picks=picks, t0=t0)
        S, B = len(roots), self.budget
        if S == 0:
            return tok
        nr = np.maximum(n[roots], 1)  # [S, 2]
        legal = np.arange(C)[None, None] < nr[:, :, None]  # [S, 2, C]
        prior = np.where(legal, np.exp(logp[roots].astype(np.float64)), 0.0)
        prior /= prior.sum(-1, keepdims=True)
        # the cells: (1 - floor) from the priors, floor uniform over the legal pairs
        if self.plan == "cpp":
            cells = ataraxos_pi.plan(prior.astype(np.float32), nr[:, 0].astype(np.uint8), nr[:, 1].astype(np.uint8),
                                     B, self.floor, int(self.rng.integers(0, np.iinfo(np.int64).max)), self.threads)
        else:
            rep = lambda a: np.repeat(a[:, None], B, 1)
            uni = self.rng.random((S, B)) < self.floor
            cells = np.empty((S, B, 2), np.int64)
            for side in (0, 1):
                from_prior = sample(self.rng, rep(prior[:, side]), rep(nr[:, side]))
                from_uni = np.minimum((self.rng.random((S, B)) * rep(nr[:, side])).astype(np.int64),
                                      rep(nr[:, side]) - 1)
                cells[..., side] = np.where(uni, from_uni, from_prior)
        battles = np.repeat(obs["battle"][roots].astype(np.int64) - offset, B).astype(np.uint32)
        seeds = self.rng.integers(0, np.iinfo(np.int64).max, S * B, dtype=np.int64).astype(np.uint64)
        cp = cells.reshape(-1, 2).astype(np.uint8)
        t1 = time.perf_counter()
        K = S * B
        done, result = np.empty(K, np.uint8), np.empty(K, np.int8)
        if self.direct:
            x = self.leaves.inputs(slot, K)
            env.rollouts_into(battles, cp, seeds, x["mon"], x["hp"], x["active"], done, result)
            leaves = None
        else:
            leaves = env.rollouts(battles, cp, seeds)
            done, result = leaves["done"], leaves["result"]
        t2 = time.perf_counter()
        handle = self.leaves.launch(slot, K, leaves)
        t3 = time.perf_counter()
        tok.update(nr=nr, legal=legal, prior=prior, cells=cells, done=done, result=result, handle=handle,
                   logp=logp, plan_s=t1 - t0, rollout_s=t2 - t1, launch_s=t3 - t2)
        return tok

    def finish(self, tok):
        """Waits for the leaves, solves, overwrites the searched rows' picks.
        Returns (columns for the buffer, stats): searched u8 [R], post f32
        [R, 2, C] (the posteriors, 0 on rows not searched), swdl f32 [R, 3]
        (P1's W/D/L under them)."""
        R, roots, picks = tok["R"], tok["roots"], tok["picks"]
        searched = np.zeros(R, np.uint8)
        post = np.zeros((R, 2, C), np.float32)
        swdl = np.zeros((R, 3), np.float32)
        S, B = len(roots), self.budget
        stats = dict(search_roots=S)
        if S == 0:
            return dict(searched=searched, post=post, swdl=swdl), stats
        t3 = time.perf_counter()
        wdl = self.leaves.collect(tok["handle"], tok["done"], tok["result"])  # [S * B, 3]
        t4 = time.perf_counter()
        nr, legal, prior, cells, logp = tok["nr"], tok["legal"], tok["prior"], tok["cells"], tok["logp"]
        m = np.where(legal, logp[roots], -1e9).astype(np.float32)
        n1, n2 = nr[:, 0].astype(np.uint8), nr[:, 1].astype(np.uint8)
        if self.solve == "cpp":
            l1, l2, sw, its = ataraxos_pi.solve(cells.astype(np.uint8), wdl, n1, n2, np.ascontiguousarray(m[:, 0]),
                                                np.ascontiguousarray(m[:, 1]), self.shrink, self.beta, 1.0,
                                                self.iters, self.tol, self.threads)
        else:
            # cell sums of Q and of the W/D/L, counts
            flat = (np.repeat(np.arange(S), B) * C + cells[..., 0].ravel()) * C + cells[..., 1].ravel()
            count = np.bincount(flat, minlength=S * C * C).reshape(S, 1, C, C).astype(np.float64)
            q = (wdl[:, 2] - wdl[:, 0]).astype(np.float64)
            tq = np.bincount(flat, q, S * C * C).reshape(S, 1, C, C)
            cell_legal = (legal[:, 0, :, None] & legal[:, 1, None, :])[:, None]
            Q = cell_means(tq, count, cell_legal, self.shrink)[:, 0]
            W = np.stack([cell_means(np.bincount(flat, wdl[:, k].astype(np.float64), S * C * C).reshape(S, 1, C, C),
                                     count, cell_legal, self.shrink)[:, 0] for k in range(3)], -1)  # [S, C, C, 3]
            l1, l2, its = ataraxos_pi.mmd(Q.astype(np.float32), np.ascontiguousarray(m[:, 0]),
                                          np.ascontiguousarray(m[:, 1]), n1, n2, self.beta, 1.0, self.iters,
                                          self.tol, self.threads)
        p1, p2 = np.exp(l1) * legal[:, 0], np.exp(l2) * legal[:, 1]
        p1 /= p1.sum(-1, keepdims=True)
        p2 /= p2.sum(-1, keepdims=True)
        if self.solve != "cpp":
            sw = np.einsum("sijk,si,sj->sk", W, p1, p2)
        # play the posteriors (a side with one choice keeps pick 0)
        for side, ps in ((0, p1), (1, p2)):
            picks[roots, side] = sample(self.rng, ps, nr[:, side]).astype(picks.dtype)
        searched[roots] = 1
        post[roots, 0], post[roots, 1] = p1, p2
        swdl[roots] = sw
        # KL(post || prior) over the deciding sides
        lp = np.log(np.maximum(prior, 1e-30))
        kl = np.concatenate([(ps * (np.log(np.maximum(ps, 1e-30)) - lp[:, side])).sum(-1)[nr[:, side] >= 2]
                             for side, ps in ((0, p1), (1, p2))])
        t5 = time.perf_counter()
        stats.update(search_leaves=S * B, search_leaf_done=float(tok["done"].mean()),
                     search_iters=float(its.mean()), search_kl=float(kl.mean()) if len(kl) else 0.0,
                     search_plan_s=tok["plan_s"], search_rollout_s=tok["rollout_s"],
                     search_value_s=tok["launch_s"] + t4 - t3, search_solve_s=t5 - t4)
        return dict(searched=searched, post=post, swdl=swdl), stats

    def run(self, env, obs, picks, logp, p, offset=0, slot=0):
        """begin then finish: searches a share p of obs's rows (env: the
        SelfPlay those rows are from; offset: added to its battle ids in obs,
        Envs.shift) and overwrites the searched rows' picks in place."""
        return self.finish(self.begin(env, obs, picks, logp, p, offset, slot))
