"""ataraxos/search.py's Search on the torch-free net (2026-11-21, for pkmn.ai's
bundle): the same vanilla and joint searches (search.py's docstring), the
same layouts from ataraxos_env.Rollouts, the caches as arena.actor.Actor's
numpy arrays, the net as ataraxos_net's. Only the client's entry point
(`run_client`, the ladder's) and the self-play one (`run`, for checks
against the torch search); no `check` hooks, no CUDA graphs (`graphs` is
ignored). Draws (the roots' picks; the rollouts' are the C++ env's own)
come from a numpy Generator, so a search equals the torch one only up to
those."""

import time

import numpy as np

from ataraxos._ext import ataraxos_env
from ataraxos.arena.actor import Actor, log_softmax, sample
from ataraxos.search_np import SearchConfig, cell_means, gather_rows, mmd  # noqa: F401 (SearchConfig re-exported)


def subset(obs, rows):
    """eval/agents.py's: the decisions `rows` of an env output, ends dropped."""
    off = obs["offsets"]
    lengths = off[rows + 1] - off[rows]
    idx = np.repeat(off[rows] - np.cumsum(np.r_[0, lengths[:-1]]), lengths) + np.arange(lengths.sum())
    out = {k: obs[k][rows] for k in ("battle", "game", "player", "request", "choices", "n", "state") if k in obs}
    out["tokens"] = obs["tokens"][idx]
    out["offsets"] = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    return out


def magnet(choices, legal, kind):
    """rl/train.py's magnet: log-probs [B, CHOICES], -1e9 off the legal ones."""
    legal_f = legal.astype(np.float64)
    if kind == "uniform":
        p = legal_f / legal_f.sum(-1, keepdims=True)
    elif kind == "attack_switch":
        move = ((choices.astype(np.int64) & 3) == 1) & legal
        swap = legal & ~move
        nm, ns = move.sum(-1, keepdims=True), swap.sum(-1, keepdims=True)
        groups = (nm > 0).astype(np.float64) + (ns > 0)
        p = (move / np.maximum(nm, 1) + swap / np.maximum(ns, 1)) / groups
    else:
        raise ValueError(kind)
    with np.errstate(divide="ignore"):
        return np.where(legal, np.log(p), -1e9)


def win_minus_loss(value_logits):
    p = np.exp(log_softmax(value_logits.astype(np.float64)))
    return (p[:, 2] - p[:, 0]).astype(np.float32)


def update(q, logp, choices, n, cfg: SearchConfig):
    """search.py's closed-form update for one root: log-probs [CHOICES]."""
    legal = np.arange(len(choices)) < n
    rho = magnet(choices[None], legal[None], cfg.magnet)[0]
    qq = np.zeros(len(choices))
    qq[:n] = q
    x = (qq + cfg.alpha * rho + cfg.beta * np.where(legal, logp, 0)) / (cfg.alpha + cfg.beta)
    return log_softmax(np.where(legal, x, -1e9))


def act_chunked(actor, obs, rows):
    D = len(obs["n"])
    if D <= rows:
        return actor.act(obs)
    parts = [actor.act(subset(obs, np.arange(i, min(i + rows, D)))) for i in range(0, D, rows)]
    return tuple(np.concatenate([p[j] for p in parts]) for j in range(3))


class Search:
    """`net`: ataraxos_net.Net; `teams`: the history sampler's pool (Showdown
    export files); `ratio` its rank-decay prior."""

    def __init__(self, net, teams, cfg: SearchConfig = SearchConfig(), threads=8, ratio=0.1, seed=0):
        self.net, self.cfg = net, cfg
        c = net.config
        self.r = ataraxos_env.Rollouts(list(teams), c["window"] - 1, threads=threads, state=bool(c["state"]),
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
            self.actor = Actor(self.net, max(rollouts, 64), self.rng)
        return self.actor

    @staticmethod
    def copy(actor, dst, src, real=None):
        """Slots `dst` take `src`'s caches (from `real`'s, by default the
        actor's own), as games 0."""
        real = actor if real is None else real
        actor.k[:, dst] = real.k[:, src]
        actor.v[:, dst] = real.v[:, src]
        actor.t[dst] = real.t[src]
        actor.game[dst] = 0

    def prefill(self, actor, keys, lay):
        """Runs the foe's summaries of each world into its cache at `keys` [W]
        (search.py's prefill): summary j of a world at position t0 + j."""
        summaries, off = lay["summaries"], lay["offsets"]
        count = np.diff(summaries)
        actor.game[keys] = 0
        actor.t[keys] = lay["t0"]
        tokens = lay["tokens"]
        for j in range(int(count.max()) if len(count) else 0):
            w = np.flatnonzero(count > j)
            s = summaries[w] + j
            sub = gather_rows({"tokens": tokens, "offsets": off, "request": lay["request"]}, s, keys=("request",))
            sub["choices"] = np.zeros((len(w), 9), np.uint8)
            sub["n"] = np.zeros(len(w), np.uint8)
            k = keys[w]
            actor.run(sub, k, actor.t[k].copy(), decide=False)

    def run(self, env, real: Actor, obs, logp, seed=None, truth=False):
        """search.py's run: decisions `obs` of the SelfPlay `env`, all ours,
        `real` having just acted on them (`logp` its log-probs)."""
        c = self.cfg
        keys = obs["battle"].astype(np.int64) * 2 + obs["player"]
        if c.joint:
            def add_joint(r, i, seed):
                r.add_joint_selfplay(env, int(obs["battle"][i]), int(obs["player"][i]), c.types_a, c.types_b,
                                     c.depth, seed, truth)

            return self.search_joint(add_joint, real, keys, obs["player"], obs, logp, seed)

        def add(r, i, seed):
            r.add_selfplay(env, int(obs["battle"][i]), int(obs["player"][i]), self.worlds(int(obs["n"][i])),
                           c.reps, c.depth, seed, truth)

        return self.search(add, real, keys, obs["player"], obs, logp, seed)

    def run_client(self, client, real: Actor, key, obs, logp, seed=None):
        """One search at the ladder Client's request: `obs` its one decision,
        which `real` ran at cache key `key`."""
        c = self.cfg
        keys, players = np.array([key], np.int64), np.array([client.viewer()])
        if c.joint:
            def add_joint(r, i, seed):
                r.add_joint_client(client, c.types_a, c.types_b, c.depth, seed)

            return self.search_joint(add_joint, real, keys, players, obs, logp, seed)

        def add(r, i, seed):
            r.add_client(client, self.worlds(int(obs["n"][0])), c.reps, c.depth, seed)

        return self.search(add, real, keys, players, obs, logp, seed)

    def _lap(self):
        tic = time.perf_counter()

        def lap(what):
            nonlocal tic
            now = time.perf_counter()
            self.times[what] = self.times.get(what, 0.0) + now - tic
            tic = now

        return lap

    def _rollout_loop(self, actor, r, seed):
        out = r.start(seed)
        steps = 0
        while len(out["n"]):
            picks, _, value = act_chunked(actor, out, self.cfg.act_rows)
            out = r.step(picks, win_minus_loss(value))
            steps += 1
        assert r.done
        return steps

    def search(self, add, real, keys, players, obs, logp, seed=None):
        D = len(obs["n"])
        seed = int(self.rng.integers(1 << 62)) if seed is None else seed
        r = self.r
        lap = self._lap()
        r.clear()
        for i in range(D):
            add(r, i, seed + i)
        lay = r.layout()
        N = r.rollouts
        lap("sample")
        actor = self._actor(N)
        world_of = lay["world"].astype(np.int64)
        root_of = lay["root"].astype(np.int64)[world_of]
        players = np.asarray(players, np.int64)
        us = players[root_of]
        idx = np.arange(N, dtype=np.int64)
        ours, foes = idx * 2 + us, idx * 2 + 1 - us
        self.copy(actor, ours, np.asarray(keys, np.int64)[root_of], real)  # ours: the real game's cache
        _, first = np.unique(world_of, return_index=True)  # the foe's: each world's history
        first_keys = idx[first] * 2 + 1 - us[first]
        self.prefill(actor, first_keys, lay)
        self.copy(actor, foes, first_keys[world_of])
        lap("prefill")
        steps = self._rollout_loop(actor, r, seed)
        lap("rollouts")
        picks = np.zeros(D, np.uint8)
        qs, lps = [], np.empty_like(logp)
        for i in range(D):
            q = r.q(i)
            lp = update(q, logp[i], obs["choices"][i], int(obs["n"][i]), self.cfg)
            picks[i] = sample(lp[None], self.rng)[0]
            qs.append(q)
            lps[i] = lp
        lap("update")
        s = self.stats
        s["roots"] += D
        s["rollouts"] += N
        s["steps"] += steps
        s["differs"] += int(lay["differs"].sum())
        return picks, qs, lps

    def search_joint(self, add_joint, real, keys, players, obs, logp, seed=None):
        c = self.cfg
        D = len(obs["n"])
        seed = int(self.rng.integers(1 << 62)) if seed is None else seed
        r = self.r
        C = logp.shape[1]
        lap = self._lap()
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
        self.copy(actor, tkeys[real_y], np.asarray(keys, np.int64)[lay["root"][real_y].astype(np.int64)], real)
        # each type's policy at the root: A0's the real one's, a deciding
        # type's from its root decision, else its one choice
        choices = np.ascontiguousarray(lay["choices"])
        legal = np.arange(C)[None] < lay["n"].astype(np.int64)[:, None]
        lp_t = np.where(legal, 0.0, -1e9)
        lp_t[:, 0] = 0.0
        lp_t = log_softmax(lp_t)
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
            lp_t[dec] = lp
        lp_t[real_y] = logp[lay["root"][real_y].astype(np.int64)]
        rho_np = magnet(choices, legal, c.magnet)
        lp_np = lp_t.astype(np.float64)
        lap("prefill")

        beta2 = c.beta if c.foe_beta < 0 else c.foe_beta
        lam1, lam2 = c.alpha + c.beta, c.alpha + beta2
        roots = []
        for i in range(D):
            A, B = int(na[i]), int(nb[i])
            ya, yb = first[i] + np.arange(A), first[i] + A + np.arange(B)
            mag1 = (c.alpha * rho_np[ya] + c.beta * lp_np[ya]) / lam1
            mag2 = (c.alpha * rho_np[yb] + beta2 * lp_np[yb]) / lam2
            roots.append(dict(A=A, B=B, ya=ya, yb=yb, m1=mag1, m2=mag2, g1=legal[ya], g2=legal[yb],
                              total=np.zeros((A, B, C, C)), count=np.zeros((A, B, C, C)), l1=None, l2=None))
        steps = 0
        rng = np.random.default_rng(seed)
        for rd in range(c.rounds):
            cells = []
            for i, x in enumerate(roots):
                p1 = np.exp(self._marginal(x, "l1", "m1", "g1"))
                p2 = np.exp(self._marginal(x, "l2", "m2", "g2"))
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
            root_of = vals["root"].astype(np.int64)
            us = players[root_of]
            idx = np.arange(N, dtype=np.int64)
            ka = tkeys[first[root_of] + vals["a"]]
            kb = tkeys[first[root_of] + na[root_of] + vals["b"]]
            self.copy(actor, idx * 2 + us, ka)
            self.copy(actor, idx * 2 + 1 - us, kb)
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
                self._solve(x, lam1, lam2)
            lap("update")

        picks = np.zeros(D, np.uint8)
        qs, lps = [], np.empty_like(logp)
        for i, x in enumerate(roots):
            l1 = self._marginal(x, "l1", "m1", "g1")[0]  # A0
            n = int(obs["n"][i])
            lp = log_softmax(np.where(np.arange(C) >= n, -1e9, l1)).astype(logp.dtype)
            picks[i] = sample(lp[None], self.rng)[0]
            qs.append(x["q1"][:n])
            lps[i] = lp
        self.stats["roots"] += D
        self.stats["steps"] += steps
        lap("update")
        return picks, qs, lps

    def _marginal(self, x, l, m, g):
        if x[l] is None:
            return np.where(x[g], x[m] - np.logaddexp.reduce(np.where(x[g], x[m], -1e9), -1, keepdims=True), -1e9)
        if self.cfg.joint_mode == "per_type":
            return x[l][0]
        A, B = x["A"], x["B"]
        p = np.exp(x[l]).reshape(A, B, -1)
        p = p.mean(1) if l == "l1" else p.mean(0)
        return np.log(np.maximum(p, 1e-300))

    def _solve(self, x, lam1, lam2):
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
            m1 = np.repeat(x["m1"], B, 0)[:, None]
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
