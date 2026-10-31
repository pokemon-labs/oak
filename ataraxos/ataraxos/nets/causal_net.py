"""The small causal net ("cs", 2026-10-26), a faster successor to the
HistoryNet (run D's net, removed 2026-10-30) for training on one 4090 (the
user's ask: under 5M parameters, the same input; a causal history encoder,
the user's idea).

  1. LineEncoder and UpdateEncoder as the HistoryNet's: one summary per
     update, from the update's lines and the request that ends it.
  2. History: `causal_depth` pre-LN layers over a player's summaries, each
     attending to itself and the `window - 1` before it, with learned
     positions by decision index in the game. Causal, so the actor keeps
     each layer's keys and values per (battle, player) and a decision costs
     one new position (ataraxos/nets/causal_actor.py); the learner runs one pass per
     trajectory.
  3. Decision block: `depth` layers over [value, the history's output, the
     11 request tokens, and with `state` the 9 bookkeeping tokens
     (src/state_tokens.hpp: the field, the two actives, the foe's 6
     Pokémon)], then the Policy's heads.
  4. Value: W/D/L logits, (loss, draw, win) (the user's, 2026-10-27: draws
     happen in stall against stall), by cross-entropy against TD(lambda)
     returns over the distribution (ataraxos/rl/buffer.py).

Measured on the 4090 (2026-10-26, D's data, 2048 games): the learner 2.4 s
an iteration against the HistoryNet's 8.2 s; the actor 130k decisions/s
CUDA-graphed at 8192 battles against 26k.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ataraxos.nets.policy import (ACTIVE_MOVES, CHOICES, REQUEST_TOKENS, SIZES, SLOTS, Policy,  # noqa: E402
                                  PolicyConfig, buckets, targets)

# src/state_tokens.hpp's layout: written with the Gen 1 bookkeeping (phase
# 2); until then the env has no state and `state` is off.
STATE_FIELDS = 1

# lines per update are padded up to one of these (the learner's shapes stay
# few, so the compiled graphs don't multiply)
LINE_EDGES = (8, 16, 32)


@dataclass
class CausalConfig(PolicyConfig):
    d: int = 256
    heads: int = 4
    ff: int = 768
    depth: int = 1          # the decision block
    update_depth: int = 2
    field_d: int = 32
    line_ff: int = 512
    max_lines: int = 32     # 31 seen at most on D's games (2026-10-26)
    causal_depth: int = 2
    window: int = 32        # a decision attends to itself and the 31 before
    max_pos: int = 256      # decision index in the game, clamped
    state: bool = False     # the bookkeeping tokens: Gen 1's are phase 2


class CausalLayer(nn.Module):
    """Pre-LN self-attention over a player's summaries, causal within a
    sliding window; `step` runs one new position against cached keys and
    values."""

    def __init__(self, c: CausalConfig):
        super().__init__()
        d = c.d
        self.h = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, c.ff), nn.GELU(), nn.Linear(c.ff, d))

    def heads(self, x):  # [..., n, d] -> [..., h, n, d/h]
        return x.unflatten(-1, (self.h, -1)).transpose(-3, -2)

    def forward(self, x, allowed):
        """x [G, T, d], allowed [T, T] bool."""
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        a = F.scaled_dot_product_attention(self.heads(q), self.heads(k), self.heads(v), attn_mask=allowed)
        x = x + self.out(a.transpose(-3, -2).flatten(-2))
        return x + self.ff(self.ln2(x))

    def step(self, x, kc, vc, valid):
        """x [B, d]; kc, vc [B, W-1, d] the earlier positions' keys and
        values (in any order); valid [B, W-1]. Returns (y [B, d], k [B, d],
        v [B, d])."""
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        K = torch.cat([kc.to(k.dtype), k[:, None]], 1)
        V = torch.cat([vc.to(v.dtype), v[:, None]], 1)
        m = torch.cat([valid, torch.ones_like(valid[:, :1])], 1)[:, None, None]
        a = F.scaled_dot_product_attention(self.heads(q[:, None]), self.heads(K), self.heads(V), attn_mask=m)
        x = x + self.out(a.transpose(-3, -2).flatten(-2)[:, 0])
        return x + self.ff(self.ln2(x)), k, v


class CausalPolicy(Policy):
    def __init__(self, c: CausalConfig = CausalConfig()):
        super().__init__(c)
        d = c.d
        self.line_encoder.mlp = nn.Sequential(nn.Linear(SLOTS * c.field_d, c.line_ff), nn.GELU(),
                                              nn.Linear(c.line_ff, d), nn.LayerNorm(d))
        self.layers = nn.ModuleList(CausalLayer(c) for _ in range(c.causal_depth))
        self.pos = nn.Parameter(torch.randn(c.max_pos, d) * c.pos_std)
        self.out_norm = nn.LayerNorm(d)
        self.h_type = nn.Parameter(torch.randn(d) * c.pos_std)
        if c.state:
            raise NotImplementedError("the Gen 1 state tokens are phase 2")

    def run_core(self, x):
        """x [G, T, d] a trajectory's summaries per row (right-padded) ->
        the history's output at each position [G, T, d]."""
        T = x.shape[1]
        t = torch.arange(T, device=x.device)
        allowed = (t[None] <= t[:, None]) & (t[None] > t[:, None] - self.c.window)
        x = x + self.pos[t.clamp(max=self.c.max_pos - 1)]
        for layer in self.layers:
            x = layer(x, allowed)
        return self.out_norm(x)

    def decide(self, h, request, choices, n, state=None):
        """h [B, d] the history's output at this decision. Returns (logits
        [B, CHOICES], -inf past n; value logits [B, 3])."""
        B = h.shape[0]
        dev = h.device
        parts = [self.value_token.expand(B, 1, -1), (h.to(self.value_token.dtype) + self.h_type)[:, None],
                 self.embed_request(request)]
        x = self.norm(self.history_encoder(torch.cat(parts, 1)))
        value = self.value_head(x[:, 0])
        req = x[:, 2:2 + REQUEST_TOKENS]
        moves, mons = req[:, 1:1 + ACTIVE_MOVES], req[:, 1 + ACTIVE_MOVES:]
        q = self.query(mons[:, 0])
        pool = torch.cat([torch.zeros_like(moves[:, :1]), moves, mons], 1)
        idx = targets(choices)
        target = pool.gather(1, idx[..., None].expand(-1, -1, pool.shape[-1]))
        k = self.key(target + self.choice(choices.long()))
        logits = torch.einsum("bd,bcd->bc", q, k) / self.c.d ** 0.5
        live = torch.arange(CHOICES, device=dev)[None] < n[:, None]
        return logits.masked_fill(~live, float("-inf")), value


# Registered beside policy.SIZES' (importing this module adds them): "cs" the
# measured default, "cs_tiny" for CPU smoke tests.
SIZES["cs"] = CausalConfig()
# CS2 (the user, 2026-10-27): 5 decision layers, 4 causal layers, d 256
SIZES["cs2"] = CausalConfig(depth=5, causal_depth=4)
SIZES["cs_tiny"] = CausalConfig(d=32, heads=2, ff=64, field_d=8, line_ff=32, window=6)


def make_net(c: CausalConfig) -> CausalPolicy:
    return CausalPolicy(c)


def line_edge(n: int) -> int:
    """The padded line count for an update of n lines."""
    for e in LINE_EDGES:
        if n <= e:
            return e
    return max(n, 1)


def summarize_grouped(net, lines, lengths, request, out_dtype, min_rows=256):
    """Every update's summary [N, d]: lines [N, L, COLUMNS] (int, any
    device) padded to the longest, grouped by line count, each group padded
    to its edge (LINE_EDGES)."""
    N = lines.shape[0]
    dev = request.device
    out = torch.empty(N, net.c.d, device=dev, dtype=out_dtype)
    for rows in buckets(lengths, list(LINE_EDGES), min_rows):
        L = line_edge(int(lengths[rows].max()))
        x = lines[torch.from_numpy(rows).to(lines.device), :min(L, lines.shape[1])]
        if x.shape[1] < L:
            x = torch.cat([x, x.new_zeros(x.shape[0], L - x.shape[1], x.shape[2])], 1)
        r = torch.from_numpy(rows).to(dev)
        out[r] = net.summarize(x.to(dev, non_blocking=True), request[r]).to(out_dtype)
    return out


def upload(a, dev):
    """A numpy array or CPU tensor on `dev`, through pinned memory on CUDA
    (an asynchronous copy; a pageable one blocks the thread)."""
    t = a if torch.is_tensor(a) else torch.from_numpy(np.ascontiguousarray(a))
    return t.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else t


class DeviceData:
    """An iteration's buffer.Data (and its returns and advantages) on the
    device, uploaded once (2026-10-27: the learner's per-pass copies, CPU
    padding and .item() calls left the GPU idle most of the step). `src`,
    `ret` and `adv` are the objects it was built from."""

    def __init__(self, data, dev, ret=None, adv=None):
        self.src, self.ret_src, self.adv_src = data, ret, adv
        self.tokens = upload(data.tokens, dev)
        self.offsets = upload(data.offsets, dev)
        self.request = upload(data.request, dev)
        self.choices = upload(data.choices, dev)
        self.n = upload(data.n, dev)
        self.pick = upload(data.pick.astype(np.int64), dev)
        self.logp = upload(data.logp, dev)
        self.state = upload(data.state, dev) if data.state is not None else None
        self.ret = upload(ret, dev) if ret is not None else None
        self.adv = upload(adv, dev) if adv is not None else None


def learner_forward(net, dd, group, keep, dev, amp):
    """The learner's pass over trajectories `group` of `dd` (a DeviceData):
    every decision's summary, the history over each trajectory (its length
    padded to a multiple of 16), and the decision block on the kept
    decisions. The indices are built on the CPU and uploaded in one copy;
    the lines are gathered and padded on the device, grouped by line count
    as summarize_grouped's. Returns (ids [K] numpy, ids [K] on the device,
    logits [K, CHOICES], value logits [K, 3], choices [K, CHOICES], n [K])."""
    data = dd.src
    start, length = data.start[group], data.length[group]
    base = np.r_[0, np.cumsum(length)[:-1]]
    N = int(length.sum())
    ids = np.repeat(start - base, length) + np.arange(N)
    pos = np.arange(N) - np.repeat(base, length)
    row = np.repeat(np.arange(len(group)), length)
    k = np.flatnonzero(keep[ids])
    lengths = np.minimum(data.offsets[ids + 1] - data.offsets[ids], net.c.max_lines)
    groups = buckets(lengths, list(LINE_EDGES), 256)
    edges = [line_edge(int(lengths[g].max())) for g in groups]
    idx = upload(np.concatenate([ids, pos, row, k, *groups]).astype(np.int64), dev)
    ids_t, pos_t, row_t, k_t, order = idx.split([N, N, N, len(k), N])
    end = dd.offsets[ids_t + 1]
    count = (end - dd.offsets[ids_t]).clamp(max=net.c.max_lines)
    first = end - count  # each chunk's last max_lines lines, as pad_chunks
    request = dd.request[ids_t]
    out_dtype = torch.bfloat16 if dev.type == "cuda" and amp else torch.float32
    last = max(dd.tokens.shape[0] - 1, 0)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda" and amp):
        s = torch.empty(N, net.c.d, device=dev, dtype=out_dtype)
        a = 0
        for g, L in zip(groups, edges):
            r = order[a:a + len(g)]
            a += len(g)
            j = torch.arange(L, device=dev)
            x = dd.tokens[(first[r, None] + j).clamp(max=last)] * (j < count[r, None])[..., None]
            s[r] = net.summarize(x, request[r]).to(out_dtype)
        T = -(-int(length.max()) // 16) * 16
        x = torch.zeros(len(group), T, net.c.d, device=dev, dtype=s.dtype)
        x[row_t, pos_t] = s
        h = net.run_core(x)
        hk = h[row_t[k_t], pos_t[k_t]]
        kid = ids_t[k_t]
        choices, n = dd.choices[kid], dd.n[kid]
        logits, value = net.decide(hk, request[k_t], choices, n, dd.state[kid] if net.c.state else None)
    return ids[k], kid, logits, value, choices, n


def check(seed=0):
    """Padding the lines doesn't change a summary; the history at a
    position doesn't depend on later ones; params at the default size."""
    torch.manual_seed(seed)
    from ataraxos.nets.policy import ARG_TYPES, COLUMNS, KIND, REQUEST_FIELDS
    c = CausalConfig(d=32, heads=2, ff=64, field_d=8, line_ff=32, window=4)
    net = CausalPolicy(c).eval()
    with torch.no_grad():
        lines = torch.zeros(3, 5, COLUMNS, dtype=torch.int16)
        lines[:, :3, KIND] = torch.randint(1, ARG_TYPES, (3, 3), dtype=torch.int16)
        req = torch.randint(0, 3, (3, REQUEST_FIELDS), dtype=torch.int16)
        a = net.summarize(lines, req)
        b = net.summarize(torch.cat([lines, torch.zeros(3, 4, COLUMNS, dtype=torch.int16)], 1), req)
        worst = (a - b).abs().max().item()
        x = torch.randn(2, 9, c.d)
        y = x.clone()
        y[:, 6:] = torch.randn(2, 3, c.d)
        worst = max(worst, (net.run_core(x)[:, :6] - net.run_core(y)[:, :6]).abs().max().item())
    return worst, sum(p.numel() for p in CausalPolicy().parameters())


if __name__ == "__main__":
    worst, params = check()
    print(f"causal: padding and later positions change the outputs by at most {worst:.2e}; "
          f"{params / 1e6:.2f}M params at the default size")
