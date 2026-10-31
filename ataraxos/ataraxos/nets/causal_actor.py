"""Acting with the causal net (ataraxos/nets/causal_net.py, 2026-10-26): each layer's
keys and values per (battle, player), a ring of the last window - 1
positions, so a decision runs its new update and one position of history.

`graphs` (CUDA only): every step padded to fixed shapes (rows to a multiple
of `bucket`, lines to the net's max_lines) and run as one compiled step with
CUDA graphs (torch.compile's "reduce-overhead"); the cache writes stay
outside it. Measured 2026-10-26 at 8192 battles: 130k decisions/s graphed,
78k eager. Padding rows read and write scratch slots past the real ones.

A game's cache mixes the weights of the moments it was written; the learner
recomputes everything.
"""


import numpy as np
import torch

from ataraxos.nets.causal_net import CausalPolicy, summarize_grouped  # noqa: E402
from ataraxos.nets.policy import CHOICES, pad_chunks  # noqa: E402


class CausalActor:
    def __init__(self, net: CausalPolicy, battles: int, device, dtype=torch.bfloat16, amp=True, graphs=False,
                 bucket=4096):
        self.net = net
        self.device = torch.device(device)
        self.dtype = dtype
        self.amp = amp and self.device.type == "cuda"
        self.graphs = graphs and self.device.type == "cuda"
        # a step has at most two decisions a battle
        self.bucket = min(bucket, -(-2 * battles // 256) * 256)
        c = net.c
        self.R = c.window - 1
        self.slots = battles * 2
        S = self.slots + (self.bucket if self.graphs else 0)  # scratch slots for padding rows
        cache = dtype if self.amp else torch.float32
        self.k = torch.zeros(c.causal_depth, S, self.R, c.d, device=self.device, dtype=cache)
        self.v = torch.zeros_like(self.k)
        self.t = torch.zeros(S, dtype=torch.long, device=self.device)
        self.game = torch.full((S,), -1, dtype=torch.long, device=self.device)
        if self.graphs:
            self.fn = torch.compile(self._step, mode="reduce-overhead", dynamic=False)

    def _step(self, lines, request, state, choices, n, t, kc, vc):
        """The net on one padded batch: (logits, value logits, new keys
        [layers, B, d], new values)."""
        net = self.net
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.amp):
            x = net.summarize(lines, request)
            x = x + net.pos[t.clamp(max=net.c.max_pos - 1)]
            valid = torch.arange(self.R, device=lines.device)[None] < t[:, None]
            ks, vs = [], []
            for l, layer in enumerate(net.layers):
                x, k, v = layer.step(x, kc[l], vc[l], valid)
                ks.append(k)
                vs.append(v)
            logits, value = net.decide(net.out_norm(x), request, choices, n, state)
        return logits.float(), value.float(), torch.stack(ks), torch.stack(vs)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        """Returns (picks u8 [D] numpy, log-probs [D, CHOICES] f32, value
        logits [D, 3] f32), the tensors on the device."""
        D = len(obs["n"])
        dev, net = self.device, self.net
        if D == 0:
            return np.zeros(0, np.uint8), torch.zeros(0, CHOICES, device=dev), torch.zeros(0, 3, device=dev)
        to = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)
        B = -(-D // self.bucket) * self.bucket if self.graphs else D
        pad = B - D
        key = obs["battle"].astype(np.int64) * 2 + obs["player"]
        game = obs["game"].astype(np.int64)
        if pad:
            key = np.r_[key, self.slots + np.arange(pad) % self.bucket]
            game = np.r_[game, np.zeros(pad, np.int64)]
        key, game = to(key), to(game)
        new = self.game[key] != game
        self.game[key] = game
        t = torch.where(new, 0, self.t[key])
        if pad:
            t[D:] = 0
        off = obs["offsets"]
        lengths = np.minimum(off[1:] - off[:-1], net.c.max_lines)
        chunks = pad_chunks(torch.from_numpy(obs["tokens"]), torch.from_numpy(off), net.c.max_lines)
        state = obs.get("state")

        def padded(a, fill=0):
            if not pad:
                return a
            out = np.full((B,) + a.shape[1:], fill, a.dtype)
            out[:D] = a
            return out

        request = to(padded(obs["request"]))
        choices = to(padded(obs["choices"]))
        n = to(padded(obs["n"], 2))
        st = to(padded(state)) if net.c.state else None
        if self.graphs:
            lines = torch.zeros(B, net.c.max_lines, chunks.shape[2], dtype=chunks.dtype)
            lines[:D, :chunks.shape[1]] = chunks
            logits, value, k, v = self.fn(lines.to(dev, non_blocking=True), request, st, choices, n, t,
                                          self.k[:, key], self.v[:, key])
            logits, value = logits[:D].clone(), value[:D].clone()  # the graph's outputs are reused
        else:
            with torch.autocast(dev.type, dtype=self.dtype, enabled=self.amp):
                x = summarize_grouped(net, chunks, lengths, request, self.k.dtype)
                x = x + net.pos[t.clamp(max=net.c.max_pos - 1)]
                valid = torch.arange(self.R, device=dev)[None] < t[:, None]
                ks, vs = [], []
                for l, layer in enumerate(net.layers):
                    x, kl, vl = layer.step(x, self.k[l, key], self.v[l, key], valid)
                    ks.append(kl)
                    vs.append(vl)
                k, v = torch.stack(ks), torch.stack(vs)
                logits, value = net.decide(net.out_norm(x), request, choices, n, st)
            logits, value = logits.float(), value.float()
        slot = t % self.R
        self.k[:, key, slot] = k.to(self.k.dtype)
        self.v[:, key, slot] = v.to(self.v.dtype)
        self.t[key] = t + 1
        logp = torch.log_softmax(logits[:D], -1)
        picks = logp.argmax(-1) if greedy else torch.multinomial(logp.exp(), 1)[:, 0]
        return picks.to(torch.uint8).cpu().numpy(), logp, value[:D]


def make_actor(net, battles, device, amp=True, graphs=False):
    return CausalActor(net, battles, device, amp=amp, graphs=graphs)
