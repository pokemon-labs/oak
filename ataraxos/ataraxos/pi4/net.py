"""pi4's net (2026-11-24, docs/pi4.md): rby4's decision block with no history,
on perfect-info input, both players' policies from one forward.

Input per row (ataraxos_env's `pi4_obs`, src/pi4.hpp):
  - state i16 [B, 2, PI4_FIELDS]: from P1's side and from P2's, each rby4's
    state (src/state_tokens.hpp + src/features.hpp, STATE_FIELDS) on the
    perfect-info Bookkeeping, then the last-move state (encode_last,
    LAST_FIELDS);
  - request i16 [B, 2, REQUEST_FIELDS]: each player's Showdown request;
  - choices u8 [B, 2, CHOICES], n u8 [B, 2]: each player's Showdown choices
    (n 1 where a side doesn't decide in the row).

The tokens: [value, P1's 11 request tokens, P2's 11, the 14 state tokens].
The state tokens are rby4's embed_state from P1's side (cs3's features on
the actives) plus the last-move state on the two active tokens; each
player's request tokens are rby4's embed_request + request_features from
that player's own side (so P2's moves and switch-ins get P2's matchup
features), plus a learned side vector. rby4's decision block (pre-LN
encoder layers) runs over them; then, as rby4's decide: W/D/L value from the
value token (P1's), and per player a policy (query from its active's request
token, keys from each choice's target token + the choice embedding) and a Q
head (W/D/L per choice, that player's: Expected SARSA).

Built on CausalPolicy for its embeddings and decision-block code; its
history parts (line / update encoders, causal layers) are dropped.
"""

import dataclasses

import torch
import torch.nn as nn

from ataraxos.nets.causal_net import STATE_FIELDS, CausalConfig, CausalPolicy, zero_linear
from ataraxos.nets.policy import ACTIVE_MOVES, CHOICES, MOVES_N, REQUEST_TOKENS, targets

LAST_SIDE_FIELDS = 4
LAST_FIELDS = 1 + 2 * LAST_SIDE_FIELDS  # src/state_tokens.hpp's encode_last
PI4_FIELDS = STATE_FIELDS + LAST_FIELDS

SIZES = {
    # the user's 10-15M (2026-11-24): rby4's 64-dim heads and 3x ff, wider and deeper; ~13.5M
    "pi4": CausalConfig(d=384, heads=6, ff=1152, depth=8, causal_depth=0, state=True, features=True),
    "pi4_tiny": CausalConfig(d=32, heads=2, ff=64, depth=1, field_d=8, line_ff=32, causal_depth=0, state=True,
                             features=True),
}

HISTORY = ("line_encoder", "line_pos", "update_encoder", "request_kind", "layers", "pos", "out_norm", "h_type",
           "kind", "sub", "ident", "other", "num", "hp", "query", "key")


class Pi4Net(CausalPolicy):
    def __init__(self, c: CausalConfig):
        super().__init__(dataclasses.replace(c, q_head=False))
        for name in HISTORY:
            delattr(self, name)
        d = c.d
        self.side_type = nn.Parameter(torch.randn(2, d) * c.pos_std)
        self.last_selected = zero_linear(d, d)
        self.last_used = zero_linear(d, d)
        self.last_scalars = zero_linear(3, d)  # last_damage / 512, counterable, desync possible
        self.queries = nn.ModuleList(nn.Linear(d, d) for _ in range(2))
        self.keys = nn.ModuleList(nn.Linear(d, d) for _ in range(2))
        self.q_heads = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3)) for _ in range(2))

    def embed_last(self, last):
        """last [B, LAST_FIELDS] -> the two active tokens' additions [B, 2, d]."""
        last = last.long()
        B = last.shape[0]
        side = last[:, 1:].view(B, 2, LAST_SIDE_FIELDS)
        ld = (last[:, :1].float() / 512).expand(B, 2)
        scalars = torch.stack([ld, side[..., 2].float(), side[..., 3].float()], -1)
        return (self.last_selected(self.move(side[..., 0].clamp(0, MOVES_N - 1)))
                + self.last_used(self.move(side[..., 1].clamp(0, MOVES_N - 1))) + self.last_scalars(scalars))

    def forward(self, state, request, choices, n, with_q=False):
        """Returns (value logits [B, 3] (loss, draw, win), P1's; logits [B,
        2, CHOICES], -1e9 past n) and with `with_q` Q's logits [B, 2,
        CHOICES, 3], each player's own W/D/L."""
        B = state.shape[0]
        dev = state.device
        s0 = state[:, 0]
        st = self.embed_state(s0[:, :STATE_FIELDS])
        st = st + torch.cat([self.embed_last(s0[:, STATE_FIELDS:]).to(st.dtype),
                             torch.zeros_like(st[:, 2:])], 1)
        reqs = []
        for p in range(2):
            r = self.embed_request(request[:, p])
            r = self.request_features(r, request[:, p], state[:, p, :STATE_FIELDS])
            reqs.append(r + self.side_type[p])
        x = torch.cat([self.value_token.expand(B, 1, -1), reqs[0], reqs[1], st], 1)
        x = self.norm(self.history_encoder(x))
        value = self.value_head(x[:, 0])
        logits, qs = [], []
        live_all = torch.arange(CHOICES, device=dev)[None]
        for p in range(2):
            req = x[:, 1 + p * REQUEST_TOKENS:1 + (p + 1) * REQUEST_TOKENS]
            moves, mons = req[:, 1:1 + ACTIVE_MOVES], req[:, 1 + ACTIVE_MOVES:]
            q = self.queries[p](mons[:, 0])
            pool = torch.cat([torch.zeros_like(moves[:, :1]), moves, mons], 1)
            ch = choices[:, p]
            idx = targets(ch)
            tc = pool.gather(1, idx[..., None].expand(-1, -1, pool.shape[-1])) + self.choice(ch.long())
            lg = torch.einsum("bd,bcd->bc", q, self.keys[p](tc)) / self.c.d ** 0.5
            live = live_all < n[:, p, None].clamp(min=1)
            logits.append(lg.float().masked_fill(~live, -1e9))
            if with_q:
                qs.append(self.q_heads[p](x[:, :1] + tc))
        out = (value, torch.stack(logits, 1))
        return out + (torch.stack(qs, 1),) if with_q else out

    def value(self, state, request, choices, n):
        return self.forward(state, request, choices, n)[0]


def make_net(size: str) -> Pi4Net:
    return Pi4Net(SIZES[size])


def is_pi4(size: str) -> bool:
    return size in SIZES


def load_net(path, device, weights="ema") -> Pi4Net:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = make_net(ck["config"]["size"]).to(device).eval()
    net.load_state_dict(ck[weights])
    return net
