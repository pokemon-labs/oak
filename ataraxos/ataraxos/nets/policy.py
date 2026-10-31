"""The policy's shared pieces: the embedding tables, the line and update
encoders, the request's tokens and the choice embedding that
ataraxos/nets/causal_net.py's decision block keys its choices with, and the
size registry (filled by causal_net). birch's move_net.py, history_net.py
and policy.py for Gen 1, without the MoveNet's trunk or the HistoryNet's
history encoder (the user's, 2026-10-31: only the causal net's
infrastructure).

  1. a log line (src/tokens.hpp's Token) becomes one vector: each field
     embedded (tables shared with the request), cut to field_d by its slot's
     linear, concatenated, and passed through an MLP;
  2. an update is the chunk of log lines between one of a side's decisions
     and the next; the UpdateEncoder reads a summary token, a request-kind
     token (move, forceSwitch or wait), the request that ends the update
     (REQUEST_TOKENS tokens) and the lines (positions within the chunk),
     both ways, and the summary token's output is the update's summary;
  3. each player reads their own view of the log (src/view.hpp), as at test
     time.

A decision's choices are Showdown's, as libpkmn choice bytes (type in bits
0-1, data above), up to 9, in the env's order (src/request_data.hpp's
showdown_choices: switch(s) for "switch s", move(i) for "move i", a
one-entry request's lock, Fight or Struggle being move(1)). The logit of choice c is q . k_c /
sqrt(d):
  - q from the active Pokémon's request token (side slot 1, listed first);
  - k_c = key(E[c's byte] + target), E a learned embedding per choice byte,
    the target the decision block's output for what c points to: move slot
    i's request token for move(i), side slot s's for switch(s) (move(0)
    no longer reaches the net).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# src/tokens.hpp's Column
KIND, SUB, ACTOR, OTHER, MOVE, MOVE2, SPECIES, NUM, HP, MAX_HP, STATUS = range(11)
COLUMNS = 11
SLOTS = 10  # COLUMNS, with HP and MAX_HP as one slot

# libpkmn's Gen 1 enum sizes (last + 1)
ARG_TYPES = 43
SPECIES_N = 152
MOVES_N = 166
IDENTS = 17  # 1 + (side << 3 | slot)
BYTE = 256  # SUB, NUM (clamped), STATUS

# src/tokens.hpp's request layout
HEADER = 7
ACTIVE_MOVES, MOVE_FIELDS = 4, 4
SIDE, POKEMON_FIELDS = 6, 14
REQUEST_FIELDS = HEADER + ACTIVE_MOVES * MOVE_FIELDS + SIDE * POKEMON_FIELDS
REQUEST_TOKENS = 1 + ACTIVE_MOVES + SIDE
CHOICES = 9  # libpkmn's Gen 1 MAX_CHOICES: 4 moves, 5 switches


@dataclass
class PolicyConfig:
    d: int = 384
    depth: int = 8
    heads: int = 8
    ff: int = 1536
    pos_std: float = 0.1
    field_d: int = 64
    update_depth: int = 2
    max_lines: int = 64


SIZES = {}  # ataraxos/nets/causal_net.py's: "cs", "cs2", "cs_tiny"


def targets(choices: torch.Tensor) -> torch.Tensor:
    """Choice bytes [B, C] -> the index of each one's target in
    cat(nothing, 4 moves, 6 side slots): move(i) -> i (0 for Struggle),
    switch(s) -> 4 + s."""
    kind, slot = choices & 3, (choices >> 2).long()
    return torch.where(kind == 2, ACTIVE_MOVES + slot, slot)


def encoder(c: PolicyConfig, depth: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(c.d, c.heads, c.ff, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class LineEncoder(nn.Module):
    """A log line's field embeddings -> one vector: each cut to field_d by
    its slot's linear, concatenated, then an MLP."""

    def __init__(self, c: PolicyConfig):
        super().__init__()
        self.slot = nn.ModuleList(nn.Linear(c.d, c.field_d, bias=False) for _ in range(SLOTS))
        self.mlp = nn.Sequential(nn.Linear(SLOTS * c.field_d, c.ff), nn.GELU(), nn.Linear(c.ff, c.d),
                                 nn.LayerNorm(c.d))

    def forward(self, fields):
        return self.mlp(torch.cat([s(f) for s, f in zip(self.slot, fields)], -1))


class UpdateEncoder(nn.Module):
    """An update's prefix (the request-kind token and the request tokens)
    and line vectors (positions added) -> its summary: the summary token's
    output, attending both ways."""

    def __init__(self, c: PolicyConfig):
        super().__init__()
        self.summary_token = nn.Parameter(torch.randn(c.d) * c.pos_std)
        self.encoder = encoder(c, c.update_depth)
        self.norm = nn.LayerNorm(c.d)

    def forward(self, prefix, x, pad):
        """prefix [B, k, d] (never padded), x [B, n, d], pad [B, n] bool.
        Returns [B, d]."""
        B, k = prefix.shape[:2]
        x = torch.cat([self.summary_token.expand(B, 1, -1), prefix, x], 1)
        pad = torch.cat([torch.zeros(B, 1 + k, dtype=torch.bool, device=x.device), pad], 1)
        return self.norm(self.encoder(x, src_key_padding_mask=pad)[:, 0])


class Policy(nn.Module):
    """The tables, LineEncoder -> UpdateEncoder, the request's tokens, and
    `history_encoder`, the layers the causal net's decision block runs, with
    its value and policy heads."""

    def __init__(self, c: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.c = c
        d = c.d
        # Tables shared between columns of the same domain.
        self.kind = nn.Embedding(ARG_TYPES, d)
        self.sub = nn.Embedding(BYTE, d)
        self.ident = nn.Embedding(IDENTS, d)
        self.other = nn.Embedding(IDENTS, d)  # a target is not an actor
        self.move = nn.Embedding(MOVES_N, d)
        self.species = nn.Embedding(SPECIES_N, d)
        self.num = nn.Embedding(BYTE, d)
        self.status = nn.Embedding(BYTE, d)
        self.hp = nn.Linear(2, d)  # (fraction, present)

        # Request tokens: a type embedding each, plus their fields.
        self.req_type = nn.Parameter(torch.randn(REQUEST_TOKENS, d) * c.pos_std)
        self.header = nn.Linear(HEADER, d)
        self.move_scalars = nn.Linear(3, d)  # pp / maxpp, maxpp / 64, disabled
        self.mon_scalars = nn.Linear(4 + 4, d)  # level, hp frac, active, absence, 4 stats

        self.value_token = nn.Parameter(torch.randn(d) * c.pos_std)
        self.norm = nn.LayerNorm(d)
        self.value_head = nn.Linear(d, 3)  # (loss, draw, win), as ataraxos/rl/buffer.py reads it
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)

        self.line_encoder = LineEncoder(c)
        self.line_pos = nn.Parameter(torch.randn(c.max_lines, d) * c.pos_std)
        self.update_encoder = UpdateEncoder(c)
        self.request_kind = nn.Embedding(3, d)  # RequestData::Kind: move, forceSwitch, wait
        self.history_encoder = encoder(c, c.depth)
        self.choice = nn.Embedding(256, c.d)
        nn.init.normal_(self.choice.weight, std=c.pos_std)

    def embed_request(self, q: torch.Tensor) -> torch.Tensor:
        """q: [B, REQUEST_FIELDS] int. Returns [B, REQUEST_TOKENS, d]."""
        q = q.long()
        B = q.shape[0]
        header = self.header(q[:, :HEADER].float().clamp(0, 8))[:, None]
        m = q[:, HEADER:HEADER + ACTIVE_MOVES * MOVE_FIELDS].view(B, ACTIVE_MOVES, MOVE_FIELDS)
        pp, maxpp = m[..., 1].float(), m[..., 2].float()
        moves = self.move(m[..., 0]) + self.move_scalars(
            torch.stack([pp / maxpp.clamp(min=1), maxpp / 64, m[..., 3].float()], -1))
        s = q[:, HEADER + ACTIVE_MOVES * MOVE_FIELDS:].view(B, SIDE, POKEMON_FIELDS)
        stats = s[..., 6:10].float() / 512
        scalars = torch.cat([
            torch.stack([s[..., 1].float() / 100, s[..., 2].float() / s[..., 3].float().clamp(min=1),
                         s[..., 5].float(), (s[..., 0] == 0).float()], -1),
            stats], -1)
        mons = (self.species(s[..., 0]) + self.status(s[..., 4] & 0xFF)
                + self.move(s[..., 10:14]).sum(-2)  # the moveset is a set
                + self.mon_scalars(scalars))
        return torch.cat([header, moves, mons], 1) + self.req_type

    def embed_lines(self, t: torch.Tensor) -> torch.Tensor:
        """t [B, n, COLUMNS] int -> [B, n, d], one vector per log line."""
        t = t.long()
        max_hp = t[..., MAX_HP].float()
        hp = self.hp(torch.stack([t[..., HP].float() / max_hp.clamp(min=1), (max_hp > 0).float()], -1))
        fields = [self.kind(t[..., KIND]), self.sub(t[..., SUB]),
                  self.ident(t[..., ACTOR]), self.other(t[..., OTHER]),
                  self.move(t[..., MOVE]), self.move(t[..., MOVE2]), self.species(t[..., SPECIES]),
                  self.num(t[..., NUM].clamp(0, BYTE - 1)), hp, self.status(t[..., STATUS] & 0xFF)]
        return self.line_encoder(fields)

    def summarize(self, t: torch.Tensor, request: torch.Tensor, lines: torch.Tensor = None) -> torch.Tensor:
        """One update's lines t [B, n, COLUMNS], right-padded with KIND 0
        (n may be 0), and the request [B, REQUEST_FIELDS] that ends it.
        `lines` is embed_lines(t) if the caller already has it: the two
        players' views of an update share most lines, so a driver embeds
        each distinct line once and gathers. Returns its summary [B, d]."""
        n = t.shape[1]
        if lines is None:
            lines = self.embed_lines(t)
        prefix = torch.cat([self.request_kind(request[:, :1].long()), self.embed_request(request)], 1)
        return self.update_encoder(prefix, lines + self.line_pos[:n], t[..., KIND] == 0)


def pad_chunks(tokens: torch.Tensor, offsets: torch.Tensor, max_lines: int) -> torch.Tensor:
    """The env's flat tokens [T, COLUMNS] and offsets [D + 1] -> [D, l,
    COLUMNS], right-padded with zeros (KIND 0), each chunk cut to its last
    max_lines lines; l is the longest kept, at least 1."""
    D = offsets.numel() - 1
    lengths = (offsets[1:] - offsets[:-1]).clamp(max=max_lines)
    l = max(1, int(lengths.max())) if D else 1
    out = torch.zeros(D, l, tokens.shape[-1], dtype=tokens.dtype)
    if D == 0 or tokens.numel() == 0:
        return out
    row = torch.repeat_interleave(torch.arange(D), lengths)
    first = offsets[1:] - lengths  # the first kept line of each chunk
    col = torch.arange(row.numel()) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
    out[row, col] = tokens[first[row] + col]
    return out


def buckets(cnt: np.ndarray, edges, min_rows: int) -> list:
    """Rows [D] grouped by history length, so each group pads the history to
    its own longest: sorted by `cnt`, cut after each edge (a group holds
    counts up to its edge, the last everything above), a group under
    `min_rows` rows joined to the next longer one (the longest to the one
    before). No edges: one group of every row."""
    order = np.argsort(cnt, kind="stable")
    if not len(edges):
        return [order]
    bounds = [0]
    for c in [*np.searchsorted(cnt[order], edges, side="right"), len(order)]:
        if c - bounds[-1] >= max(1, min_rows):
            bounds.append(int(c))
    if bounds[-1] < len(order):
        if len(bounds) > 1:
            bounds[-1] = len(order)
        else:
            bounds.append(len(order))
    return [order[a:b] for a, b in zip(bounds, bounds[1:])]


def check():
    """pad_chunks keeps each chunk's last max_lines lines, right-padded;
    padding doesn't change a summary."""
    tok = torch.arange(20, dtype=torch.int16)[:, None].expand(-1, COLUMNS).contiguous()
    out = pad_chunks(tok, torch.tensor([0, 3, 3, 20]), 5)
    assert out.shape == (3, 5, COLUMNS)
    assert out[0, :3, 0].tolist() == [0, 1, 2] and (out[1] == 0).all()
    assert out[2, :, 0].tolist() == [15, 16, 17, 18, 19]
    torch.manual_seed(0)
    net = Policy(PolicyConfig(d=32, heads=2, ff=64, field_d=8, depth=1)).eval()
    with torch.no_grad():
        t = torch.zeros(3, 7, COLUMNS, dtype=torch.int16)
        t[:, :5, KIND] = torch.randint(1, ARG_TYPES, (3, 5), dtype=torch.int16)
        t[:, :5, MOVE] = torch.randint(0, MOVES_N, (3, 5), dtype=torch.int16)
        t[0, 3:] = 0
        req = torch.randint(0, 3, (3, REQUEST_FIELDS), dtype=torch.int16)
        a = net.summarize(t, req)
        b = net.summarize(torch.cat([t, torch.zeros(3, 4, COLUMNS, dtype=torch.int16)], 1), req)
    return (a - b).abs().max().item()


if __name__ == "__main__":
    print(f"policy: pad_chunks ok; padding changes a summary by at most {check():.2e}")
