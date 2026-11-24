"""The perfect-info net (2026-11-18, the user's design), for oak's IS-MCTS
unmodified: one forward per battle state gives the value and both players'
policies.

Input: src/pi_env.hpp's features, one token per Pokémon (side, order slot;
slot 0 the active). Kept cacheable at search time, as oak's cache.h:

  - the Pokémon net (species, status, HP on oak's 50 buckets) and the moves
    net (4 x move id x PP bucket) are separate, so a search precomputes
    each over its whole key space per Pokémon (750 and 2401 entries), and
    the token is linear in them: tok = W_p p + W_m m + pos[side, slot];
  - the active net (oak's Active features, hidden counters out) is added to
    slot 0's token, computed live per leaf as oak does;
  - fainted or empty slots are masked out of attention and the pool.

Each subnet is two layers with ReLU after both, as oak's EmbeddingNet. The
trunk is the bench's (~/nn-bench tbench3.cc, "d=64 1 block MLP x2 -> pool ->
MLP128"): pre-LN attention, pre-LN per-token MLP, then a masked mean pool
per side (concatenated), an LN, a ReLU layer, and the heads: value W/D/L
logits (loss, draw, win) from P1's side; P1's and P2's policy over oak's
flat 315 (Encode::Battle::Policy: move id, or 164 + species for a switch),
each a ReLU layer then one row per output, so a search computes only the
legal rows.

Every layer is a plain Linear / Embedding / LayerNorm, so the C++ forward is
a port of this file.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ataraxos._ext import ataraxos_pi as P

SPECIES = P.SPECIES_DIM
STATUS = P.STATUS_DIM
MOVES = P.MOVE_DIM
ACTIVE = P.ACTIVE_DIM
POLICY = P.POLICY_DIM
CHOICES = P.MAX_CHOICES


@dataclass
class PiConfig:
    d: int = 64
    heads: int = 2
    blocks: int = 1
    mlp_ratio: int = 2
    pokemon_hidden: int = 64
    pokemon_out: int = 32
    moves_hidden: int = 64
    moves_out: int = 32
    active_hidden: int = 64
    active_out: int = 32
    hidden: int = 128  # the pooled MLP
    value_hidden: int = 32
    policy_hidden: int = 64


SIZES = {
    "pi": PiConfig(),
    # pi3 (2026-11-23, the user's: pi2 lost 0.157 to rby4, "the net's too
    # small"; the cheapest per CPU leaf of the 1-2M shapes): 1.42M params,
    # trunk 0.59M (~7M MACs a leaf, ~15x pi's), wide heads (run once a leaf)
    "pi3": PiConfig(d=128, heads=4, blocks=3, mlp_ratio=4, pokemon_hidden=128, pokemon_out=64, moves_hidden=128,
                    moves_out=64, active_hidden=128, active_out=64, hidden=512, value_hidden=128,
                    policy_hidden=256),
    "pi_tiny": PiConfig(d=32, heads=1, pokemon_hidden=32, pokemon_out=16, moves_hidden=32, moves_out=16,
                        active_hidden=32, active_out=16, hidden=32, value_hidden=16, policy_hidden=16),
}


class Block(nn.Module):
    def __init__(self, c: PiConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.out = nn.Linear(c.d, c.d)
        self.ln2 = nn.LayerNorm(c.d)
        self.fc1 = nn.Linear(c.d, c.mlp_ratio * c.d)
        self.fc2 = nn.Linear(c.mlp_ratio * c.d, c.d)
        self.heads = c.heads

    def forward(self, x, mask):
        """x [B, T, d], mask [B, T] (True: a token)."""
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        x = x + self.out(a.transpose(1, 2).reshape(B, T, d))
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))


class PiNet(nn.Module):
    def __init__(self, c: PiConfig):
        super().__init__()
        self.c = c
        # the Pokémon net: species and status one-hots and the HP value into
        # the first layer, as oak's sparse Pokemon::write (index 0 of species
        # is the masked slot's, never read)
        self.species = nn.Embedding(SPECIES, c.pokemon_hidden)
        self.status = nn.Embedding(STATUS, c.pokemon_hidden)
        self.hp = nn.Parameter(torch.zeros(c.pokemon_hidden))
        self.pokemon_bias = nn.Parameter(torch.zeros(c.pokemon_hidden))
        self.pokemon_fc = nn.Linear(c.pokemon_hidden, c.pokemon_out)
        # the moves net: a sum over the 4 slots' (move, PP bucket) rows; the
        # last row is the empty slot's, zero and frozen
        self.moves = nn.Embedding(MOVES + 1, c.moves_hidden, padding_idx=MOVES)
        self.moves_bias = nn.Parameter(torch.zeros(c.moves_hidden))
        self.moves_fc = nn.Linear(c.moves_hidden, c.moves_out)
        # the active net, dense
        self.active_fc0 = nn.Linear(ACTIVE, c.active_hidden)
        self.active_fc1 = nn.Linear(c.active_hidden, c.active_out)
        # the token: linear in the subnets' outputs, plus (side, slot)
        self.tok_pokemon = nn.Linear(c.pokemon_out, c.d)
        self.tok_moves = nn.Linear(c.moves_out, c.d, bias=False)
        self.tok_active = nn.Linear(c.active_out, c.d, bias=False)
        self.pos = nn.Parameter(torch.randn(12, c.d) * 0.02)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.blocks))
        self.ln = nn.LayerNorm(2 * c.d)
        self.fc = nn.Linear(2 * c.d, c.hidden)
        self.value_fc = nn.Linear(c.hidden, c.value_hidden)
        self.value_out = nn.Linear(c.value_hidden, 3)
        self.policy_fc = nn.ModuleList(nn.Linear(c.hidden, c.policy_hidden) for _ in range(2))
        self.policy_out = nn.ModuleList(nn.Linear(c.policy_hidden, POLICY) for _ in range(2))
        nn.init.normal_(self.species.weight, std=0.1)
        nn.init.normal_(self.status.weight, std=0.1)
        nn.init.normal_(self.moves.weight, std=0.1)
        with torch.no_grad():
            self.moves.weight[MOVES].zero_()

    def tokens(self, mon, hp, active):
        """mon i64 [B, 2, 6, 6], hp [B, 2, 6], active [B, 2, ACTIVE] ->
        tokens [B, 12, d], mask [B, 12]."""
        B = mon.shape[0]
        species, status, moves = mon[..., 0], mon[..., 1], mon[..., 2:]
        p = F.relu(self.species(species) + self.status(status) + hp[..., None] * self.hp + self.pokemon_bias)
        p = F.relu(self.pokemon_fc(p))
        m = self.moves(torch.where(moves < 0, MOVES, moves)).sum(-2)
        m = F.relu(self.moves_fc(F.relu(m + self.moves_bias)))
        a = F.relu(self.active_fc1(F.relu(self.active_fc0(active))))  # [B, 2, active_out]
        tok = self.tok_pokemon(p) + self.tok_moves(m)  # [B, 2, 6, d]
        act = self.tok_active(a)[:, :, None]
        tok = tok + torch.cat([act, act.new_zeros(B, 2, 5, act.shape[-1])], 2)
        tok = tok.reshape(B, 12, -1) + self.pos
        return tok, (species > 0).reshape(B, 12)

    def trunk(self, mon, hp, active):
        """The shared hidden layer [B, hidden] both heads read."""
        x, mask = self.tokens(mon, hp, active)
        for b in self.blocks:
            x = b(x, mask)
        w = mask.view(-1, 2, 6, 1).to(x.dtype)
        pooled = (x.view(-1, 2, 6, self.c.d) * w).sum(2) / w.sum(2).clamp(min=1)  # [B, 2, d]
        return F.relu(self.fc(self.ln(pooled.flatten(1))))

    def value(self, mon, hp, active):
        """forward's value logits alone (search in training's leaves)."""
        return self.value_out(F.relu(self.value_fc(self.trunk(mon, hp, active))))

    def forward(self, mon, hp, active):
        """Returns (value logits [B, 3] (loss, draw, win) for P1, policy
        logits [B, 2, POLICY])."""
        h = self.trunk(mon, hp, active)
        value = self.value_out(F.relu(self.value_fc(h)))
        policy = torch.stack([out(F.relu(fc(h))) for fc, out in zip(self.policy_fc, self.policy_out)], 1)
        return value, policy


def make_net(size: str) -> PiNet:
    return PiNet(SIZES[size])


def config_dict(net: PiNet) -> dict:
    return asdict(net.c)


def legal_logits(policy, index, n):
    """policy [B, 2, POLICY], oak's index [B, 2, CHOICES], n [B, 2] ->
    the choices' logits [B, 2, CHOICES], -1e9 past n. A side with one
    choice gets that one's logit (log-prob 0 after the softmax)."""
    x = policy.gather(-1, index.long())
    legal = torch.arange(CHOICES, device=policy.device) < n[..., None].clamp(min=1)
    return x.float().masked_fill(~legal, -1e9), legal
