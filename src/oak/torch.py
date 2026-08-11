import sys
import struct
import hashlib
import itertools
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import oak.train


class EncodedBattleFrames:
    def __init__(self, frames: oak.train.EncodedBattleFrames):
        self.size = frames.size
        self.k = torch.from_numpy(frames.k)
        self.empirical_policies = torch.from_numpy(frames.empirical_policies)
        self.nash_policies = torch.from_numpy(frames.nash_policies)
        self.empirical_value = torch.from_numpy(frames.empirical_value)
        self.nash_value = torch.from_numpy(frames.nash_value)
        self.score = torch.from_numpy(frames.score)
        self.hp = torch.from_numpy(frames.hp)
        self.pokemon = torch.from_numpy(frames.pokemon)
        self.active = torch.from_numpy(frames.active)
        self.moves = torch.from_numpy(frames.moves)
        self.choice_indices = torch.from_numpy(frames.choice_indices)

    def permute_pokemon(self):
        perms = torch.stack([torch.randperm(5) for _ in range(self.size)], dim=0)
        perms_expanded = perms[:, None, :, None].expand(
            -1, 2, -1, oak.train.pokemon_in_dim
        )
        perms_expanded_moves = perms[:, None, :, None].expand(
            -1, 2, -1, oak.train.moves_in_dim
        )
        self.pokemon[:, :, 1:, :] = torch.gather(
            self.pokemon[:, :, 1:, :], dim=2, index=perms_expanded
        )
        self.moves[:, :, 1:, :] = torch.gather(
            self.moves[:, :, 1:, :], dim=2, index=perms_expanded_moves
        )

    def permute_sides(self, prob=0.5):
        mask = torch.rand(self.size) < prob
        self.k[mask] = self.k[mask].flip(dims=[1])
        self.empirical_policies[mask] = self.empirical_policies[mask].flip(dims=[1])
        self.nash_policies[mask] = self.nash_policies[mask].flip(dims=[1])
        self.empirical_value[mask] = 1 - self.empirical_value[mask]
        self.nash_value[mask] = 1 - self.nash_value[mask]
        self.score[mask] = 1 - self.score[mask]
        self.pokemon[mask] = self.pokemon[mask].flip(dims=[1])
        self.active[mask] = self.active[mask].flip(dims=[1])
        self.moves[mask] = self.moves[mask].flip(dims=[1])
        self.hp[mask] = self.hp[mask].flip(dims=[1])
        self.choice_indices[mask] = self.choice_indices[mask].flip(dims=[1])

    def to(self, device):
        self.k = self.k.to(device)
        self.empirical_policies = self.empirical_policies.to(device)
        self.nash_policies = self.nash_policies.to(device)
        self.empirical_value = self.empirical_value.to(device)
        self.nash_value = self.nash_value.to(device)
        self.score = self.score.to(device)
        self.pokemon = self.pokemon.to(device)
        self.active = self.active.to(device)
        self.moves = self.moves.to(device)
        self.hp = self.hp.to(device)
        self.choice_indices = self.choice_indices.to(device)
        return self


def hash_bytes(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def combine_hash(h1: int, h2: int) -> int:
    # A simple 64-bit mixing function
    return (h1 ^ (h2 + 0x9E3779B97F4A7C15 + (h1 << 6) + (h1 >> 2))) & 0xFFFFFFFFFFFFFFFF


class BuildTrajectories:
    def __init__(self, traj: oak.train.BuildTrajectories, n=None, device="cpu"):
        if n is None:
            n = 31
        self.size = traj.size
        self.action = torch.from_numpy(traj.action[:, :n]).long().to(device)
        self.mask = torch.from_numpy(traj.mask[:, :n]).long().to(device)
        self.policy = torch.from_numpy(traj.policy[:, :n]).float().to(device)
        self.value = torch.from_numpy(traj.value[:, :n]).float().to(device)
        self.score = torch.from_numpy(traj.score[:, :n]).float().to(device)
        self.start = torch.from_numpy(traj.start[:, :n]).long().to(device)
        self.end = torch.from_numpy(traj.end[:, :n]).long().to(device)

    def sample(self, p=1):
        r = torch.rand((self.size,)) < p
        with torch.no_grad():
            self.action = self.action[r].clone()
            self.mask = self.mask[r].clone()
            self.policy = self.policy[r].clone()
            self.value = self.value[r].clone()
            self.score = self.score[r].clone()
            self.start = self.start[r].clone()
            self.end = self.end[r].clone()
        self.size = sum(r).item()


# Networks


class Activation:
    same: int = -1
    none: int = 0
    relu: int = 1
    clamp: int = 2
    relu_scaled: int = 3


class Affine(nn.Module):
    def __init__(self, in_dim, out_dim, activation=Activation.relu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = torch.nn.Linear(in_dim, out_dim)
        self.activation = activation

    def read_parameters(self, f):
        dims = f.read(8)
        in_dim, out_dim = struct.unpack("<II", dims)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = torch.nn.Linear(self.in_dim, self.out_dim)

        self.layer.bias.data.copy_(
            torch.frombuffer(
                bytearray(f.read(self.layer.bias.numel() * 4)), dtype=torch.float32
            )
        )
        self.layer.weight.data.copy_(
            torch.frombuffer(
                bytearray(f.read(self.layer.weight.numel() * 4)), dtype=torch.float32
            ).reshape(self.layer.weight.shape)
        )

    def write_parameters(self, f):
        f.write(struct.pack("<II", self.in_dim, self.out_dim))
        f.write(self.layer.bias.detach().cpu().numpy().astype("f4").tobytes())
        f.write(self.layer.weight.detach().cpu().numpy().astype("f4").tobytes())

    def clamp_parameters(self):
        self.layer.weight.data.clamp_(-2, 2)

    def forward(self, x):
        return apply_activation(self.layer(x), self.activation)

    def hash(self) -> int:
        data = (
            self.layer.weight.detach().cpu().numpy().astype("f4").tobytes()
            + self.layer.bias.detach().cpu().numpy().astype("f4").tobytes()
            + struct.pack("<?II", self.activation, self.in_dim, self.out_dim)
        )
        return hash_bytes(data)


def apply_activation(x, activation):
    """Extracted from the old Affine.forward -- the only piece of that
    class actually specific to the battle net (everything else was just
    nn.Linear bookkeeping pyoaksearch.Network already does).
    """
    if activation == Activation.none:
        return x
    elif activation == Activation.relu:
        return F.relu(x)
    elif activation == Activation.clamp:
        return torch.clamp(x, 0, 1)
    elif activation == Activation.relu_scaled:
        x = torch.relu(x)
        x = x.view(*x.shape[:-1], -1, 32)
        chunk_max = x.amax(dim=-1, keepdim=True).clamp(min=1)
        x = x / chunk_max
        x = x.view(*x.shape[:-2], -1)
        return x
    else:
        assert False, "apply_activation: bad activation"


class TeamBuildingNet(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim_0,
        hidden_dim_1,
        out_dim,
        activation0=Activation.relu,
        activation1=Activation.relu,
        activation2=Activation.relu,
    ):
        super().__init__()
        self.fc0 = Affine(in_dim, hidden_dim_0, activation=activation0)
        self.fc1 = Affine(hidden_dim_0, hidden_dim_1, activation=activation1)
        self.fc2 = Affine(hidden_dim_1, out_dim, activation=activation2)

    def set_activation(self, act):
        self.fc0.activation = act
        self.fc1.activation = act

    def read_parameters(self, f):
        self.fc0.read_parameters(f)
        self.fc1.read_parameters(f)
        self.fc2.read_parameters(f)

    def write_parameters(self, f):
        self.fc0.write_parameters(f)
        self.fc1.write_parameters(f)
        self.fc2.write_parameters(f)

    def forward(self, x):
        return self.fc2(self.fc1(self.fc0(x)))

    def hash(self) -> int:
        h = self.fc0.hash()
        h = combine_hash(h, self.fc1.hash())
        h = combine_hash(h, self.fc2.hash())
        return h


# holds the output of the embedding nets, the input to main net, and value/policy output of main net
class OutputBuffer:
    def __init__(self, buffers: oak.train.OutputBuffer):
        self.size = buffers.size
        self.pokemon_out_dim = buffers.pokemon_out_dim
        self.active_out_dim = buffers.active_out_dim
        self.pokemon = torch.from_numpy(buffers.pokemon)
        self.active = torch.from_numpy(buffers.active)
        self.moves = torch.from_numpy(buffers.moves)
        self.sides = torch.from_numpy(buffers.sides)
        self.value = torch.from_numpy(buffers.value)
        self.logit = torch.from_numpy(buffers.logit)
        self.policy_logit = torch.from_numpy(buffers.policy_logit)
        self.policy = torch.from_numpy(buffers.policy)

    def to(self, device):
        self.pokemon = self.pokemon.to(device)
        self.active = self.active.to(device)
        self.moves = self.moves.to(device)
        self.sides = self.sides.to(device)
        self.value = self.value.to(device)
        self.logit = self.logit.to(device)
        self.policy_logit = self.policy_logit.to(device)
        self.policy = self.policy.to(device)
        return self


# No BattleNets tuple of nn.Module clones either. The battle net's
# Parameters live in exactly one place: a pyoaksearch.Network's Eigen
# storage. Python's only job is wrapping that storage as torch.nn.Parameter
# (for autograd) and running the differentiable forward pass -- there is no
# from-scratch/unbound construction path any more. To build a network, use
# pyoaksearch.Network itself (resize()+initialize(seed), or
# read_parameters(path)); to save/hash it, use its write_parameters(path)/
# hash(); Python never re-implements any of that.
#
# named_parameters()'s keys ARE the architecture (pokemon_net.fc0,
# pokemon_net.fc1, active_net.fc0/fc1, moves_net.fc0/fc1, main_net.fc0/fc1/
# value_fc2/value_fc3/p1_policy_fc2/p1_policy_fc3/p2_policy_fc2/
# p2_policy_fc3 -- see for_each_float_layer() in pyoaksearch.cc); every
# layer's in_dim/out_dim comes from that layer's own weight.shape, never
# from oak.train constants or any other hardcoded dimension.
BattleParams = Dict[str, "tuple[nn.Parameter, nn.Parameter]"]


def bind_live_params(network) -> BattleParams:
    """The only 'binding' step left: wrap every (weight, bias) numpy view
    from network.named_parameters() as an nn.Parameter. No copy -- each
    Parameter aliases `network`'s live Eigen storage directly, so
    optimizer.step() on these mutates `network` in place.

    Same aliasing/lifetime rules as named_parameters() itself: `network`
    must already be at its final shape (resize()+initialize()/
    read_parameters() done) before calling this, and must not be resized
    afterwards -- that reallocates the Eigen matrices these Parameters
    alias, silently. Raises if `network` is quantized (no float layers).
    """
    return {
        name: (nn.Parameter(torch.from_numpy(w)), nn.Parameter(torch.from_numpy(b)))
        for name, (w, b) in network.named_parameters().items()
    }


def battle_parameters(params: BattleParams):
    """Flat iterator of every Parameter in `params`, e.g. for
    torch.optim.Adam(battle_parameters(params), lr=...).
    """
    return itertools.chain.from_iterable(params.values())


def clamp_battle_parameters(params: BattleParams, lo=-2, hi=2):
    # Matches the old Affine.clamp_parameters(): weights only, biases
    # untouched.
    for weight, _bias in params.values():
        weight.data.clamp_(lo, hi)


def _embedding_forward(params: BattleParams, prefix: str, x, activation):
    h = apply_activation(F.linear(x, *params[f"{prefix}.fc0"]), activation)
    return apply_activation(F.linear(h, *params[f"{prefix}.fc1"]), activation)


def _main_forward(params: BattleParams, x, activation, use_policy: bool):
    b0 = apply_activation(F.linear(x, *params["main_net.fc0"]), activation)
    b1 = apply_activation(F.linear(b0, *params["main_net.fc1"]), activation)

    value_h = apply_activation(F.linear(b1, *params["main_net.value_fc2"]), activation)
    value = torch.sigmoid(F.linear(value_h, *params["main_net.value_fc3"]))

    if not use_policy:
        return value, None, None

    p1_h = apply_activation(F.linear(b1, *params["main_net.p1_policy_fc2"]), activation)
    p1_logit = F.linear(p1_h, *params["main_net.p1_policy_fc3"])  # Activation.none
    p2_h = apply_activation(F.linear(b1, *params["main_net.p2_policy_fc2"]), activation)
    p2_logit = F.linear(p2_h, *params["main_net.p2_policy_fc3"])  # Activation.none
    return value, p1_logit, p2_logit


def battle_forward(
    params: BattleParams,
    input: "EncodedBattleFrames",
    output: "OutputBuffer",
    activation=Activation.relu,
    use_policy: bool = True,
):
    """The forward pass formerly known as BattleNetwork.inference(), then
    battle_forward(nets, ...) over EmbeddingNet/MainNet clones. Now just
    F.linear against whatever `params` bind_live_params() handed back --
    `activation` is the one piece of state that isn't recoverable from the
    tensors themselves; pass network.is_clamped and map it yourself
    (Activation.clamp if network.is_clamped else Activation.relu), since
    the caller already has `network` at the point it built `params`.
    """
    size = min(input.size, output.size)
    output.pokemon[:size] = _embedding_forward(
        params, "pokemon_net", input.pokemon[:size, :, :], activation
    )
    output.active[:size] = _embedding_forward(
        params, "active_net", input.active[:size, :, :], activation
    )
    output.moves[:size] = _embedding_forward(
        params, "moves_net", input.moves[:size, :, :], activation
    )
    # mask output for hp
    output.pokemon[:size] *= (input.hp[:size, :, :] != 0).float()
    output.active[:size] *= (input.hp[:size, :, :1] != 0).float()
    output.moves[:size] *= (input.hp[:size, :, :] != 0).float()

    active_out_dim = output.active_out_dim
    output.sides[:size, :, :, :active_out_dim] = output.active[:size]
    output.sides[:size, :, :, active_out_dim:] = torch.cat(
        [output.pokemon, output.moves], dim=3
    ).view(size, 2, 1, -1)
    # side_out_dim taken from the buffer's own allocated width, not
    # recomputed from any dim constant.
    side_out_dim = output.sides.shape[-1]
    battle = output.sides[:size].view(size, 2 * side_out_dim)

    value, p1_logit, p2_logit = _main_forward(params, battle, activation, use_policy)
    if use_policy:
        output.value[:size] = value
        output.logit[:size, 0, :-1] = p1_logit
        output.logit[:size, 1, :-1] = p2_logit
    else:
        output.value = value

    output.policy_logit[:size, 0] = torch.gather(
        output.logit[:size, 0], 1, input.choice_indices[:size, 0]
    )
    output.policy_logit[:size, 1] = torch.gather(
        output.logit[:size, 1], 1, input.choice_indices[:size, 1]
    )


class BuildNetwork(nn.Module):
    def __init__(
        self,
        policy_hidden_dim=oak.train.build_policy_hidden_dim,
        value_hidden_dim=oak.train.build_value_hidden_dim,
    ):
        super().__init__()
        self.policy_net = TeamBuildingNet(
            len(oak.train.species_move_list),
            policy_hidden_dim,
            policy_hidden_dim,
            len(oak.train.species_move_list),
            Activation.relu,
            Activation.relu,
            Activation.none,
        )
        self.value_net = TeamBuildingNet(
            len(oak.train.species_move_list),
            value_hidden_dim,
            value_hidden_dim,
            1,
            Activation.relu,
            Activation.relu,
            Activation.none,
        )

    def read_parameters(self, f):
        self.policy_net.read_parameters(f)
        self.value_net.read_parameters(f)

    def write_parameters(self, f):
        self.policy_net.write_parameters(f)
        self.value_net.write_parameters(f)

    def forward(self, x):
        return self.policy_net.forward(x), self.value_net.forward(x)
