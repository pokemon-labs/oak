import sys
import os
import struct
import hashlib
import itertools
from collections import namedtuple
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

    def bind_live(self, weight_view, bias_view):
        """Replace this layer's Parameters with zero-copy views over a live
        pyoaksearch.Network's Eigen buffers (as returned by
        Network.named_parameters()/weights()/biases()).

        After this call, self.layer.weight / .bias ARE the C++ network's
        storage -- no torch.Tensor copy exists anywhere. optimizer.step()
        mutates the C++ network in place; no read_parameters()/
        write_parameters() round trip is needed to keep them in sync.

        Caller must not call resize() on the source Network afterwards --
        that reallocates the underlying Eigen matrices and silently
        invalidates weight_view/bias_view (see pyoaksearch.cc).
        """
        out_dim, in_dim = weight_view.shape
        assert (out_dim, in_dim) == (self.out_dim, self.in_dim), (
            f"bind_live: shape mismatch, layer is ({self.out_dim}, {self.in_dim}) "
            f"but view is ({out_dim}, {in_dim})"
        )
        # torch.from_numpy shares memory; it does not copy, even when the
        # numpy view is strided (e.g. embedding nets' ColMajor fc0 layers --
        # see affine_weights_view in pyoaksearch.cc). Non-contiguous
        # Parameters work fine with nn.Linear/F.linear; PyTorch will
        # transparently materialize a contiguous copy where needed for a
        # given op, and autograd still routes gradients back to this
        # Parameter's original storage, so optimizer updates land in the
        # right place.
        self.layer.weight = nn.Parameter(torch.from_numpy(weight_view))
        self.layer.bias = nn.Parameter(torch.from_numpy(bias_view))

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
        x = self.layer(x)
        if self.activation == Activation.none:
            return x
        elif self.activation == Activation.relu:
            return torch.nn.functional.relu(x)
        elif self.activation == Activation.clamp:
            return torch.clamp(x, 0, 1)
        elif self.activation == Activation.relu_scaled:
            x = torch.relu(x)
            x = x.view(*x.shape[:-1], -1, 32)
            chunk_max = x.amax(dim=-1, keepdim=True).clamp(min=1)
            x = x / chunk_max
            x = x.view(*x.shape[:-2], -1)
            return x
        else:
            assert False, "Affine: Bad activation"

    def hash(self) -> int:
        data = (
            self.layer.weight.detach().cpu().numpy().astype("f4").tobytes()
            + self.layer.bias.detach().cpu().numpy().astype("f4").tobytes()
            + struct.pack("<?II", self.activation, self.in_dim, self.out_dim)
        )
        return hash_bytes(data)


class EmbeddingNet(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        activation0=Activation.relu,
        activation1=Activation.relu,
    ):
        super().__init__()
        self.fc0 = Affine(in_dim, hidden_dim, activation=activation0)
        self.fc1 = Affine(hidden_dim, out_dim, activation=activation1)

    def set_activation(self, act):
        self.fc0.activation = act
        self.fc1.activation = act

    def bind_live(self, params: Dict[str, "tuple"], prefix: str):
        self.fc0.bind_live(*params[f"{prefix}.fc0"])
        self.fc1.bind_live(*params[f"{prefix}.fc1"])

    def read_parameters(self, f):
        self.fc0.read_parameters(f)
        self.fc1.read_parameters(f)

    def write_parameters(self, f):
        self.fc0.write_parameters(f)
        self.fc1.write_parameters(f)

    def clamp_parameters(self):
        self.fc0.clamp_parameters()
        self.fc1.clamp_parameters()

    def forward(self, x):
        return self.fc1(self.fc0(x))

    def hash(self) -> int:
        h = self.fc0.hash()
        h = combine_hash(h, self.fc1.hash())
        return h


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


class MainNet(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        value_hidden_dim,
        policy_hidden_dim,
        policy_out_dim,
        activation=Activation.relu,
    ):
        super().__init__()
        self.fc0 = Affine(in_dim, hidden_dim, activation)
        self.fc1 = Affine(hidden_dim, hidden_dim, activation)
        self.value_fc1 = Affine(hidden_dim, value_hidden_dim, activation)
        self.value_fc2 = Affine(value_hidden_dim, 1, Activation.none)
        self.policy1_fc1 = Affine(hidden_dim, policy_hidden_dim, activation)
        self.policy1_fc2 = Affine(policy_hidden_dim, policy_out_dim, Activation.none)
        self.policy2_fc1 = Affine(hidden_dim, policy_hidden_dim, activation)
        self.policy2_fc2 = Affine(policy_hidden_dim, policy_out_dim, Activation.none)

    def set_activation(self, act):
        self.fc0.activation = act
        self.fc1.activation = act
        self.value_fc1.activation = act
        self.policy1_fc1.activation = act
        self.policy2_fc1.activation = act

    # Maps this module's attribute names to the layer names used by
    # pyoaksearch.Network.named_parameters() (see for_each_float_layer in
    # pyoaksearch.cc). The trunk (fc0/fc1) is numbered fc0/fc1 on both
    # sides; each head restarts its own fc2/fc3 numbering C++-side instead
    # of continuing fc1's local _fc1/_fc2 naming.
    _CPP_LAYER_NAMES = {
        "fc0": "main_net.fc0",
        "fc1": "main_net.fc1",
        "value_fc1": "main_net.value_fc2",
        "value_fc2": "main_net.value_fc3",
        "policy1_fc1": "main_net.p1_policy_fc2",
        "policy1_fc2": "main_net.p1_policy_fc3",
        "policy2_fc1": "main_net.p2_policy_fc2",
        "policy2_fc2": "main_net.p2_policy_fc3",
    }

    def bind_live(self, params: Dict[str, "tuple"]):
        for attr, cpp_name in self._CPP_LAYER_NAMES.items():
            getattr(self, attr).bind_live(*params[cpp_name])

    def read_parameters(self, f):
        self.fc0.read_parameters(f)
        self.fc1.read_parameters(f)
        self.value_fc1.read_parameters(f)
        self.value_fc2.read_parameters(f)
        self.policy1_fc1.read_parameters(f)
        self.policy1_fc2.read_parameters(f)
        self.policy2_fc1.read_parameters(f)
        self.policy2_fc2.read_parameters(f)
        pos = f.tell()
        f.seek(0, 2)
        end = f.tell()
        f.seek(pos)
        assert pos == end

    def write_parameters(self, f):
        self.fc0.write_parameters(f)
        self.fc1.write_parameters(f)
        self.value_fc1.write_parameters(f)
        self.value_fc2.write_parameters(f)
        self.policy1_fc1.write_parameters(f)
        self.policy1_fc2.write_parameters(f)
        self.policy2_fc1.write_parameters(f)
        self.policy2_fc2.write_parameters(f)

    def clamp_parameters(self):
        self.fc0.clamp_parameters()
        self.fc1.clamp_parameters()
        self.value_fc1.clamp_parameters()
        self.value_fc2.clamp_parameters()
        self.policy1_fc1.clamp_parameters()
        self.policy1_fc2.clamp_parameters()
        self.policy2_fc1.clamp_parameters()
        self.policy2_fc2.clamp_parameters()

    def forward(self, x):
        b0 = self.fc0(x)
        b1 = self.fc1(b0)
        value_b1 = self.value_fc1(b1)
        value_b2 = self.value_fc2(value_b1)
        value = torch.sigmoid(value_b2)
        p1_policy_b1 = self.policy1_fc1(b1)
        p1_policy_b2 = self.policy1_fc2(p1_policy_b1)
        p2_policy_b1 = self.policy2_fc1(b1)
        p2_policy_b2 = self.policy2_fc2(p2_policy_b1)
        return value, p1_policy_b2, p2_policy_b2

    def forward_value_only(self, x):
        b0 = self.fc0(x)
        b1 = self.fc1(b0)
        value_b1 = self.value_fc1(b1)
        value_b2 = self.value_fc2(value_b1)
        value = torch.sigmoid(value_b2)
        return value

    def hash(self) -> int:
        h = self.fc0.hash()
        for sub in [
            self.fc1,
            self.value_fc1,
            self.value_fc2,
            self.policy1_fc1,
            self.policy1_fc2,
            self.policy2_fc1,
            self.policy2_fc2,
        ]:
            h = combine_hash(h, sub.hash())
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


# BattleNetwork is gone. There is no class holding pokemon_net/active_net/
# moves_net/main_net and a forward()/inference() method anymore -- just a
# plain, behaviorless tuple of the four sub-nets' Parameters, and free
# functions that operate on it. This matches how the C++ side already
# works: Search::Network is a plain data owner, not a hand-written forward
# pass -- the "forward pass" for TRAINING lives here, functionally, and
# gets its Parameters either freshly initialized or bound in place onto
# a live pyoaksearch.Network via bind_battle_nets_live().
BattleNets = namedtuple("BattleNets", ["pokemon_net", "active_net", "moves_net", "main_net"])


def build_battle_nets(
    phd=oak.train.pokemon_hidden_dim,
    ahd=oak.train.active_hidden_dim,
    mhd=oak.train.moves_hidden_dim,
    pod=oak.train.pokemon_out_dim,
    aod=oak.train.active_out_dim,
    mod=oak.train.moves_out_dim,
    hd=oak.train.hidden_dim,
    vhd=oak.train.value_hidden_dim,
    pohd=oak.train.policy_hidden_dim,
    activation=Activation.relu,
) -> BattleNets:
    """Construct a fresh (randomly-initialized-by-nn.Linear) BattleNets.
    Equivalent to the old BattleNetwork(...) constructor, minus the class.
    """
    side_out_dim = aod + 6 * (pod + mod)
    pokemon_net = EmbeddingNet(
        oak.train.pokemon_in_dim, phd, pod, activation, activation
    )
    active_net = EmbeddingNet(oak.train.active_in_dim, ahd, aod, activation, activation)
    moves_net = EmbeddingNet(oak.train.moves_in_dim, mhd, mod, activation, activation)
    main_net = MainNet(
        2 * side_out_dim, hd, vhd, pohd, oak.train.policy_out_dim, activation=activation
    )
    return BattleNets(pokemon_net, active_net, moves_net, main_net)


def bind_battle_nets_live(nets: BattleNets, network) -> BattleNets:
    """Bind every Affine layer across `nets` to zero-copy views over
    `network`'s (a pyoaksearch.Network) live Eigen storage. See
    Affine.bind_live() for the aliasing/lifetime rules -- in particular:
    do not call network.resize() again after this, and `network` must not
    be quantized. Mutates `nets` in place and returns it for chaining.
    """
    params = network.named_parameters()
    nets.pokemon_net.bind_live(params, "pokemon_net")
    nets.active_net.bind_live(params, "active_net")
    nets.moves_net.bind_live(params, "moves_net")
    nets.main_net.bind_live(params)
    return nets


def battle_parameters(nets: BattleNets):
    """Flat iterator of every torch.nn.Parameter in `nets`, e.g. for
    torch.optim.Adam(battle_parameters(nets), lr=...).
    """
    return itertools.chain(
        nets.pokemon_net.parameters(),
        nets.active_net.parameters(),
        nets.moves_net.parameters(),
        nets.main_net.parameters(),
    )


def set_battle_activation(nets: BattleNets, act):
    nets.pokemon_net.set_activation(act)
    nets.active_net.set_activation(act)
    nets.moves_net.set_activation(act)
    nets.main_net.set_activation(act)


def clamp_battle_parameters(nets: BattleNets):
    nets.pokemon_net.clamp_parameters()
    nets.active_net.clamp_parameters()
    nets.moves_net.clamp_parameters()
    nets.main_net.clamp_parameters()


def read_battle_parameters(nets: BattleNets, f) -> BattleNets:
    """Legacy struct-packed loader. Do not call this on a `nets` that was
    passed through bind_battle_nets_live() -- its Parameters ARE a live
    C++ Network's storage, not a loadable copy; load new weights via
    network.read_parameters(path) on the (unbound) Network instead, before
    binding.
    """
    header = f.read(8)
    act = struct.unpack("<BBBBBBBB", header)[0]
    set_battle_activation(nets, act + 1)
    nets.pokemon_net.read_parameters(f)
    nets.active_net.read_parameters(f)
    nets.moves_net.read_parameters(f)
    nets.main_net.read_parameters(f)
    return nets


def write_battle_parameters(nets: BattleNets, f):
    """Legacy struct-packed writer, for `nets` NOT bound to a live C++
    Network. If `nets` is bound (bind_battle_nets_live()), call
    network.write_parameters(path) on the underlying Network directly
    instead -- same on-disk format, written straight from the Eigen
    buffers, but it takes a path (not an open file object).
    """
    # fc0's activation stands in for "the" network-wide hidden activation
    # (matches the old BattleNetwork.activation attribute / set_activation()
    # contract: every hidden layer shares one activation, output layers
    # keep Activation.none regardless).
    f.write(struct.pack("<Q", nets.pokemon_net.fc0.activation - 1))
    nets.pokemon_net.write_parameters(f)
    nets.active_net.write_parameters(f)
    nets.moves_net.write_parameters(f)
    nets.main_net.write_parameters(f)


def hash_battle_parameters(nets: BattleNets) -> int:
    """blake2b-based hash, for `nets` NOT bound to a live C++ Network. If
    `nets` is bound, call network.hash() on the underlying Network instead
    -- NOT bit-compatible with this one (FNV-1a vs. blake2b), do not
    compare the two.
    """
    h = nets.pokemon_net.hash()
    h = combine_hash(h, nets.active_net.hash())
    h = combine_hash(h, nets.main_net.hash())
    return h & 0xFFFFFFFFFFFFFFFF


def battle_forward(
    nets: BattleNets,
    input: "EncodedBattleFrames",
    output: "OutputBuffer",
    use_policy: bool = True,
):
    """The forward pass formerly known as BattleNetwork.inference(). Free
    function, no class, no `self` -- operates on whatever `nets` you hand
    it, freshly built or bound live onto a pyoaksearch.Network.
    """
    size = min(input.size, output.size)
    output.pokemon[:size] = nets.pokemon_net.forward(input.pokemon[:size, :, :])
    output.active[:size] = nets.active_net.forward(input.active[:size, :, :])
    output.moves[:size] = nets.moves_net.forward(input.moves[:size, :, :])
    # mask output for hp
    output.pokemon[:size] *= (input.hp[:size, :, :] != 0).float()
    output.active[:size] *= (input.hp[:size, :, :1] != 0).float()
    output.moves[:size] *= (input.hp[:size, :, :] != 0).float()

    active_out_dim = output.active_out_dim
    output.sides[:size, :, :, :active_out_dim] = output.active[:size]
    output.sides[:size, :, :, active_out_dim:] = torch.cat(
        [output.pokemon, output.moves], dim=3
    ).view(size, 2, 1, -1)
    # side_out_dim taken from the buffer's own allocated width rather than
    # recomputed from aod/pod/mod -- one less place for those to drift out
    # of sync with what OutputBuffer was actually constructed with.
    side_out_dim = output.sides.shape[-1]
    battle = output.sides[:size].view(size, 2 * side_out_dim)

    if use_policy:
        (
            output.value[:size],
            output.logit[:size, 0, :-1],
            output.logit[:size, 1, :-1],
        ) = nets.main_net.forward(battle)
    else:
        output.value = nets.main_net.forward_value_only(battle)

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
