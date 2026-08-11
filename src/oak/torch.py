import sys
import os
import struct
import hashlib
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


class BattleNetwork(torch.nn.Module):
    # only remaining hard-coded dims
    pokemon_in_dim = oak.train.pokemon_in_dim
    active_in_dim = oak.train.active_in_dim
    moves_in_dim = oak.train.moves_in_dim
    policy_out_dim = oak.train.policy_out_dim

    def __init__(
        self,
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
    ):
        super().__init__()
        self.pokemon_hidden_dim = phd
        self.active_hidden_dim = ahd
        self.moves_hidden_dim = mhd
        self.pokemon_out_dim = pod
        self.active_out_dim = aod
        self.moves_out_dim = mod
        self.side_out_dim = aod + 6 * (pod + mod)
        self.hidden_dim = hd
        self.value_hidden_dim = vhd
        self.policy_hidden_dim = pohd
        self.activation = activation

        self.pokemon_net = EmbeddingNet(
            self.pokemon_in_dim,
            self.pokemon_hidden_dim,
            self.pokemon_out_dim,
            activation,
            activation,
        )
        self.active_net = EmbeddingNet(
            self.active_in_dim,
            self.active_hidden_dim,
            self.active_out_dim,
            activation,
            activation,
        )
        self.moves_net = EmbeddingNet(
            self.moves_in_dim,
            self.moves_hidden_dim,
            self.moves_out_dim,
            activation,
            activation,
        )
        self.main_net = MainNet(
            2 * self.side_out_dim,
            self.hidden_dim,
            self.value_hidden_dim,
            self.policy_hidden_dim,
            self.policy_out_dim,
            activation=activation,
        )

        # Set once bind_live() succeeds. When present, write_parameters()/
        # hash() delegate to the C++ Network object instead of the legacy
        # Python re-implementations below, and read_parameters() is not a
        # valid way to load new weights any more (the params ARE the C++
        # network's storage; use pyoaksearch.Network.read_parameters()
        # against the bound network instead, before binding).
        self._bound_network = None

    def bind_live(self, network):
        """Bind every Affine layer's weight/bias Parameters to zero-copy
        views over `network`'s (a pyoaksearch.Network) live Eigen storage.

        This is the intended replacement for the read_parameters()/
        write_parameters()/hash() struct.pack byte-plumbing below: once
        bound, training this module (e.g. loss.backward(); optimizer.step())
        mutates `network`'s C++ weights directly -- the same object usable
        for search/MCTS -- with no serialize-to-disk-and-reload step in the
        training loop.

        Preconditions (caller's responsibility, not checked here beyond
        what named_parameters() itself checks):
          - `network` must already be at its final architecture (resize()
            called, and read_parameters()/initialize() done) BEFORE this
            call. Do not call network.resize() again afterwards -- that
            reallocates the Eigen matrices and silently invalidates every
            Parameter bound here (dangling storage, not a Python exception).
          - `network` must not be quantized (named_parameters() raises for
            quantized networks -- there are no float layers to view).
          - If `network` is concurrently used for search (e.g. self-play
            workers calling forward_side()/run() on it) while this module
            is being trained, the caller is responsible for synchronizing
            reads vs. writes -- there is no locking here, and none in the
            C++ side either.
        """
        params = network.named_parameters()
        self.pokemon_net.bind_live(params, "pokemon_net")
        self.active_net.bind_live(params, "active_net")
        self.moves_net.bind_live(params, "moves_net")
        self.main_net.bind_live(params)
        self._bound_network = network
        return self

    @property
    def is_bound(self) -> bool:
        return self._bound_network is not None

    def set_activation(self, act):
        self.activation = act
        self.pokemon_net.set_activation(act)
        self.active_net.set_activation(act)
        self.moves_net.set_activation(act)
        self.main_net.set_activation(act)

    def read_parameters(self, f):
        assert not self.is_bound, (
            "read_parameters: this module is bound to a live C++ Network "
            "(bind_live()); its Parameters ARE that network's storage, so "
            "loading new weights here would either no-op onto a stale copy "
            "or corrupt the binding. Load new weights via "
            "network.read_parameters(path) on the (unbound) Network before "
            "calling bind_live(), instead."
        )
        header = f.read(8)
        act = struct.unpack("<BBBBBBBB", header)[0]
        self.set_activation(act + 1)
        self.pokemon_net.read_parameters(f)
        self.active_net.read_parameters(f)
        self.moves_net.read_parameters(f)
        self.main_net.read_parameters(f)

    def write_parameters(self, f):
        if self.is_bound:
            # pyoaksearch.Network.write_parameters() takes a path (it opens
            # its own std::ofstream), not a file object like the legacy
            # Python path below -- it isn't a drop-in signature match, so
            # callers writing `with open(p, "wb") as f: net.write_parameters(f)`
            # need `net.write_parameters(p)` instead when bound. Verified
            # the on-disk format itself is identical either way: 8-byte
            # header (byte 0 = 0 relu / 1 clamp), then each float layer as
            # (in_dim: u32, out_dim: u32, biases: f32[out_dim],
            # weights: f32[out_dim*in_dim] row-major), same layer order as
            # for_each_float_layer() in pyoaksearch.cc -- so files written
            # bound vs. unbound are interchangeable.
            if not isinstance(f, (str, os.PathLike)):
                raise TypeError(
                    "write_parameters: this module is bound to a live C++ "
                    "Network, whose write_parameters() writes directly to a "
                    "path (it opens its own file), not an already-open file "
                    "object. Pass a path string/PathLike, not a file handle."
                )
            self._bound_network.write_parameters(str(f))
            return
        f.write(struct.pack("<Q", self.activation - 1))
        self.pokemon_net.write_parameters(f)
        self.active_net.write_parameters(f)
        self.moves_net.write_parameters(f)
        self.main_net.write_parameters(f)

    def clamp_parameters(self):
        self.pokemon_net.clamp_parameters()
        self.active_net.clamp_parameters()
        self.moves_net.clamp_parameters()
        self.main_net.clamp_parameters()

    # TODO
    def inference(
        self, input: EncodedBattleFrames, output: OutputBuffer, use_policy: bool = True
    ):
        size = min(input.size, output.size)
        output.pokemon[:size] = self.pokemon_net.forward(input.pokemon[:size, :, :])
        output.active[:size] = self.active_net.forward(input.active[:size, :, :])
        output.moves[:size] = self.moves_net.forward(input.moves[:size, :, :])
        # mask output for hp
        output.pokemon[:size] *= (input.hp[:size, :, :] != 0).float()
        output.active[:size] *= (input.hp[:size, :, :1] != 0).float()
        output.moves[:size] *= (input.hp[:size, :, :] != 0).float()

        output.sides[:size, :, :, : self.active_out_dim] = output.active[:size]
        output.sides[:size, :, :, self.active_out_dim :] = torch.cat(
            [output.pokemon, output.moves], dim=3
        ).view(size, 2, 1, -1)
        battle = output.sides[:size].view(size, 2 * self.side_out_dim)

        if use_policy:
            (
                output.value[:size],
                output.logit[:size, 0, :-1],
                output.logit[:size, 1, :-1],
            ) = self.main_net.forward(battle)
        else:
            output.value = self.main_net.forward_value_only(battle)

        output.policy_logit[:size, 0] = torch.gather(
            output.logit[:size, 0], 1, input.choice_indices[:size, 0]
        )
        output.policy_logit[:size, 1] = torch.gather(
            output.logit[:size, 1], 1, input.choice_indices[:size, 1]
        )

    def hash(self) -> int:
        if self.is_bound:
            # NOTE: this is the C++ FNV-1a hash (Network::hash() in
            # pyoaksearch.cc), NOT the blake2b-based hash below. They are
            # different algorithms and will NOT agree on the same weights --
            # do not compare a bound network's hash() against one recorded
            # from an unbound/legacy-loaded BattleNetwork as a "did the
            # weights change" check across that boundary.
            return self._bound_network.hash()
        h = self.pokemon_net.hash()
        h = combine_hash(h, self.active_net.hash())
        h = combine_hash(h, self.main_net.hash())
        return h & 0xFFFFFFFFFFFFFFFF


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
