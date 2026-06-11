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
        self.choice_indices = torch.from_numpy(frames.choice_indices)

    def permute_pokemon(self):
        perms = torch.stack([torch.randperm(5) for _ in range(self.size)], dim=0)
        perms_expanded = perms[:, None, :, None].expand(
            -1, 2, -1, oak.train.pokemon_in_dim
        )
        self.pokemon[:, :, 1:, :] = torch.gather(
            self.pokemon[:, :, 1:, :], dim=2, index=perms_expanded
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
        self.active_pokemon = torch.from_numpy(buffers.active_pokemon)
        self.sides = torch.from_numpy(buffers.sides)
        self.value = torch.from_numpy(buffers.value)
        self.logit = torch.from_numpy(buffers.logit)
        self.policy_logit = torch.from_numpy(buffers.policy_logit)
        self.policy = torch.from_numpy(buffers.policy)

    def to(self, device):
        self.pokemon = self.pokemon.to(device)
        self.active_pokemon = self.active_pokemon.to(device)
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
    policy_out_dim = oak.train.policy_out_dim

    def __init__(
        self,
        phd=oak.train.pokemon_hidden_dim,
        ahd=oak.train.active_hidden_dim,
        pod=oak.train.pokemon_out_dim,
        aod=oak.train.active_out_dim,
        hd=oak.train.hidden_dim,
        vhd=oak.train.value_hidden_dim,
        pohd=oak.train.policy_hidden_dim,
        activation=Activation.relu,
    ):
        super().__init__()
        self.pokemon_hidden_dim = phd
        self.active_hidden_dim = ahd
        self.pokemon_out_dim = pod
        self.active_out_dim = aod
        self.side_out_dim = (1 + aod) + 5 * (1 + pod)
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
        self.main_net = MainNet(
            2 * self.side_out_dim,
            self.hidden_dim,
            self.value_hidden_dim,
            self.policy_hidden_dim,
            self.policy_out_dim,
            activation=activation,
        )

    def set_activation(self, act):
        self.activation = act
        self.pokemon_net.set_activation(act)
        self.active_net.set_activation(act)
        self.main_net.set_activation(act)

    def read_parameters(self, f):
        header = f.read(8)
        act = struct.unpack("<BBBBBBBB", header)[0]
        self.set_activation(act + 1)
        self.pokemon_net.read_parameters(f)
        self.active_net.read_parameters(f)
        self.main_net.read_parameters(f)

    def write_parameters(self, f):
        f.write(struct.pack("<Q", self.activation - 1))
        self.pokemon_net.write_parameters(f)
        self.active_net.write_parameters(f)
        self.main_net.write_parameters(f)

    def clamp_parameters(self):
        self.pokemon_net.clamp_parameters()
        self.active_net.clamp_parameters()
        self.main_net.clamp_parameters()

    def inference(
        self, input: EncodedBattleFrames, output: OutputBuffer, use_policy: bool = True
    ):
        size = min(input.size, output.size)
        output.pokemon[:size] = self.pokemon_net.forward(input.pokemon[:size, :, 1:])
        output.active_pokemon[:size] = self.active_net.forward(
            torch.cat([input.active[:size], input.pokemon[:size, :, :1]], dim=3)
        )
        # mask output for hp
        output.pokemon[:size] *= (input.hp[:size, :, 1:] != 0).float()
        output.active_pokemon[:size] *= (input.hp[:size, :, :1] != 0).float()
        # active hp
        output.sides[:size, :, :, 0] = input.hp[:size, :, :1, 0]
        # active word
        output.sides[:size, :, :, 1 : self.active_out_dim + 1] = output.active_pokemon[
            :size
        ]
        # pokemon hp/word
        pokemon_flat = torch.cat(
            (input.hp[:size, :, 1:], output.pokemon[:size]), dim=3
        ).view(size, 2, 1, 5 * (1 + self.pokemon_out_dim))
        output.sides[:size, :, :, 1 + self.active_out_dim :] = pokemon_flat[:size]
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

        # print(output.sides)

    def hash(self) -> int:
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
