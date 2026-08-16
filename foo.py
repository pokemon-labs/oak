import argparse

parser = argparse.ArgumentParser(description="Oak Tutorial")
parser.add_argument(
    "--data-path",
    default=None,
    type=str,
    help="Path to data file or directory containing data files",
)
parser.add_argument(
    "--network", default=None, type=str, help="Path to battle/build network"
)
parser.add_argument(
    "--build-network", default=None, type=str, help="Path to build network"
)
fn_parser = parser.add_argument_group("")
fn_parser.add_argument("--games", default=None, type=int)
fn_parser.add_argument("--discrete", default=False, type=bool)
fn_parser.add_argument("--eps", default=0.0, type=float)
fn_parser.add_argument("--cache", default=True, type=bool)
args = parser.parse_args()
assert args.data_path, "Provide path to data file to inspect"
assert args.network, "Provide path to network to test"

import torch
import oak
import oak.torch
import oak.search

network = oak.search.Network()
network.read_parameters(args.network)

torch_network = oak.torch.BattleNetwork()
with open(args.network, "rb") as f:
    torch_network.read_parameters(f)

assert(network.activation() == torch_network.activation)
# Check weights/biases
cpp_params = network.named_parameters()
assert torch.all(cpp_params["pokemon_net.fc0"][0] == torch_network.pokemon_net.fc0.layer.weight)
assert torch.all(cpp_params["pokemon_net.fc0"][1] == torch_network.pokemon_net.fc0.layer.bias)
assert torch.all(cpp_params["pokemon_net.fc1"][0] == torch_network.pokemon_net.fc1.layer.weight)
assert torch.all(cpp_params["pokemon_net.fc1"][1] == torch_network.pokemon_net.fc1.layer.bias)
assert torch.all(cpp_params["active_net.fc0"][0] == torch_network.active_net.fc0.layer.weight)
assert torch.all(cpp_params["active_net.fc0"][1] == torch_network.active_net.fc0.layer.bias)
assert torch.all(cpp_params["active_net.fc1"][0] == torch_network.active_net.fc1.layer.weight)
assert torch.all(cpp_params["active_net.fc1"][1] == torch_network.active_net.fc1.layer.bias)
assert torch.all(cpp_params["moves_net.fc0"][0] == torch_network.moves_net.fc0.layer.weight)
assert torch.all(cpp_params["moves_net.fc0"][1] == torch_network.moves_net.fc0.layer.bias)
assert torch.all(cpp_params["moves_net.fc1"][0] == torch_network.moves_net.fc1.layer.weight)
assert torch.all(cpp_params["moves_net.fc1"][1] == torch_network.moves_net.fc1.layer.bias)
assert torch.all(cpp_params["main_net.fc0"][0] == torch_network.main_net.fc0.layer.weight)
assert torch.all(cpp_params["main_net.fc0"][1] == torch_network.main_net.fc0.layer.bias)
assert torch.all(cpp_params["main_net.fc1"][0] == torch_network.main_net.fc1.layer.weight)
assert torch.all(cpp_params["main_net.fc1"][1] == torch_network.main_net.fc1.layer.bias)
assert torch.all(cpp_params["main_net.value_fc2"][0] == torch_network.main_net.value_fc1.layer.weight)
assert torch.all(cpp_params["main_net.value_fc2"][1] == torch_network.main_net.value_fc1.layer.bias)
assert torch.all(cpp_params["main_net.value_fc3"][0] == torch_network.main_net.value_fc2.layer.weight)
assert torch.all(cpp_params["main_net.value_fc3"][1] == torch_network.main_net.value_fc2.layer.bias)
assert torch.all(cpp_params["main_net.p1_policy_fc2"][0] == torch_network.main_net.policy1_fc1.layer.weight)
assert torch.all(cpp_params["main_net.p1_policy_fc2"][1] == torch_network.main_net.policy1_fc1.layer.bias)
assert torch.all(cpp_params["main_net.p1_policy_fc3"][0] == torch_network.main_net.policy1_fc2.layer.weight)
assert torch.all(cpp_params["main_net.p1_policy_fc3"][1] == torch_network.main_net.policy1_fc2.layer.bias)
assert torch.all(cpp_params["main_net.p2_policy_fc2"][0] == torch_network.main_net.policy2_fc1.layer.weight)
assert torch.all(cpp_params["main_net.p2_policy_fc2"][1] == torch_network.main_net.policy2_fc1.layer.bias)
assert torch.all(cpp_params["main_net.p2_policy_fc3"][0] == torch_network.main_net.policy2_fc2.layer.weight)
assert torch.all(cpp_params["main_net.p2_policy_fc3"][1] == torch_network.main_net.policy2_fc2.layer.bias)

if args.discrete:
    network.quantize()

buffer_list = oak.train.read_battle_data(args.data_path)
max_games = min(args.games or len(buffer_list), len(buffer_list))

for buffer, n_frames in buffer_list[:max_games]:
    encoded_frames = oak.train.EncodedBattleFrames.from_bytes(buffer, n_frames)
    encoded_frames_torch = oak.torch.EncodedBattleFrames(encoded_frames)
    o1 = oak.train.OutputBuffer(encoded_frames.size)
    python_output = oak.torch.OutputBuffer(o1)
    torch_network.inference(encoded_frames_torch, python_output)
    python_output.policy_logit[torch.isneginf(python_output.policy_logit)] = 0.0

    battle_frames = oak.train.BattleFrames.from_bytes(buffer, n_frames)
    battle = oak.Battle(battle_frames.battle[0].tobytes())

    oak.search.run(
        battle,
        oak.Durations(),
        oak.search.Iterations(0),
        oak.search.PUCB(1.0),
        oak.search.Node(),
        network,
    )
    # p1_cache = oak.search.SideCache()
    # p2_cache = oak.search.SideCache()
    # for i in range(6):
    #     p1_cache.precompute(network, battle.side(0), i)
    #     p2_cache.precompute(network, battle.side(1), i)

    o2 = oak.search.cpp_inference(battle_frames, network, oak.search.Iterations(0), 1000000)
    cpp_output = oak.torch.OutputBuffer(o2)

    value_diff = torch.abs(python_output.value - cpp_output.value)
    logit_diff = torch.abs(python_output.policy_logit - cpp_output.policy_logit)

    print(f"Max value diff: {torch.max(value_diff).item()}")
    print(f"Max logit diff: {torch.max(logit_diff).item()}")
    print(f"Avg value diff: {torch.mean(value_diff).item()}")
    print(f"Avg logit diff: {(torch.sum(logit_diff) / torch.sum(logit_diff != 0))}")

    assert torch.all(value_diff < args.eps)
    assert torch.all(logit_diff < args.eps)

print(
    f"CPP and Python agree on value/policy inference for the first {max_games} games of the data file."
)
