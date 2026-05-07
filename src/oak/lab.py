import sys

import argparse

parser = argparse.ArgumentParser(description="Oak Tutorial")
parser.add_argument("main", default=None, type=str, help="Utility to run")
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


def test_consistency():
    fn_parser = parser.add_argument_group("")
    fn_parser.add_argument("--games", default=None, type=int)
    fn_parser.add_argument("--discrete", default=False, type=bool)
    fn_parser.add_argument("--eps", default=0.0, type=float)
    args = parser.parse_args()
    assert args.data_path, "Provide path to data file to inspect"
    assert args.network, "Provide path to network to test"

    import torch
    import oak
    import oak.torch

    activation = (
        oak.torch.Activation.clamp if args.discrete else oak.torch.Activation.relu
    )
    network = oak.torch.BattleNetwork(activation=activation)

    with open(args.network, "rb") as f:
        network.read_parameters(f)

    buffer_list = oak.read_battle_data(args.data_path)

    max_games = min(args.games or len(buffer_list), len(buffer_list))

    for buffer, n_frames in buffer_list[:max_games]:
        encoded_frames = oak.EncodedBattleFrames.from_bytes(buffer, n_frames)
        encoded_frames_torch = oak.torch.EncodedBattleFrames(encoded_frames)
        o1 = oak.OutputBuffer(encoded_frames.size)
        python_output = oak.torch.OutputBuffer(o1)
        network.inference(encoded_frames_torch, python_output)
        python_output.policy_logit[torch.isneginf(python_output.policy_logit)] = 0.0

        o2 = oak.cpp_inference(
            oak.BattleFrames.from_bytes(buffer, n_frames),
            args.network,
            args.discrete,
            "0",
        )
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


def battle_frame_stats():
    fn_parser = parser.add_argument_group("")
    args = parser.parse_args()
    assert args.data_path, "Provide path to recursively search for .battle.data files"

    import oak

    files = oak.util.find_data_files(args.data_path, ext=".battle.data")
    print(f"Found {len(files)} data files")

    total_frames = 0
    total_battles = 0
    iteration_counts = {}
    total_var = 0

    for file in files:
        data = oak.read_battle_data(file)

        for buf, n in data:
            frames = oak.BattleFrames.from_bytes(buf, n)

            for i in range(n):
                it = frames.iterations[i].item()
                if it in iteration_counts:
                    iteration_counts[it] += 1
                else:
                    iteration_counts[it] = 1

            total_battles += 1
            total_frames += n

    print(f"Total battle frames: {total_frames}")
    print(f"Average battle length: {total_frames / total_battles}")
    print(f"Iteration counts:")
    for key in iteration_counts:
        print(f"{key}: {iteration_counts[key]}")


def build_trajectory_stats():
    fn_parser = parser.add_argument_group("")
    args = parser.parse_args()

    # using only the head gives most recent files
    files = oak.util.find_data_files(".", ext=".build.data")
    assert len(files) > 0, "No build files found in cwd"

    from random import sample

    build_trajectories, read = oak.read_build_trajectories(files, 1024, 1)
    print(build_trajectories.size, read)
    assert build_trajectories.size == read, f"Bad read from {file}."
    for i in range(min(10, build_trajectories.size)):
        index = sample(list(range(build_trajectories.size)), 1)[0]
        print(f"Sample {index}:")
        species_move = [
            oak.species_move_list[_]
            for _ in build_trajectories.action[index].reshape(-1)
        ]
        names = []
        for sm in species_move:
            s, m = sm
            names.append(oak.species_names[s] + " " + oak.move_names[m])
        selection_probs = [
            int(1000 * float(_)) / 10
            for _ in build_trajectories.policy[index].reshape(-1)
        ]
        actions = build_trajectories.action[index].reshape(-1)

        data = [
            f"{n}:{p}" for n, p, a in zip(names, selection_probs, actions) if a >= 0
        ]

        l = 31
        print(data)
        print("value", build_trajectories.value[index].item())
        print("score", build_trajectories.score[index].item())
        # continue
        print("actions", build_trajectories.action[index, :l])
        print("legal actions mask", build_trajectories.mask[index, :l, :20])
        print("log probs", np.log(build_trajectories.policy[index, :l]))
        print(
            "start/end",
            np.concatenate(
                [
                    build_trajectories.start[index, :l],
                    build_trajectories.end[index, :l],
                ],
                axis=0,
            ),
        )


def create_team():
    fn_parser = parser.add_argument_group("")
    fn_parser.add_argument("--teams", default=1, type=int)
    fn_parser.add_argument("--max-pokemon", default=1, type=int)
    fn_parser.add_argument("--temp", default=1, type=float)
    args = parser.parse_args()

    import oak

    def get_index(s: int, m: int):
        x = oak.species_move_table[s][m]
        assert x >= 0, f"Invalid species move table access: {s}, {m}"
        return x

    class Set:
        def __init__(self, species: int = 0):
            self.species = species
            self.moves = set()
            self.moves_remaining = set()
            if species > 0:
                for m, index in enumerate(oak.species_move_table[species]):
                    if index >= 0 and m > 0:
                        self.moves_remaining.add(m)

        def add_move(self, m):
            self.moves.add(m)
            self.moves_remaining.remove(m)
            if len(self.moves) >= 4:
                self.moves_remaining = set()

        def write_to_input(self, t):
            if self.species > 0:
                t[get_index(self.species, 0)] = 1.0
                for m in self.moves:
                    t[get_index(self.species, m)] = 1.0

        def write_to_mask(self, t):
            if self.species > 0:
                t[get_index(self.species, 0)] = 0.0
                for m in self.moves:
                    t[get_index(self.species, m)] = 0.0
                if len(self.moves_remaining):
                    for m in self.moves_remaining:
                        t[get_index(self.species, m)] = 1.0
                else:
                    # delete move choices
                    for m, index in enumerate(oak.species_move_table[self.species]):
                        if index >= 0:
                            t[index] = 0.0

        def print_(
            self,
        ):
            print(
                f"{oak.species_names[self.species]}: {[oak.move_names[m] for m in self.moves]} | {len([oak.move_names[m] for m in self.moves_remaining])}"
            )

    def update_team(team, s, m):
        if m == 0:
            for i, set_ in enumerate(team):
                if set_.species == 0:
                    team[i] = Set(s)
                    break
        else:
            for set_ in team:
                if set_.species == s:
                    set_.add_move(m)
                    break

    def print_team(team):
        for set_ in team:
            set_.print_()

    if args.network:
        print("--network was provided but --build-network has priority")

    network_path = args.build_network or args.network or None

    if network_path:
        print(f"Using '{network_path}' as network path.")
    else:
        print("No build network path provided, using random network.")

    import torch
    from oak.torch import BuildNetwork

    network = BuildNetwork()
    if network_path:
        with open(network_path, "rb") as params:
            network.read_parameters(params)

    def sample_masked_logits(logits, mask):
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        probs = torch.softmax(masked_logits, dim=-1)
        probs_p = torch.softmax(masked_logits * args.temp, dim=-1)
        sampled = torch.multinomial(probs_p, 1)
        return (
            sampled,
            int(probs[sampled].item() * 10000) / 100,
            int(probs_p[sampled].item() * 10000) / 100,
        )

    def initial_mask():
        mask = torch.zeros([len(oak.species_move_list)])
        for i, sm in enumerate(oak.species_move_list):
            s, m = sm
            if m == 0:
                mask[i] = 1
        return mask

    def clear_species(mask):
        for i, sm in enumerate(oak.species_move_list):
            s, m = sm
            if m == 0:
                mask[i] = 0

    for t in range(args.teams):

        team = [Set() for _ in range(args.max_pokemon)]

        encoded_team = torch.zeros([len(oak.species_move_list)])
        mask = initial_mask()

        steps = 10000

        while any((s.species == 0) or len(s.moves_remaining) for s in team):

            steps += -1
            if steps == 0:
                exit()

            for set_ in team:
                set_.write_to_input(encoded_team)
                set_.write_to_mask(mask)
            if all(s.species > 0 for s in team):
                clear_species(mask)

            logits, _ = network.forward(encoded_team)
            index, p, q = sample_masked_logits(logits * args.temp, mask)
            species, move = oak.species_move_list[index]
            update_team(team, species, move)

        print("\nTeam", t)
        print_team(team)


def main():
    args, _ = parser.parse_known_args()

    assert args.main

    program_data = [
        ("test_consistency", test_consistency),
        ("battle_frame_stats", battle_frame_stats),
        ("create_team", create_team),
    ]

    args.main = args.main.replace("-", "_")

    for key, func in program_data:
        if args.main == key:
            func()
            exit()

    print(
        f"Bad 'main' value, expected one of: {", ".join([key for key, func in program_data])}"
    )


if __name__ == "__main__":
    main()
