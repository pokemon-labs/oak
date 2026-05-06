import os
import argparse
import datetime
import subprocess
import signal
import threading
import sys
from typing import Dict


parser = argparse.ArgumentParser(
    description="Reinforcement learning using a generate process and battle.py (and optionally build.py)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


generate_parser = parser.add_argument_group("Data generation training arguments")
battle_parser = parser.add_argument_group("Battle network training arguments")
build_parser = parser.add_argument_group("Build network training arguments")


generate_parser.add_argument(
    "--generate-threads",
    type=int,
    default=(((os.cpu_count() or 2) - 1) or 1),
    help="Number of threads for self-play data generation",
)
generate_parser.add_argument(
    "--network-path",
    type=str,
    default="",
    help="Initial network ('mc' = monte-carlo, no battle net training/"
    " = random initial)",
)
generate_parser.add_argument(
    "--build-network-path",
    type=str,
    default="",
    help="Initial build network ('' = random initial)",
)
generate_parser.add_argument(
    "--budget",
    type=int,
    help="Number of iterations for training frames",
    required=True,
)
generate_parser.add_argument(
    "--fast-budget",
    type=int,
    help="Number of iterations for non-training frames",
    required=True,
)
generate_parser.add_argument(
    "--t1-budget",
    type=int,
    default=None,
    help="Number of iterations for turn 1 search, used when skipping battles for build data generation",
)
generate_parser.add_argument(
    "--fast-search-prob",
    type=float,
    help="Probability that fast-budget is used instead of budget",
    required=True,
)
generate_parser.add_argument(
    "--bandit",
    type=str,
    help="Bandit algorithm and parameters (exp3/pexp3/p2exp3/ucb/pucb)-(exp3 'gamma'/ucb 'c')",
    required=True,
)
generate_parser.add_argument(
    "--policy-mode",
    type=str,
    help="Mode for move selection (n/e/x/m)",
    required=True,
)
generate_parser.add_argument(
    "--fast-policy-mode",
    type=str,
    help="Mode for move selection (n/e/x/m)",
    default="x",
)
generate_parser.add_argument(
    "--policy-temp",
    type=float,
    default=1,
    help="P-norm exponent applied just before clipping and sampling",
)
generate_parser.add_argument(
    "--fast-policy-temp",
    type=float,
    help="See policy-temp",
    default=1,
)
generate_parser.add_argument(
    "--policy-min",
    type=float,
    default=0,
    help="Min prob of action before clamping to 0",
)
generate_parser.add_argument("--teams", type=str, default="", help="Path to teams file")
generate_parser.add_argument(
    "--max-pokemon",
    type=int,
    default=6,
    # help="Min prob of action before clamping to 0",
)
generate_parser.add_argument(
    "--pokemon-delete-prob",
    type=float,
    default=0,
)
generate_parser.add_argument(
    "--team-modify-prob",
    type=float,
    default=0,
)
generate_parser.add_argument(
    "--move-delete-prob",
    type=float,
    default=0,
)
generate_parser.add_argument(
    "--battle-skip-prob",
    type=float,
    default=0,
)

# get args from battle.py/build.py
import oak.common_args
import oak.battle
import oak.build

oak.common_args.add_common_args(battle_parser, "", True)
oak.battle.add_local_args(battle_parser, "", True)

oak.common_args.add_common_args(build_parser, "build", True)
oak.build.add_local_args(build_parser, "build", True)

parser.add_argument(
    "--generate-path",
    type=str,
    default="release/generate",
    help="Path to generate binary",
)
parser.add_argument(
    "--battle-path",
    type=str,
    default="py/battle.py",
    help="Path to battle.py script",
)
parser.add_argument(
    "--build-path",
    type=str,
    default="py/build.py",
    help="Path to build.py script",
)


def main():

    args = parser.parse_args()

    use_battle: bool = args.network_path != "mc"
    if not use_battle:
        print("Skipping battle.py")
    use_build: bool = (args.team_modify_prob > 0) and (
        (args.pokemon_delete_prob > 0) or (args.move_delete_prob > 0)
    )
    if not use_build:
        print("Skipping build.py")

    import oak
    import torch
    import oak.torch

    # arg set
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.t1_budget is None:
        args.t1_budget = args.budget

    # paths set
    now = datetime.datetime.now()
    working_dir = now.strftime("rl-%Y-%m-%d-%H:%M:%S")
    os.makedirs(working_dir, exist_ok=False)
    oak.util.save_args(args, working_dir)
    network_path = os.path.join(working_dir, "current.battle.net")
    build_network_path = os.path.join(working_dir, "current.build.net")
    data_dir = os.path.join(working_dir, "generate")
    nets_dir = os.path.join(working_dir, "nets")
    build_nets_dir = os.path.join(working_dir, "build_nets")

    # load/save initial dirs
    if use_battle:
        with open(network_path, "wb") as f:
            network = oak.torch.BattleNetwork(
                args.pokemon_hidden_dim,
                args.active_hidden_dim,
                args.pokemon_out_dim,
                args.active_out_dim,
                args.hidden_dim,
                args.value_hidden_dim,
                args.policy_hidden_dim,
                activation=(
                    oak.torch.Activation.clamp
                    if args.discrete
                    else oak.torch.Activation.relu
                ),
            )
            if args.network_path != "":
                print(f"Reading initial network from {args.network_path}")
                with open(args.network_path, "rb") as g:
                    network.read_parameters(g)
            network.write_parameters(f)

    if use_build:
        with open(build_network_path, "wb") as f:
            build_network = oak.torch.BuildNetwork(
                args.build_policy_hidden_dim, args.build_value_hidden_dim
            )
            if args.build_network_path != "":
                print(f"Reading initial build network from {args.build_network_path}")
                with open(args.build_network_path, "rb") as g:
                    build_network.read_parameters(g)
            build_network.write_parameters(f)

    generate_cmd = [
        "generate",
        f"--budget={args.budget}",
        f"--fast-budget={args.fast_budget}",
        f"--t1-budget={args.t1_budget}",
        f"--bandit={args.bandit}",
        f"--eval={network_path if use_battle else "mc"}",
        f"--policy-mode={args.policy_mode}",
        f"--fast-policy-mode={args.fast_policy_mode}",
        f"--policy-temp={args.policy_temp}",
        f"--fast-policy-temp={args.fast_policy_temp}",
        f"--policy-min={args.policy_min}",
        f"--dir={data_dir}",
        f"--threads={args.generate_threads}",
        "--buffer-size=1",
        "--max-battles=0",
        f"--max-battle-length={args.max_battle_length}",
        f"--fast-search-prob={args.fast_search_prob}",
        f"--teams={args.teams}",
        f"--max-pokemon={args.max_pokemon}",
        f"--team-modify-prob={args.team_modify_prob}",
        f"--pokemon-delete-prob={args.pokemon_delete_prob}",
        f"--move-delete-prob={args.move_delete_prob}",
        f"--battle-skip-prob={args.battle_skip_prob}",
        f"--build-network-path={build_network_path}",
    ]

    def get_common_cmd(args, prefix):
        p = prefix or ""
        return [
            f"--network-path={network_path if prefix == "" else build_network_path}",
            f"--data-dir={data_dir}",
            "--in-place",
            f"--device={getattr(args, f'{p}device')}",
            f"--threads={getattr(args, f'{p}threads')}",
            f"--batch-size={getattr(args, f'{p}batch_size')}",
            f"--lr={getattr(args, f'{p}lr')}",
            f"--lr-decay={getattr(args, f'{p}lr_decay')}",
            f"--lr-decay-start={getattr(args, f'{p}lr_decay_start')}",
            f"--lr-decay-interval={getattr(args, f'{p}lr_decay_interval')}",
            f"--data-window={getattr(args, f'{p}data_window')}",
            f"--min-files={getattr(args, f'{p}min_files')}",
            f"--sleep={getattr(args, f'{p}sleep')}",
            f"--checkpoint={getattr(args, f'{p}checkpoint')}",
            f"--delete-window={getattr(args, f'{p}delete_window')}",
        ]

    battle_cmd = (
        [
            sys.executable,
            "-u",
            "-m",
            "oak.battle",
            f"--dir={nets_dir}",
        ]
        + get_common_cmd(args, "")
        + [
            f"--max-battle-length={args.max_battle_length}",
            f"--min-iterations={args.fast_budget + 1}",
            f"--value-nash-weight={args.value_nash_weight}",
            f"--value-empirical-weight={args.value_empirical_weight}",
            f"--value-score-weight={args.value_score_weight}",
            f"--p-nash-weight={args.p_nash_weight}",
            f"--policy-loss-weight={args.policy_loss_weight}",
            # Don't need to pass network hyperparams since those are overwritten by the read
        ]
    )

    build_cmd = (
        [
            sys.executable,
            "-u",
            "-m",
            "oak.build",
            f"--dir={build_nets_dir}",
        ]
        + get_common_cmd(args, "build_")
        + [
            f"--trajectories-per-step={args.build_trajectories_per_step}",
            f"--keep-prob={args.build_keep_prob}",
            f"--value-weight={args.build_value_weight}",
            f"--entropy-loss-weight={args.build_entropy_loss_weight}",
            f"--value-loss-weight={args.build_value_loss_weight}",
            f"--entropy-loss-weight={args.build_entropy_loss_weight}",
            f"--gamma={args.build_gamma}",
            f"--lam={args.build_lam}",
            f"--clip-eps={args.build_clip_eps}",
        ]
    )

    if args.discrete:
        generate_cmd.append("--use-discrete")
        battle_cmd.append("--discrete")

    generate_proc = subprocess.Popen(
        generate_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=1,
    )

    battle_proc = None
    if use_battle:
        battle_proc = subprocess.Popen(
            battle_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=1,
        )

    build_proc = None
    if use_build:
        build_proc = subprocess.Popen(
            build_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=1,
        )

    procs = [(generate_proc, "GEN"), (battle_proc, "BATTLE"), (build_proc, "TEAM")]

    def stream(prefix, pipe):
        for line in iter(pipe.readline, b""):
            print(f"[{prefix}] {line.decode()}", end="")
        pipe.close()

    for p, s in procs:
        if p is not None:
            threading.Thread(target=stream, args=(s, p.stdout), daemon=True).start()

    try:
        generate_proc.wait()
    except KeyboardInterrupt:
        print("\nCtrl-C received, killing children...")

        for p in (generate_proc, battle_proc, build_proc):
            try:
                if p is not None:
                    os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    print("Finished.")


if __name__ == "__main__":
    main()
