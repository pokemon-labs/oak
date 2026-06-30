import argparse
import time
import os
import fcntl
import random
from typing import List

import oak

# Extension for pickled optimizer params, current step, RNG (doesn't matter)
TRAIN_STATE_EXT = ".train_state"


def add_common_args(
    parser: argparse.ArgumentParser, prefix: str = "", rl: bool = False
):

    if prefix:
        prefix = prefix + "-"
    prefix = "--" + prefix

    if not rl:
        parser.add_argument(
            prefix + "network-path", type=str, help="Path for initial network weights"
        )
        parser.add_argument(
            prefix + "dir",
            type=str,
            help="Output directory. Timestamp if not provided",
        )
        parser.add_argument(
            prefix + "data-dir",
            default=".",
            help="Read directory for data files. All subdirectories are scanned",
        )
        parser.add_argument(
            prefix + "in-place",
            action="store_true",
            help="The parameters saved in --network-path will be updated after every step",
        )
        parser.add_argument(
            prefix + "steps",
            type=int,
            default=-1,
            help="Total training steps. A negative value is treated as infinity",
        )

    # Program
    parser.add_argument(
        prefix + "device",
        type=str,
        help='"cpu" or "cuda". Defaults to CUDA if available.',
    )
    parser.add_argument(
        prefix + "threads",
        type=int,
        default=1,
        help="Number of threads for data loading/training",
    )

    # Basic Training
    parser.add_argument(
        prefix + "batch-size", required=True, type=int, help="Batch size"
    )
    parser.add_argument(prefix + "lr", type=float, required=True, help="Learning rate")
    parser.add_argument(
        prefix + "lr-decay",
        type=float,
        default=1.0,
        help="Applied each step after decay begins",
    )
    parser.add_argument(
        prefix + "lr-decay-start",
        type=int,
        default=0,
        help="The first step to begin applying lr decay",
    )
    parser.add_argument(
        prefix + "lr-decay-interval",
        type=int,
        default=1,
        help="Interval at which to apply decay",
    )

    # Train/Generate interop

    parser.add_argument(
        prefix + "data-window",
        type=int,
        default=0,
        help="Only use the n-most recent files for freshness (0 means use all files)",
    )
    parser.add_argument(
        prefix + "min-files",
        type=int,
        default=1,
        help="Minimum number of data files before learning begins",
    )
    parser.add_argument(
        prefix + "sleep",
        type=float,
        default=0,
        help="Number of seconds to sleep after parameter update",
    )

    # QoL
    parser.add_argument(
        prefix + "checkpoint", type=int, default=50, help="Checkpoint interval (steps)"
    )
    parser.add_argument(
        prefix + "delete-window",
        type=int,
        default=0,
        help="Anything outside the most recent N files is deleted",
    )
    parser.add_argument(prefix + "seed", type=int, help="Random seed for determinism")


def get_files(args: argparse.ArgumentParser, ext: str) -> [List[str], bool]:
    data_files = oak.util.find_data_files(args.data_dir, ext)

    if len(data_files) < args.min_files:
        print("Minimum files not reached. Sleeping")
        time.sleep(5)
        return data_files, False

    if args.delete_window > 0:
        to_delete = data_files[args.delete_window :]
        for file in to_delete:
            os.remove(file)

    if args.data_window > 0:
        data_files = data_files[: args.data_window]

    return data_files, True


def train_state_path(path: str) -> str:
    return path + TRAIN_STATE_EXT


def save_train_state(path: str, opt, step: int):
    import torch

    state = {
        "step": step,
        "optimizer": opt.state_dict(),
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_train_state(path: str, opt) -> int:
    step = 0
    if not os.path.exists(path):
        print(
            f"No train state found at {path}; starting optimizer/step fresh "
            "(this checkpoint predates resumable training state)."
        )
    else:
        import torch

        state = torch.load(path, map_location="cpu")
        opt.load_state_dict(state["optimizer"])
        if "random_state" in state:
            random.setstate(state["random_state"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

        step = state.get("step", 0)
        print(f"Loaded train state from {path}, resuming at step {step}.")
    return step


def save_and_decay(args, network, opt, step: int, ext: str):
    if step >= args.lr_decay_start:
        if (step % args.lr_decay_interval) == 0:
            for group in opt.param_groups:
                group["lr"] *= args.lr_decay

    if args.in_place:
        ckpt_path = args.network_path
        tmp_path = ckpt_path + ".tmp"
        with open(tmp_path, "wb") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            network.write_parameters(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        os.replace(tmp_path, ckpt_path)
        save_train_state(train_state_path(ckpt_path), opt, step + 1)

    if (step + 1) % args.checkpoint == 0:
        ckpt_path = os.path.join(args.dir, f"{step + 1}{ext}")
        with open(ckpt_path, "wb") as f:
            network.write_parameters(f)
        save_train_state(train_state_path(ckpt_path), opt, step + 1)
        print(f"Checkpoint saved at step {step + 1}: {ckpt_path}")

    time.sleep(args.sleep)
