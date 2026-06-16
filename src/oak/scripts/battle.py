import os
import argparse
import random
import time
import datetime
import itertools
import sys
import numpy as np
import signal

import oak
import oak.train

parser = argparse.ArgumentParser(
    description="Train an Oak battle network.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

import oak.common_args

oak.common_args.add_common_args(parser)


def add_local_args(parser, prefix: str = "", rl: bool = False):
    if prefix:
        prefix = prefix + "-"
    prefix = "--" + prefix

    parser.add_argument(
        prefix + "max-battle-length",
        type=int,
        default=10000,
        help="Ignore games past this length (in updates not turns.)",
    )
    parser.add_argument(
        prefix + "discrete",
        action="store_true",
        help="Use clamping for activation [0, 1] and parameters [-2, 2], for discrete network usage",
    )

    if not rl:
        parser.add_argument(
            prefix + "min-iterations",
            type=int,
            default=1,
            help="Ignore samples with fewer than these iterations.",
        )
    parser.add_argument(
        prefix + "clamp-parameters",
        action="store_true",
        help="Clamp parameters [-2, 2] to support Quantized-style quantization",
    )
    parser.add_argument(
        prefix + "value-nash-weight",
        type=float,
        default=0.0,
        help="Weight for Nash value in value target",
    )
    parser.add_argument(
        prefix + "value-empirical-weight",
        type=float,
        default=0.0,
        help="Weight for empirical value in value target",
    )
    parser.add_argument(
        prefix + "value-score-weight",
        type=float,
        default=1.0,
        help="Weight for score in value target",
    )
    parser.add_argument(
        prefix + "p-nash-weight",
        type=float,
        default=0.0,
        help="Weight for Nash in policy target (empirical weight = 1 - this)",
    )
    parser.add_argument(
        prefix + "no-value-loss",
        action="store_true",
        dest="no_value_loss",
        help="Disable value loss computation",
    )
    parser.add_argument(
        prefix + "no-policy-loss",
        action="store_true",
        dest="no_policy_loss",
        help="Disable policy loss computation",
    )
    parser.add_argument(
        prefix + "policy-loss-weight",
        type=float,
        default=1.0,
        help="Weight for policy loss relative to value loss",
    )
    parser.add_argument(
        prefix + "no-apply-symmetries",
        action="store_true",
        help="Whether to skip permuting Bench/Sides",
    )

    # Network hyperparameters
    parser.add_argument(
        prefix + "pokemon-hidden-dim",
        type=int,
        default=oak.train.pokemon_hidden_dim,
        help="Pokemon encoding net hidden dim",
    )
    parser.add_argument(
        prefix + "active-hidden-dim",
        type=int,
        default=oak.train.active_hidden_dim,
        help="ActivePokemon encoding net hidden dim",
    )
    parser.add_argument(
        prefix + "pokemon-out-dim",
        type=int,
        default=oak.train.pokemon_out_dim,
        help="Pokemon encoding net output dim",
    )
    parser.add_argument(
        prefix + "active-out-dim",
        type=int,
        default=oak.train.active_out_dim,
        help="ActivePokemon encoding net output dim",
    )
    parser.add_argument(
        prefix + "hidden-dim",
        type=int,
        default=oak.train.hidden_dim,
        help="Main subnet hidden dim",
    )
    parser.add_argument(
        prefix + "value-hidden-dim",
        type=int,
        default=oak.train.value_hidden_dim,
        help="Value head hidden dim",
    )
    parser.add_argument(
        prefix + "policy-hidden-dim",
        type=int,
        default=oak.train.policy_hidden_dim,
        help="Policy head hidden dim",
    )
    parser.add_argument(
        prefix + "print-window",
        type=int,
        default=15,
        help="Number of samples to print for debug output",
    )


add_local_args(parser)


suspended = False


def handle_sigtstp(signum, frame):
    global suspended
    suspended = not suspended

    if suspended:
        print("Paused (Ctrl+Z to resume)")
    else:
        print("Resumed")


signal.signal(signal.SIGTSTP, handle_sigtstp)


def main():

    args = parser.parse_args()

    assert (
        not args.in_place or args.network_path
    ), "--network-path must be provided when --in-place is used."

    import torch
    import oak.torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    class Optimizer:
        def __init__(self, network: torch.nn.Module, lr):
            self.opt = torch.optim.Adam(network.parameters(), lr=lr)

        def step(self):
            self.opt.step()

        def zero_grad(self):
            self.opt.zero_grad()

    def masked_kl_div(logit, target):
        log_probs = torch.log_softmax(logit, dim=-1)
        log_probs = log_probs.masked_fill(torch.isneginf(log_probs), 0)
        log_target = torch.log(target)
        log_target = log_target.masked_fill(torch.isneginf(log_target), 0)
        kl = target * (log_target - log_probs)
        # kl = kl.masked_fill(torch.isneginf(logit), 0)
        return kl.sum(dim=1).mean(dim=0)

    def masked_cross_entropy(logit, target):
        log_probs = torch.log_softmax(logit, dim=-1)
        log_probs = log_probs.masked_fill(torch.isneginf(log_probs), 0)
        support_size = (target != 0).sum(dim=1).clamp_min(1)
        ce = -target * log_probs
        loss_per_sample = ce.sum(dim=1) / support_size
        return loss_per_sample.mean()

    def loss(
        input: oak.torch.EncodedBattleFrames,
        output: oak.torch.OutputBuffer,
        args,
        print_flag=False,
    ):
        size = min(input.size, output.size)

        # Value target
        value_nash_weight = args.value_nash_weight
        value_empirical_weight = args.value_empirical_weight
        value_score_weight = args.value_score_weight

        assert (
            value_nash_weight + value_empirical_weight + value_score_weight
        ) == 1, "value target weights don't sum to 1"
        value_target = (
            value_nash_weight * input.nash_value[:size]
            + value_empirical_weight * input.empirical_value[:size]
            + value_score_weight * input.score[:size]
        )

        # Policy target
        p_nash_weight = args.p_nash_weight
        value_empirical_weight_p = 1 - p_nash_weight
        assert (p_nash_weight + value_empirical_weight_p) == 1

        p1_policy_target = (
            value_empirical_weight_p * input.empirical_policies[:size, 0]
            + p_nash_weight * input.nash_policies[:size, 0]
        )
        p2_policy_target = (
            value_empirical_weight_p * input.empirical_policies[:size, 1]
            + p_nash_weight * input.nash_policies[:size, 1]
        )

        loss = 0.0

        if not args.no_value_loss:
            loss = loss + torch.nn.functional.mse_loss(
                output.value[:size], value_target
            )
        value_loss = loss.detach().clone()

        if not args.no_policy_loss:
            p1_policy_loss = masked_cross_entropy(
                output.policy_logit[:size, 0], p1_policy_target
            )
            p2_policy_loss = masked_cross_entropy(
                output.policy_logit[:size, 1], p2_policy_target
            )
            loss = loss + args.policy_loss_weight * (p1_policy_loss + p2_policy_loss)

        if print_flag:
            window = args.print_window
            print(
                torch.cat(
                    [
                        output.value[:window],
                        value_target[:window],
                        input.empirical_value[:window],
                        input.nash_value[:window],
                        input.score[:window],
                    ],
                    dim=1,
                )
            )
            if not args.no_policy_loss:
                print("P1 policy inference/target")
            if not args.no_value_loss:
                print(f"loss: v:{value_loss.mean()}")

        return loss

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.dir is None:
        now = datetime.datetime.now()
        args.dir = now.strftime("battle-%Y-%m-%d-%H:%M:%S")

    os.makedirs(args.dir, exist_ok=False)
    oak.util.save_args(args, args.dir)

    network = oak.torch.BattleNetwork(
        args.pokemon_hidden_dim,
        args.active_hidden_dim,
        args.pokemon_out_dim,
        args.active_out_dim,
        args.hidden_dim,
        args.value_hidden_dim,
        args.policy_hidden_dim,
        activation=(
            oak.torch.Activation.clamp if args.discrete else oak.torch.Activation.relu
        ),
    ).to(device)

    if args.network_path:
        with open(args.network_path, "rb") as f:
            network.read_parameters(f)

    with open(os.path.join(args.dir, "initial.battle.net"), "wb") as f:
        network.write_parameters(f)
        print("Saved initial network in output directory.")

    print(f"Initial network hash: {network.hash()}")

    encoded_frames = oak.EncodedBattleFrames(args.batch_size)
    encoded_frames_torch = oak.torch.EncodedBattleFrames(encoded_frames).to(device)

    output_buffer = oak.OutputBuffer(
        args.batch_size, args.pokemon_out_dim, args.active_out_dim
    )

    optimizer = Optimizer(network, args.lr)

    sample_indexer = oak.SampleIndexer()

    step_iterator = range(args.steps) if args.steps >= 0 else itertools.count()

    skipped_steps = 0

    for s in step_iterator:

        while suspended:
            time.sleep(0.1)

        step = s - skipped_steps

        data_files, enough = oak.common_args.get_files(args, ".battle.data")
        if not enough:
            skipped_steps += 1
            continue

        sample_indexer.prune(data_files)
        for file in data_files:
            sample_indexer.get(file)

        encoded_frames.clear()
        output_buffer.clear()
        samples_read = oak.sample(
            encoded_frames,
            sample_indexer,
            args.threads,
            args.max_battle_length,
            args.min_iterations,
        )

        if samples_read < args.batch_size:
            skipped_steps += 1
            print("Error during sampling, continuing...")
            continue

        if not args.no_apply_symmetries:
            encoded_frames_torch.permute_pokemon()
            encoded_frames_torch.permute_sides()

        output_buffer_torch = oak.torch.OutputBuffer(output_buffer).to(device)
        network.inference(
            encoded_frames_torch, output_buffer_torch, not args.no_policy_loss
        )

        optimizer.zero_grad()
        print_flag = (step % args.checkpoint) == 0
        loss_value = loss(
            encoded_frames_torch,
            output_buffer_torch,
            args,
            print_flag,
        )
        loss_value.backward()
        optimizer.step()

        if args.discrete or args.clamp_parameters:
            network.clamp_parameters()

        oak.common_args.save_and_decay(
            args, network, optimizer.opt, step, ".battle.net"
        )


if __name__ == "__main__":
    main()
