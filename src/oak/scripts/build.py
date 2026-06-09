import os
import argparse
import random
import time
import datetime
import itertools

import oak

parser = argparse.ArgumentParser(
    description="Train an Oak build network.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

oak.common_args.add_common_args(parser)


def add_local_args(parser, prefix: str = "", rl: bool = False):
    if prefix:
        prefix = prefix + "-"
    prefix = "--" + prefix
    parser.add_argument(
        prefix + "avg-gamma",
        type=float,
        default=0.995,
        help="Rolling average gamma",
    )
    parser.add_argument(
        prefix + "trajectories-per-step",
        type=int,
        required=True,
        help="How many trajectories to read from a single oak.read_build_trajectories call",
    )
    parser.add_argument(
        prefix + "keep-prob",
        type=float,
        required=True,
        default=1.0,
        help="The concatenated trajectories are sampled from at this rate to produce a minibatch",
    )
    parser.add_argument(
        prefix + "value-weight",
        type=float,
        default=1.0,
        help="Value vs Score weighting.",
    )
    parser.add_argument(
        prefix + "policy-loss-weight",
        type=float,
        default=1.0,
        help="Coeff to. KLdiv in the loss",
    )
    parser.add_argument(
        prefix + "value-loss-weight",
        type=float,
        default=0.5,
        help="Coeff. to value head MSE in the loss",
    )
    parser.add_argument(
        prefix + "entropy-loss-weight",
        type=float,
        default=0.01,
        help="Coeff. to entropy in the loss",
    )
    parser.add_argument(
        prefix + "gamma",
        type=float,
        default=0.99,
        help="PPO",
    )
    parser.add_argument(
        prefix + "lam",
        type=float,
        default=0.90,
        help="PPO",
    )
    parser.add_argument(
        prefix + "clip-eps",
        type=float,
        default=0.10,
        help="PPO",
    )
    parser.add_argument(
        prefix + "policy-hidden-dim",
        type=int,
        default=oak.build_policy_hidden_dim,
        help="Policy head hidden dim",
    )
    parser.add_argument(
        prefix + "value-hidden-dim",
        type=int,
        default=oak.build_value_hidden_dim,
        help="Value head hidden dim",
    )


add_local_args(parser)


def main():

    args = parser.parse_args()

    assert (
        not args.in_place or args.network_path
    ), "--network-path must be provided when --in-place is used."

    import torch
    import oak.torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    @torch.no_grad()
    def rolling_average(average_network, network, gamma):
        for avg_p, p in zip(average_network.parameters(), network.parameters()):
            avg_p.mul_(gamma).add_(p, alpha=1 - gamma)

    # Turn [b, T, 1] actions into [b, T, N] state
    def get_state(actions):
        b, T, _ = actions.shape
        state = torch.zeros(
            (b, T, len(oak.species_move_list) + 1), dtype=torch.float32
        )  # [b, T, N+1]
        state = state.scatter(2, actions + 1, 1.0)
        state = torch.cumsum(state, dim=1).clamp_max(1.0)
        state = state[:, :-1, 1:]  # [b, T, N]
        state = torch.concat([torch.zeros(state[:, :1].shape), state], dim=1)
        return state

    def apply_force_with_threshold(
        decision_outputs: torch.Tensor,
        force: torch.Tensor,
        threshold: float,
        threshold_center: torch.Tensor,
    ) -> torch.Tensor:
        can_decrease = decision_outputs - threshold_center > -threshold
        can_increase = decision_outputs - threshold_center < threshold
        force_negative = torch.clamp(force, max=0.0)
        force_positive = torch.clamp(force, min=0.0)
        clipped_force = can_decrease * force_negative + can_increase * force_positive
        return decision_outputs * clipped_force.detach()

    def process_targets(
        network: oak.torch.BuildNetwork,
        traj: oak.torch.BuildTrajectories,
    ):
        b, T, _ = traj.mask.shape
        # only do up to max traj length
        T = torch.max(traj.end).item()

        valid_actions = traj.action != -1  # [b, T, 1]
        valid_choices = traj.policy > 0  # [b, T, 1]
        valid = torch.logical_and(valid_actions, valid_choices).squeeze(-1)

        state = get_state(traj.action)
        valid_state = state[valid]
        logits, v = network.forward(valid_state)
        value = torch.sigmoid(v)

        # [v, 339]
        valid_mask = traj.mask[valid]
        mask_logits = torch.where(
            valid_mask >= 0,
            torch.gather(logits, 1, torch.clamp(valid_mask, min=0)),
            -torch.inf,
        )
        # print(mask_logits.shape)
        logits = mask_logits[:, 0]
        # print(logits.shape)
        mask_weights = torch.exp(mask_logits)
        mask_probs = torch.nn.functional.normalize(mask_weights, dim=1, p=1)

        # [v, 1]
        valid_logp = torch.log(mask_probs[:, 0]).unsqueeze(-1)
        valid_old_logp = torch.log(traj.policy[valid])
        # valid_ratio = torch.exp(valid_logp - valid_old_logp)

        # [b, T, 1]
        # ratio = torch.ones_like(traj.policy)
        # ratio[valid] = valid_ratio
        score_weight = 1 - args.value_weight
        r = args.value_weight * traj.value + score_weight * traj.score
        rewards = torch.zeros_like(traj.policy).scatter(
            1, traj.end.unsqueeze(-1) - 1, r.unsqueeze(-1)
        )
        value_full = torch.zeros_like(traj.policy)
        value_full[valid] = value
        next_value_full = torch.cat(
            [value_full[:, 1:], torch.zeros_like(value_full[:, -1:])], dim=1
        )
        deltas = rewards + args.gamma * next_value_full - value_full

        advantages = []
        advantage = torch.zeros((b, 1))
        for t in reversed(range(T)):
            # Only update advantage if step is valid
            advantage = (
                deltas[:, t, :] * valid[:, t].unsqueeze(-1)
                + args.gamma * args.lam * advantage
            )
            advantages.insert(0, advantage)
        gae = torch.stack(advantages, dim=1)  # [b, T, 1]
        returns = gae + value_full

        # print(gae[valid].shape)
        threshold_center = torch.zeros_like(logits)

        forced = apply_force_with_threshold(
            logits, gae[valid].squeeze(-1), 2, threshold_center
        )

        # surr1 = ratio * gae
        # surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * gae
        # surr = torch.min(surr1, surr2)
        logp = torch.ones_like(traj.policy)
        logp[valid] = valid_logp
        data = (
            returns,
            value_full,
            logp,
        )
        valid_data = (forced,) + tuple(x[valid] for x in data)
        # [print(x.shape) for x in valid_data]
        return valid_data

    def compute_loss_from_targets(
        surr, returns, values, logp_actions, total, remaining
    ) -> torch.Tensor:
        b = surr.shape[0]
        n = min(b, remaining)
        policy_loss = -(surr[:n]).mean()
        value_loss = (((returns - values)) ** 2)[:n].mean()
        entropy = (
            -(logp_actions * logp_actions.exp()).sum(dim=-1, keepdim=True)[:n].mean()
        )
        loss = (n / total) * (
            args.policy_loss_weight * policy_loss
            + args.value_loss_weight * value_loss
            - args.entropy_loss_weight * entropy
        )
        return loss

    if args.dir is None:
        now = datetime.datetime.now()
        args.dir = now.strftime("build-%Y-%m-%d-%H:%M:%S")

    os.makedirs(args.dir, exist_ok=False)
    oak.util.save_args(args, args.dir)

    network = oak.torch.BuildNetwork(args.policy_hidden_dim, args.value_hidden_dim)

    if args.network_path:
        with open(args.network_path, "rb") as f:
            network.read_parameters(f)

    average_network = oak.torch.BuildNetwork()

    with open(os.path.join(args.dir, "initial.build.net"), "wb") as f:
        network.write_parameters(f)
        print("Saved initial network in output directory.")

    with open(os.path.join(args.dir, "initial.build.net"), "rb") as f:
        average_network.read_parameters(f)

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    step_iterator = range(args.steps) if args.steps >= 0 else itertools.count()

    skipped_steps = 0

    for s in step_iterator:

        step = s - skipped_steps

        print(f"step: {step}")

        data_files, enough = oak.common_args.get_files(args, ".build.data")
        if not enough:
            skipped_steps += 1
            continue

        optimizer.zero_grad()
        b = 0
        # break batches up by file to limit memory use
        while b < args.batch_size:

            trajectories = oak.BuildTrajectories(args.trajectories_per_step)

            n_read = oak.read_build_trajectories(trajectories, data_files, args.threads)

            if n_read < trajectories.size:
                print(f"Error reading files, continuing")
                continue

            T = trajectories.end.max()
            # here is where we trunacte the episode length for 1v1, etc
            traj = oak.torch.BuildTrajectories(trajectories, n=T)

            surr, returns, values, logp = process_targets(network, traj)

            mask = torch.rand(surr.shape) < args.keep_prob
            surr, returns, values, logp = (
                surr[mask],
                returns[mask],
                values[mask],
                logp[mask],
            )
            if surr.numel() == 0:
                print("empty targets after sampling, probably a small buffer")
                continue

            loss = compute_loss_from_targets(
                surr,
                returns,
                values,
                logp,
                args.batch_size,
                args.batch_size - b,
            )
            loss.backward()

            b += surr.shape[0]

        optimizer.step()

        oak.common_args.save_and_decay(args, network, optimizer, step, ".build.net")

        rolling_average(average_network, network, args.avg_gamma)

        if ((step + 1) % args.checkpoint) == 0:
            with open(os.path.join(args.dir, f"{step + 1}.avg.build.net"), "wb") as f:
                average_network.write_parameters(f)


if __name__ == "__main__":
    main()
