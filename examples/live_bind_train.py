"""
Illustrates the battle-net training path end to end: a pyoaksearch.Network
is the only place the architecture/weights live. Python's job is wrapping
its parameter views as torch.nn.Parameter and running the differentiable
forward pass -- nothing else.

NOT exercised against a build in this change -- pyoaksearch/pyoaktrain are
compiled pybind11 extensions and this script wasn't run here. Treat it as a
worked example / a starting point for wiring into oak/scripts/rl.py.

There is no BattleNetwork class, no EmbeddingNet/MainNet clones, no
oak.train-derived dims anywhere below -- oak.torch.bind_live_params(network)
returns a flat {layer_name: (weight Parameter, bias Parameter)} dict whose
shapes come entirely from the layers themselves (weight.shape), and
oak.torch.battle_forward(params, ...) is the free-function forward pass.
"""

import random

import pyoaksearch
import oak.torch as ot
import torch


def build_live_network(
    seed: int,
    activation: ot.Activation = ot.Activation.relu,
    pokemon_hidden_dim=None,
    active_hidden_dim=None,
    moves_hidden_dim=None,
    hidden_dim=None,
    value_hidden_dim=None,
    policy_hidden_dim=None,
):
    """Create a pyoaksearch.Network at a given architecture, freshly
    initialized, plus its bound params dict. Returns both -- keep
    `network` around for search/MCTS, and `params` for training. They are
    the same weights in memory.
    """
    import oak.train as ot_train  # default dims live here (see pyoaktrain.cc)

    pokemon_hidden_dim = pokemon_hidden_dim or ot_train.pokemon_hidden_dim
    active_hidden_dim = active_hidden_dim or ot_train.active_hidden_dim
    moves_hidden_dim = moves_hidden_dim or ot_train.moves_hidden_dim
    hidden_dim = hidden_dim or ot_train.hidden_dim
    value_hidden_dim = value_hidden_dim or ot_train.value_hidden_dim
    policy_hidden_dim = policy_hidden_dim or ot_train.policy_hidden_dim

    network = pyoaksearch.Network(activation=activation)

    # IMPORTANT: resize() must happen BEFORE bind_live_params(). Any
    # resize() call AFTER binding silently invalidates every Parameter
    # bound below (reallocates the Eigen matrices the numpy/torch views
    # alias) -- see the resize()/named_parameters() lifetime note in
    # pyoaksearch.cc.
    network.resize(
        pokemon_hidden_dim,
        ot_train.pokemon_out_dim,
        active_hidden_dim,
        ot_train.active_out_dim,
        moves_hidden_dim,
        ot_train.moves_out_dim,
        hidden_dim,
        value_hidden_dim,
        policy_hidden_dim,
    )
    network.initialize(seed)

    params = ot.bind_live_params(network)
    return network, params


def load_live_network(path: str):
    """Load an existing checkpoint. No dims to pass -- read_parameters()
    resizes the Network internally, and bind_live_params() reads every
    layer's shape straight off the resulting weight views.
    """
    network = pyoaksearch.Network()  # activation is overwritten by the load
    network.read_parameters(path)
    activation = ot.Activation.clamp if network.is_clamped else ot.Activation.relu
    params = ot.bind_live_params(network)
    return network, params, activation


def example_train_step(network, params, activation, frames, output_buffer):
    """One training step against live weights -- no read/write_parameters
    call anywhere in this loop. `network` is immediately usable for
    search/MCTS again the instant optimizer.step() returns.
    """
    optimizer = torch.optim.Adam(ot.battle_parameters(params), lr=1e-4)

    ot.battle_forward(params, frames, output_buffer, activation=activation, use_policy=True)
    value_loss = torch.nn.functional.mse_loss(
        output_buffer.value[: frames.size], frames.empirical_value[: frames.size]
    )
    policy_loss = torch.nn.functional.cross_entropy(
        output_buffer.policy_logit[: frames.size].reshape(-1, output_buffer.policy_logit.shape[-1]),
        frames.empirical_policies[: frames.size].reshape(-1, frames.empirical_policies.shape[-1]),
    )
    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ot.clamp_battle_parameters(params)  # in-place .data.clamp_, safe under binding

    # network's Eigen buffers were just mutated by optimizer.step(). No
    # write_parameters()/read_parameters() call needed before running
    # search against it again:
    #   pyoaksearch.run(battle, durations, budget, bandit, heap,
    #                    eval=network, ...)

    return loss.item()


if __name__ == "__main__":
    network, params = build_live_network(seed=random.getrandbits(64))
    print(f"Built live network, hash={network.hash()}")
    print(f"Layers: {sorted(params.keys())}")
