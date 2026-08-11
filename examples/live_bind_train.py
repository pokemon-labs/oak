"""
Illustrates binding oak.torch.BattleNetwork directly to a live
pyoaksearch.Network, instead of the legacy read_parameters()/
write_parameters() disk round trip.

NOT exercised against a build in this change -- pyoaksearch/pyoaktrain are
compiled pybind11 extensions and this script wasn't run here. Treat it as a
worked example of the intended call sequence / a starting point for wiring
into oak/scripts/rl.py, not as tested code.

Old flow (per scripts/rl.py):
    network = oak.torch.BattleNetwork(...)             # fresh nn.Linear weights
    network.read_parameters(open(path, "rb"))           # struct.unpack into them
    optimizer = torch.optim.Adam(network.parameters())
    ...train...
    network.write_parameters(open(out_path, "wb"))       # struct.pack back out
    # separately: a Search::Network used for MCTS reads that same file back
    # in via its own read_parameters(path) to pick up the new weights.

New flow (this script): one Search::Network is both the training target and
the search/inference network -- no file round trip between train step and
next self-play batch.
"""

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
    """Create a pyoaksearch.Network at a given architecture + a
    oak.torch.BattleNetwork bound to its live storage. Returns both --
    keep `cpp_network` around for search/MCTS, and `py_network` for
    training. They are the same weights in memory.
    """
    import oak.train as ot_train  # default dims live here (see pyoaktrain.cc)

    pokemon_hidden_dim = pokemon_hidden_dim or ot_train.pokemon_hidden_dim
    active_hidden_dim = active_hidden_dim or ot_train.active_hidden_dim
    moves_hidden_dim = moves_hidden_dim or ot_train.moves_hidden_dim
    hidden_dim = hidden_dim or ot_train.hidden_dim
    value_hidden_dim = value_hidden_dim or ot_train.value_hidden_dim
    policy_hidden_dim = policy_hidden_dim or ot_train.policy_hidden_dim

    # activation=1 here is the C++-side Network ctor arg (relu); see
    # pyoaksearch.cc Network's py::init<int>(activation=1). Keep it in sync
    # with the `activation` this function was called with if you use clamp.
    cpp_network = pyoaksearch.Network(activation=1)

    # IMPORTANT: resize() must happen BEFORE bind_live(). Any resize() call
    # AFTER binding silently invalidates every Parameter bound below
    # (reallocates the Eigen matrices the numpy/torch views alias) --
    # see the resize()/named_parameters() lifetime note in pyoaksearch.cc.
    cpp_network.resize(
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
    cpp_network.initialize(seed)

    py_network = ot.BattleNetwork(
        pokemon_hidden_dim,
        active_hidden_dim,
        moves_hidden_dim,
        ot_train.pokemon_out_dim,
        ot_train.active_out_dim,
        ot_train.moves_out_dim,
        hidden_dim,
        value_hidden_dim,
        policy_hidden_dim,
        activation=activation,
    )
    py_network.bind_live(cpp_network)

    return cpp_network, py_network


def load_live_network(path: str, activation: ot.Activation = ot.Activation.relu):
    """Same as build_live_network, but loads weights from an existing
    checkpoint file (produced by either the old Python write_parameters()
    or the new Network.write_parameters() -- on-disk format is identical).
    """
    cpp_network = pyoaksearch.Network(activation=1)
    cpp_network.read_parameters(path)  # resizes internally, then loads

    dims = cpp_network.named_parameters()
    pokemon_w, _ = dims["pokemon_net.fc0"]
    active_w, _ = dims["active_net.fc0"]
    moves_w, _ = dims["moves_net.fc0"]
    pokemon_out_w, _ = dims["pokemon_net.fc1"]
    active_out_w, _ = dims["active_net.fc1"]
    moves_out_w, _ = dims["moves_net.fc1"]
    main_w, _ = dims["main_net.fc0"]
    value_w, _ = dims["main_net.value_fc2"]
    policy_w, _ = dims["main_net.p1_policy_fc2"]

    py_network = ot.BattleNetwork(
        pokemon_hidden_dim=pokemon_w.shape[0],
        active_hidden_dim=active_w.shape[0],
        moves_hidden_dim=moves_w.shape[0],
        pokemon_out_dim=pokemon_out_w.shape[0],
        active_out_dim=active_out_w.shape[0],
        moves_out_dim=moves_out_w.shape[0],
        hidden_dim=main_w.shape[0],
        value_hidden_dim=value_w.shape[0],
        policy_hidden_dim=policy_w.shape[0],
        activation=activation,
    )
    py_network.bind_live(cpp_network)
    return cpp_network, py_network


def example_train_step(cpp_network, py_network, frames, output_buffer):
    """One training step against live weights -- no read/write_parameters
    call anywhere in this loop. `cpp_network` is immediately usable for
    search/MCTS again the instant optimizer.step() returns.
    """
    optimizer = torch.optim.Adam(py_network.parameters(), lr=1e-4)

    py_network.inference(frames, output_buffer, use_policy=True)
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
    py_network.clamp_parameters()  # in-place .data.clamp_, safe under binding

    # cpp_network's Eigen buffers were just mutated by optimizer.step().
    # No write_parameters()/read_parameters() call needed before running
    # search against it again:
    #   pyoaksearch.run(battle, durations, budget, bandit, heap,
    #                    eval=cpp_network, ...)

    return loss.item()
