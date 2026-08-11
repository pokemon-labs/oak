"""
Illustrates binding oak.torch's battle nets directly to a live
pyoaksearch.Network, instead of the legacy read/write_battle_parameters()
disk round trip.

NOT exercised against a build in this change -- pyoaksearch/pyoaktrain are
compiled pybind11 extensions and this script wasn't run here. Treat it as a
worked example of the intended call sequence / a starting point for wiring
into oak/scripts/rl.py, not as tested code.

There is no BattleNetwork class. oak.torch.BattleNets is a plain
(pokemon_net, active_net, moves_net, main_net) namedtuple; build_battle_nets()
constructs one, bind_battle_nets_live() binds it onto a live C++ Network's
storage, and battle_forward() is the free-function forward pass (no `self`,
no class) that used to be BattleNetwork.inference().

Old flow (per scripts/rl.py, before this change):
    network = oak.torch.BattleNetwork(...)             # fresh nn.Linear weights
    network.read_parameters(open(path, "rb"))           # struct.unpack into them
    optimizer = torch.optim.Adam(network.parameters())
    ...train via network.inference(...)...
    network.write_parameters(open(out_path, "wb"))       # struct.pack back out
    # separately: a Search::Network used for MCTS reads that same file back
    # in via its own read_parameters(path) to pick up the new weights.

New flow (this script): one Search::Network is both the training target and
the search/inference network -- no file round trip between train step and
next self-play batch, and no class wrapping the four sub-nets.
"""

import pyoaksearch
import oak.torch as ot
import torch


def build_live_battle_nets(
    seed: int,
    activation: ot.Activation = ot.Activation.relu,
    pokemon_hidden_dim=None,
    active_hidden_dim=None,
    moves_hidden_dim=None,
    hidden_dim=None,
    value_hidden_dim=None,
    policy_hidden_dim=None,
):
    """Create a pyoaksearch.Network at a given architecture + an
    oak.torch.BattleNets bound to its live storage. Returns both -- keep
    `cpp_network` around for search/MCTS, and `nets` for training. They are
    the same weights in memory.
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

    # IMPORTANT: resize() must happen BEFORE bind_battle_nets_live(). Any
    # resize() call AFTER binding silently invalidates every Parameter
    # bound below (reallocates the Eigen matrices the numpy/torch views
    # alias) -- see the resize()/named_parameters() lifetime note in
    # pyoaksearch.cc.
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

    nets = ot.build_battle_nets(
        phd=pokemon_hidden_dim,
        ahd=active_hidden_dim,
        mhd=moves_hidden_dim,
        pod=ot_train.pokemon_out_dim,
        aod=ot_train.active_out_dim,
        mod=ot_train.moves_out_dim,
        hd=hidden_dim,
        vhd=value_hidden_dim,
        pohd=policy_hidden_dim,
        activation=activation,
    )
    ot.bind_battle_nets_live(nets, cpp_network)

    return cpp_network, nets


def load_live_battle_nets(path: str, activation: ot.Activation = ot.Activation.relu):
    """Same as build_live_battle_nets, but loads weights from an existing
    checkpoint file (produced by either the old write_battle_parameters()
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

    nets = ot.build_battle_nets(
        phd=pokemon_w.shape[0],
        ahd=active_w.shape[0],
        mhd=moves_w.shape[0],
        pod=pokemon_out_w.shape[0],
        aod=active_out_w.shape[0],
        mod=moves_out_w.shape[0],
        hd=main_w.shape[0],
        vhd=value_w.shape[0],
        pohd=policy_w.shape[0],
        activation=activation,
    )
    ot.bind_battle_nets_live(nets, cpp_network)
    return cpp_network, nets


def example_train_step(cpp_network, nets, frames, output_buffer):
    """One training step against live weights -- no read/write_parameters
    call anywhere in this loop. `cpp_network` is immediately usable for
    search/MCTS again the instant optimizer.step() returns.
    """
    optimizer = torch.optim.Adam(ot.battle_parameters(nets), lr=1e-4)

    ot.battle_forward(nets, frames, output_buffer, use_policy=True)
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
    ot.clamp_battle_parameters(nets)  # in-place .data.clamp_, safe under binding

    # cpp_network's Eigen buffers were just mutated by optimizer.step().
    # No write_parameters()/read_parameters() call needed before running
    # search against it again:
    #   pyoaksearch.run(battle, durations, budget, bandit, heap,
    #                    eval=cpp_network, ...)

    return loss.item()
