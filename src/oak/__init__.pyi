from typing import Any
import numpy as np
from numpy.typing import NDArray

BATTLE_SIZE: int
DURATIONS_SIZE: int

class Stats:
    hp: int
    atk: int
    def_: int
    spe: int
    spc: int

class MoveSlot:
    id: int
    pp: int

    def name(self) -> str: ...

class Boosts:
    atk: int
    def_: int
    spe: int
    spc: int
    acc: int
    eva: int
    raw: int

class Volatiles:
    bide: bool
    thrashing: bool
    multi_hit: bool
    flinch: bool
    charging: bool
    binding: bool
    invulnerable: bool
    confusion: bool
    mist: bool
    focus_energy: bool
    substitute: bool
    recharging: bool
    rage: bool
    leech_seed: bool
    toxic: bool
    light_screen: bool
    reflect: bool
    transform: bool

    confusion_left: int
    attacks: int
    disable_left: int

    state: int
    substitute_hp: int
    transform_species: int
    disable_move: int
    toxic_counter: int

    bits: int

class Pokemon:
    hp: int
    status: int
    species: int
    types: int
    level: int

    def stats(self) -> Stats: ...
    def move(self, index: int) -> MoveSlot: ...

    def percent(self) -> int: ...
    def species_name(self) -> str: ...
    def status_name(self) -> str: ...

class ActivePokemon:
    species: int
    types: int

    def stats(self) -> Stats: ...
    def boosts(self) -> Boosts: ...
    def volatiles(self) -> Volatiles: ...
    def move(self, index: int) -> MoveSlot: ...
    def species_name(self) -> str: ...

class Side:
    active: ActivePokemon

    order: list[int]

    last_selected_move: int
    last_used_move: int

    def pokemon(self, index: int) -> Pokemon: ...
    def slot(self, slot: int) -> Pokemon: ...
    def stored(self) -> Pokemon: ...

class Battle:
    turn: int
    last_damage: int
    rng: int

    def __init__(self, data: bytes = ...) -> None: ...

    def side(self, index: int) -> Side: ...
    def bytes(self) -> bytes: ...

class Duration:
    confusion: int
    disable: int
    attacking: int
    binding: int
    raw: int

    def sleep(self, slot: int) -> int: ...
    def set_sleep(self, slot: int, value: int) -> None: ...

class Durations:
    def __init__(self, data: bytes = ...) -> None: ...

    def get(self, side: int) -> Duration: ...
    def bytes(self) -> bytes: ...

class Heap:
    def empty(self) -> bool: ...
    def type(self) -> int: ...

class Agent:
    budget: str
    bandit: str
    eval: str
    matrix_ucb: bool
    discrete: bool
    table: bool

class Output:
    iterations: int
    empirical_value: float
    nash_value: float
    duration_ms: int

    visit_matrix: NDArray[np.uint64]
    value_matrix: NDArray[np.float64]

    p1_prior: NDArray[np.float64]
    p2_prior: NDArray[np.float64]

class BattleFrames:
    size: int

class EncodedBattleFrames:
    size: int

    def __init__(self, size: int) -> None: ...
    def clear(self) -> None: ...

class OutputBuffer:
    size: int

class SampleIndexer:
    def get(self, path: str) -> list[tuple[int, int]]: ...
    def prune(self, paths: list[str]) -> None: ...
    def size(self) -> int: ...

def solve_matrix(
    row_payoff: NDArray[np.float32],
    discretize_factor: int = ...
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    float,
]: ...

def read_battle_data(
    path: str
) -> list[tuple[bytes, int]]: ...

def cpp_inference(
    battle_frames: BattleFrames,
    network_path: str,
    discrete: bool = ...,
    budget: str = ...,
) -> OutputBuffer: ...

def parse_battle(
    battle_string: str,
    seed: int = ...
) -> tuple[Battle, Durations, int]: ...