"""Shared helpers for CS5180 PS1 FrozenLake planning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

import gymnasium as gym
import numpy as np


Action = int
State = int
Transition = tuple[float, int, float, bool]
FrozenLakeP = dict[int, dict[int, list[Transition]]]

ACTION_ARROWS = {
    0: "<",
    1: "v",
    2: ">",
    3: "^",
}


@dataclass(frozen=True)
class TimedResult:
    """Small wrapper for measured function calls."""

    value: object
    seconds: float


def make_env() -> gym.Env:
    """Create the exact environment required by the assignment."""

    return gym.make("FrozenLake-v1", is_slippery=True)


def get_model(env: gym.Env) -> FrozenLakeP:
    """Return Gymnasium's exposed transition model."""

    return env.unwrapped.P


def infer_sizes(P: FrozenLakeP) -> tuple[int, int]:
    """Infer number of states and actions from the FrozenLake transition dict."""

    n_states = len(P)
    n_actions = len(P[0])
    return n_states, n_actions


def timed(fn: Callable[[], object]) -> TimedResult:
    """Run a no-argument function and measure wall-clock time."""

    start = perf_counter()
    value = fn()
    return TimedResult(value=value, seconds=perf_counter() - start)


def as_grid(values: Iterable[float], width: int = 4) -> np.ndarray:
    """Reshape a flat state vector into the 4x4 FrozenLake layout."""

    return np.asarray(list(values), dtype=float).reshape((width, width))


def policy_to_grid(policy: Iterable[int], width: int = 4) -> list[list[str]]:
    """Render action indices as arrow strings in a grid."""

    actions = list(policy)
    return [
        [ACTION_ARROWS.get(actions[row * width + col], "?") for col in range(width)]
        for row in range(width)
    ]


def format_value_grid(values: Iterable[float], width: int = 4) -> str:
    """Create a compact printable value table."""

    grid = as_grid(values, width)
    return "\n".join(" ".join(f"{cell: .6f}" for cell in row) for row in grid)


def format_policy_grid(policy: Iterable[int], width: int = 4) -> str:
    """Create a compact printable arrow policy table."""

    return "\n".join(" ".join(row) for row in policy_to_grid(policy, width))


def terminal_states(P: FrozenLakeP) -> set[State]:
    """Find states where all actions immediately terminate by staying terminal."""

    terminals: set[State] = set()
    for state, actions in P.items():
        if all(all(done for _, _, _, done in transitions) for transitions in actions.values()):
            terminals.add(state)
    return terminals
