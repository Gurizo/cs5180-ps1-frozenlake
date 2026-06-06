"""Value iteration for CS5180 PS1 Problem 3(a)/(d)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from utils import FrozenLakeP, format_policy_grid, format_value_grid, get_model, infer_sizes, make_env


@dataclass(frozen=True)
class ValueIterationResult:
    values: np.ndarray
    policy: np.ndarray
    iterations: int
    policy_emergence_iteration: int | None = None
    policy_emergence_gap: float | None = None


def q_value(P: FrozenLakeP, state: int, action: int, values: np.ndarray, gamma: float) -> float:
    """Compute one Bellman action value."""

    total = 0.0
    for prob, next_state, reward, _terminated in P[state][action]:
        total += prob * (reward + gamma * values[next_state])
    return total


def greedy_policy(P: FrozenLakeP, values: np.ndarray, gamma: float) -> np.ndarray:
    """Return argmax_a Q(s,a) for every state."""

    n_states, n_actions = infer_sizes(P)
    policy = np.zeros(n_states, dtype=int)
    for state in range(n_states):
        q_values = [q_value(P, state, action, values, gamma) for action in range(n_actions)]
        policy[state] = int(np.argmax(q_values))
    return policy


def value_iteration(P: FrozenLakeP, gamma: float, theta: float) -> ValueIterationResult:
    """Run value iteration and return V*, greedy policy, and iteration count."""

    n_states, _ = infer_sizes(P)
    values = np.zeros(n_states, dtype=float)
    threshold = theta * (1 - gamma) / gamma
    iterations = 0

    while True:
        new_values = np.zeros(n_states, dtype=float)
        for state in range(n_states):
            n_actions = len(P[state])
            q_values = [q_value(P, state, action, values, gamma) for action in range(n_actions)]
            new_values[state] = max(q_values)

        iterations += 1
        delta = float(np.max(np.abs(new_values - values)))
        values = new_values

        if delta < threshold:
            break

    policy = greedy_policy(P, values, gamma)
    return ValueIterationResult(values=values, policy=policy, iterations=iterations)


def first_optimal_policy_iteration(
    P: FrozenLakeP,
    gamma: float,
    theta: float,
    optimal_policy: np.ndarray,
    optimal_values: np.ndarray,
) -> tuple[int | None, float | None]:
    """Track the first VI iteration whose greedy policy matches optimal_policy."""

    n_states, _ = infer_sizes(P)
    values = np.zeros(n_states, dtype=float)

    if np.array_equal(greedy_policy(P, values, gamma), optimal_policy):
        gap = float(np.max(np.abs(values - optimal_values)))
        return 0, gap

    threshold = theta * (1 - gamma) / gamma
    iterations = 0

    while True:
        new_values = np.zeros(n_states, dtype=float)
        for state in range(n_states):
            n_actions = len(P[state])
            q_values = [q_value(P, state, action, values, gamma) for action in range(n_actions)]
            new_values[state] = max(q_values)

        iterations += 1
        delta = float(np.max(np.abs(new_values - values)))
        values = new_values

        if np.array_equal(greedy_policy(P, values, gamma), optimal_policy):
            gap = float(np.max(np.abs(values - optimal_values)))
            return iterations, gap

        if delta < threshold:
            return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-4)
    args = parser.parse_args()

    env = make_env()
    P = get_model(env)
    result = value_iteration(P, gamma=args.gamma, theta=args.theta)

    print(f"VI iterations: {result.iterations}")
    print("V* grid:")
    print(format_value_grid(result.values))
    print("Policy grid:")
    print(format_policy_grid(result.policy))


if __name__ == "__main__":
    main()
