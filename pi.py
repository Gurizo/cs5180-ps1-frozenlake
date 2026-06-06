"""Policy iteration for CS5180 PS1 Problem 3(b)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from utils import FrozenLakeP, format_policy_grid, format_value_grid, get_model, infer_sizes, make_env
from vi import q_value


@dataclass(frozen=True)
class PolicyIterationResult:
    values: np.ndarray
    policy: np.ndarray
    iterations: int


def build_policy_system(P: FrozenLakeP, policy: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Build A and b for A V = b under a fixed deterministic policy."""

    n_states, _ = infer_sizes(P)
    A = np.eye(n_states, dtype=float)
    b = np.zeros(n_states, dtype=float)

    for state in range(n_states):
        action = int(policy[state])
        for prob, next_state, reward, _terminated in P[state][action]:
            A[state, next_state] -= gamma * prob
            b[state] += prob * reward

    return A, b


def evaluate_policy(P: FrozenLakeP, policy: np.ndarray, gamma: float) -> np.ndarray:
    """Evaluate a fixed policy with numpy.linalg.solve."""

    A, b = build_policy_system(P, policy, gamma)
    return np.linalg.solve(A, b)


def improve_policy(
    P: FrozenLakeP,
    values: np.ndarray,
    gamma: float,
    old_policy: np.ndarray | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """Return a greedy policy, preserving old actions when they are tied."""

    n_states, n_actions = infer_sizes(P)
    new_policy = np.zeros(n_states, dtype=int)
    for state in range(n_states):
        q_values = np.array([q_value(P, state, action, values, gamma) for action in range(n_actions)])
        best_value = float(np.max(q_values))
        best_actions = np.flatnonzero(np.isclose(q_values, best_value, atol=tol))
        if old_policy is not None and int(old_policy[state]) in best_actions:
            new_policy[state] = int(old_policy[state])
        else:
            new_policy[state] = int(best_actions[0])
    return new_policy


def policy_iteration(P: FrozenLakeP, gamma: float) -> PolicyIterationResult:
    """Run policy iteration until the policy stops changing."""

    n_states, _ = infer_sizes(P)
    policy = np.zeros(n_states, dtype=int)
    iterations = 0

    while True:
        values = evaluate_policy(P, policy, gamma)
        new_policy = improve_policy(P, values, gamma, old_policy=policy)
        iterations += 1

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    values = evaluate_policy(P, policy, gamma)
    return PolicyIterationResult(values=values, policy=policy, iterations=iterations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    env = make_env()
    P = get_model(env)
    result = policy_iteration(P, gamma=args.gamma)

    print(f"PI iterations: {result.iterations}")
    print("V* grid:")
    print(format_value_grid(result.values))
    print("Policy grid:")
    print(format_policy_grid(result.policy))


if __name__ == "__main__":
    main()
