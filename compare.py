"""Comparison runner for CS5180 PS1 Problem 3(c)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pi import policy_iteration
from utils import get_model, infer_sizes, make_env, timed
from vi import first_optimal_policy_iteration, value_iteration


GAMMAS = [0.5, 0.9, 0.99, 0.999]
THETA = 1e-4


@dataclass(frozen=True)
class ComparisonRow:
    gamma: float
    vi_iterations: int
    vi_seconds: float
    vi_backups: int
    pi_iterations: int
    pi_seconds: float
    pi_backup_equivalent: str


def vi_backup_count(n_states: int, n_actions: int, iterations: int) -> int:
    """Count state-action successor sweeps for VI."""

    return iterations * n_states * n_actions * n_states


def pi_work_summary(n_states: int, n_actions: int, iterations: int) -> str:
    """Describe PI work in the terms requested by the assignment."""

    linear_solve_work = iterations * (n_states**3)
    improvement_work = iterations * n_states * n_actions * n_states
    return f"~{linear_solve_work} solve ops + {improvement_work} improvement backups"


def run_comparison() -> list[ComparisonRow]:
    env = make_env()
    P = get_model(env)
    n_states, n_actions = infer_sizes(P)

    rows: list[ComparisonRow] = []
    for gamma in GAMMAS:
        vi_timed = timed(lambda: value_iteration(P, gamma=gamma, theta=THETA))
        pi_timed = timed(lambda: policy_iteration(P, gamma=gamma))

        vi_result = vi_timed.value
        pi_result = pi_timed.value

        rows.append(
            ComparisonRow(
                gamma=gamma,
                vi_iterations=vi_result.iterations,
                vi_seconds=vi_timed.seconds,
                vi_backups=vi_backup_count(n_states, n_actions, vi_result.iterations),
                pi_iterations=pi_result.iterations,
                pi_seconds=pi_timed.seconds,
                pi_backup_equivalent=pi_work_summary(n_states, n_actions, pi_result.iterations),
            )
        )

    return rows


def plot_iterations(rows: list[ComparisonRow], output_path: Path) -> None:
    gammas = [row.gamma for row in rows]
    vi_iters = [row.vi_iterations for row in rows]
    pi_iters = [row.pi_iterations for row in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(gammas, vi_iters, marker="o", label="Value iteration")
    plt.plot(gammas, pi_iters, marker="o", label="Policy iteration")
    plt.xlabel("gamma")
    plt.ylabel("iteration count")
    plt.title("FrozenLake iteration count vs gamma")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def main() -> None:
    rows = run_comparison()
    output_dir = Path(".")

    print("| gamma | VI iterations | VI time | VI backups | PI iterations | PI time | PI work |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row.gamma} | {row.vi_iterations} | {row.vi_seconds:.6f} | "
            f"{row.vi_backups} | {row.pi_iterations} | {row.pi_seconds:.6f} | "
            f"{row.pi_backup_equivalent} |"
        )

    (output_dir / "results.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2),
        encoding="utf-8",
    )
    plot_iterations(rows, output_dir / "iteration_counts.png")

    env = make_env()
    P = get_model(env)
    vi_star = value_iteration(P, gamma=0.99, theta=THETA)
    k_star, gap = first_optimal_policy_iteration(P, 0.99, THETA, vi_star.policy, vi_star.values)
    print(f"k*: {k_star}")
    print(f"||V_k* - V*||_infinity: {gap}")


if __name__ == "__main__":
    main()
