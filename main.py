from __future__ import annotations
import sys
import os
import json

# Resolve paths so the script works regardless of CWD
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)  # env, viz, qlearning, sarsa, dynaq

import matplotlib
matplotlib.use("Agg")

import numpy as np
from env import SlipperyGridWorld
from viz import (
    evaluate, plot_policy, plot_value_heatmap, run_to_gif, greedy_policy_from_V,
)
from vi import VIAlgorithm
from qlearning import QLearning
from sarsa import SARSA
from dynaq import DynaQ

# ─── Shared hyperparameters ──────────────────────────────────────────────────

GAMMA = 0.99

VI_PARAMS = dict(
    max_iter=100,
    gamma=GAMMA,
    theta=1e-5,
)

SHARED_HYPERPARAMS = dict(
    alpha=0.2,
    gamma=GAMMA,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    episodes=8000,
    seed=9876,
)

# ─── Environment base config ─────────────────────────────────────────────────

BASE_ENV = dict(
    rows=5,
    cols=7,
    start=(0, 0),
    goal=(4, 6),
    slip_prob=0.2,
    step_reward=-1,
    goal_reward=10,
    max_steps=50,
    seed=9876,
)

# ─── Environment configurations ──────────────────────────────────────────────
# Each entry extends BASE_ENV with additional kwargs.

CONFIGS: dict[str, dict] = {
    "config_0_base": {},

    "config_1_obstacles": dict(
        obstacles=[(1, 2), (2, 4), (3, 1)],
        obstacle_penalty=-10.0,
        obstacle_move_every=1,
    ),

    "config_1b_corner_obstacles": dict(
        corner_size=2,
        corner_obstacle_count=2,
        obstacle_penalty=-10.0,
        obstacle_move_every=1,
    ),

    "config_2_moving_goal": dict(
        moving_goal=True,
        goal_move_every=5,
    ),

    "config_3_multi_goal": dict(
        goals=[(0, 6), (2, 3)],
    ),

    "config_4_all": dict(
        corner_size=2,
        corner_obstacle_count=2,
        obstacle_penalty=-10.0,
        obstacle_move_every=2,
        moving_goal=True,
        goal_move_every=8,
        goals=[(0, 6)],
    ),
}

# ─── Algorithm factories ─────────────────────────────────────────────────────
# Each factory creates a fresh instance so epsilon always starts at epsilon_start.

ALGORITHMS: dict[str, type] = {
    "vi": VIAlgorithm,
    "qlearning": QLearning,
    "sarsa": SARSA,
    "dynaq": DynaQ,
}

ALGO_EXTRA_KWARGS: dict[str, dict] = {
    "vi": {},
    "qlearning": {},
    "sarsa": {},
    "dynaq": {"n_planning_steps_start": 0, "n_planning_steps_max": 10},
}

# ─── Results root ────────────────────────────────────────────────────────────

RESULTS_ROOT = os.path.join(_SCRIPTS_DIR, "results")


# ─── Core run function ───────────────────────────────────────────────────────

def run_one(algo_name: str, config_name: str, env_extra: dict) -> dict:
    env = SlipperyGridWorld(**{**BASE_ENV, **env_extra})

    extra = ALGO_EXTRA_KWARGS[algo_name]
    params = VI_PARAMS if algo_name == "vi" else {**SHARED_HYPERPARAMS, **extra}
    algo = ALGORITHMS[algo_name](**params)

    print(f"    [{algo_name}] training...", end=" ", flush=True)
    Q = None
    V = None
    if algo_name == "vi":
        V = algo.run(env)
    else:
        Q = algo.train(env)
        V = np.max(Q, axis=1)
    print("done.", flush=True)
    pi = greedy_policy_from_V(V, env, GAMMA)

    out_dir = os.path.join(RESULTS_ROOT, algo_name, config_name)
    os.makedirs(out_dir, exist_ok=True)

    env.reset()
    plot_policy(
        env, pi,
        filename=os.path.join(out_dir, "policy.png"),
        title=f"{algo_name} | {config_name}",
        show=False,
    )
    plot_value_heatmap(
        env, V,
        filename=os.path.join(out_dir, "value.png"),
        title=f"{algo_name} | {config_name}",
        show=False,
    )

    env.reset()
    run_to_gif(
        env, Q=Q,
        policy=pi if algo_name == "vi" else None,
        gif_path=os.path.join(out_dir, "episode.gif"),
        frames_dir=os.path.join(out_dir, "frames"),
        fps=6,
    )

    metrics = evaluate(
        env, Q=Q,
        policy=pi if algo_name == "vi" else None,
        n_episodes=50,
        seed=9876
    )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    summary: dict[str, dict] = {}

    for config_name, env_extra in CONFIGS.items():
        print(f"\n[{config_name}]")
        for algo_name in ALGORITHMS:
            metrics = run_one(algo_name, config_name, env_extra)
            key = f"{algo_name}/{config_name}"
            summary[key] = metrics
            print(f"      {metrics}")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    summary_path = os.path.join(RESULTS_ROOT, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPodsumowanie zapisano do: {summary_path}")


if __name__ == "__main__":
    main()
