from __future__ import annotations
import random
from typing import Optional, Tuple, Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
from env import ACTIONS, SlipperyGridWorld

ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def _base_grid_figure(env, title: str = ""):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)

    ax.set_xticks(np.arange(-0.5, env.cols, 1))
    ax.set_yticks(np.arange(-0.5, env.rows, 1))
    ax.grid(True, zorder=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    return fig, ax


def _draw_cell_overlays(
    ax,
    env: SlipperyGridWorld,
    obstacles: Optional[List[Tuple[int, int]]] = None,
    current_goal: Optional[Tuple[int, int]] = None,
) -> None:
    """Highlight special cells: start (blue), goals (gold/green), obstacles (red).

    Args:
        ax: matplotlib Axes to draw on.
        env: environment instance (used for start, extra goals).
        obstacles: obstacle positions to draw; falls back to env._obstacles.
        current_goal: primary goal position; falls back to env.goal_row_column.
    """
    if obstacles is not None:
        obs_list = obstacles
    elif hasattr(env, "_all_obstacle_positions"):
        obs_list = env._all_obstacle_positions()
    else:
        obs_list = getattr(env, "_obstacles", [])
    obs_set = set(map(tuple, obs_list))
    extra_goals = getattr(env, "_extra_goals", set())
    primary_goal = tuple(current_goal) if current_goal is not None else env.goal_row_column

    # Obstacles — red fill + ✕
    for (r, c) in obs_set:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                facecolor="tomato", alpha=0.7, zorder=1))
        ax.text(c, r, "✕", ha="center", va="center",
                fontsize=14, color="darkred", fontweight="bold", zorder=3)

    # Extra goals — green fill + G
    for (r, c) in extra_goals:
        if (r, c) not in obs_set:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                    facecolor="lightgreen", alpha=0.7, zorder=1))
            ax.text(c, r, "G", ha="center", va="center",
                    fontsize=14, color="darkgreen", fontweight="bold", zorder=3)

    # Primary goal — gold fill + G
    gr, gc = primary_goal
    if (gr, gc) not in obs_set:
        ax.add_patch(Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                                facecolor="gold", alpha=0.7, zorder=1))
        ax.text(gc, gr, "G", ha="center", va="center",
                fontsize=14, fontweight="bold", zorder=3)

    # Start — light blue fill + S
    sr, sc = env.start_row_column
    ax.add_patch(Rectangle((sc - 0.5, sr - 0.5), 1, 1,
                            facecolor="lightskyblue", alpha=0.7, zorder=1))
    ax.text(sc, sr, "S", ha="center", va="center",
            fontsize=14, fontweight="bold", zorder=3)


def plot_policy(
    env: SlipperyGridWorld,
    policy: np.ndarray,
    filename: Optional[str] = None,
    title: str = "Policy",
    show: bool = True,
) -> None:
    """Visualize policy for each state.

    Special cells are highlighted: start (blue), goal(s) (gold/green), obstacles (red).
    Arrows are omitted on obstacle and goal cells.

    Args:
        env (SlipperyGridWorld): Initialized environment.
        policy (np.ndarray): Policy (deterministic action per each state).
        filename (Optional[str], optional): Where to save the plot. Defaults to None.
        title (str, optional): Defaults to "Policy".
    """
    fig, ax = _base_grid_figure(env, title=title)

    all_obs = (
        env._all_obstacle_positions()
        if hasattr(env, "_all_obstacle_positions")
        else getattr(env, "_obstacles", [])
    )
    special = (
        {env.start_row_column, env.goal_row_column}
        | getattr(env, "_extra_goals", set())
        | set(map(tuple, all_obs))
    )

    for s in range(env.num_states):
        r, c = env.state_to_row_column(s)
        if (r, c) in special:
            continue
        a = int(policy[s])
        ax.text(c, r, ARROWS[a], ha="center", va="center", fontsize=14, zorder=3)

    _draw_cell_overlays(ax, env)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_value_heatmap(
    env: SlipperyGridWorld,
    V: np.ndarray,
    filename: Optional[str] = None,
    title: str = "State Value",
    show: bool = True,
) -> None:
    """Produces a heatmap of V(s) with special cells highlighted.

    Args:
        env (SlipperyGridWorld): Initialized environment.
        V (np.ndarray): V(s).
        filename (Optional[str], optional): Where to save the plot. Defaults to None.
        title (str, optional): Defaults to "State Value".
    """
    V_grid = V.reshape(env.rows, env.cols)
    fig, ax = plt.subplots()
    im = ax.imshow(V_grid, zorder=0)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _draw_cell_overlays(ax, env)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def render_episode_frames(
    env: SlipperyGridWorld,
    trajectory: List,
    out_dir: str = "frames",
    prefix: str = "frame",
    show_executed_action: bool = True,
) -> List[str]:
    """Save one PNG per step with agent position, obstacles, and current goal.

    Trajectory entries may be 6-tuples (s, a_int, a_exec, r, s_next, done)
    or 7-tuples with an additional info dict as the last element.

    Args:
        env (SlipperyGridWorld): Initialized environment.
        trajectory: list of step tuples produced by run_episode.
        out_dir (str): Directory for output PNGs.
        prefix (str): Filename prefix.
        show_executed_action (bool): Include action details in frame title.

    Returns:
        List[str]: Paths to saved PNG files.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for t, step_data in enumerate(trajectory):
        s, a_intended, a_exec, r, s_next, done = step_data[:6]
        info = step_data[6] if len(step_data) > 6 else {}

        r_next, c_next = env.state_to_row_column(s_next)

        # Dynamic positions from info (present when modifications are active)
        obstacles = [tuple(o) for o in info.get("obstacles", getattr(env, "_obstacles", []))]
        current_goal = info.get("goal", env.goal_row_column)
        if not isinstance(current_goal, tuple):
            current_goal = tuple(current_goal)

        fig, ax = _base_grid_figure(env)
        _draw_cell_overlays(ax, env, obstacles=obstacles, current_goal=current_goal)

        # Agent
        ax.add_patch(Rectangle((c_next - 0.5, r_next - 0.5), 1, 1,
                                facecolor="mediumpurple", alpha=0.8, zorder=1))
        ax.text(c_next, r_next, "A", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white", zorder=3)

        if show_executed_action:
            ax.set_title(
                f"t={t}  intended={ARROWS[a_intended]}  executed={ARROWS[a_exec]}"
                f"  r={r:.2f}  done={done}"
            )
        else:
            ax.set_title(f"t={t}  r={r:.2f}  done={done}")

        path = os.path.join(out_dir, f"{prefix}_{t:04d}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def run_to_gif(
    env: SlipperyGridWorld,
    Q: Optional[np.ndarray] = None,
    policy: Optional[np.ndarray] = None,
    gif_path: str = "episode.gif",
    fps: int = 6,
    frames_dir: str = "frames",
) -> None:
    """Create a GIF for a single agent run.

    Args:
        env (SlipperyGridWorld): Initialized environment.
        Q (Optional[np.ndarray]): Q(s,a).
        policy (Optional[np.ndarray]): pi(s).
        gif_path (str): Output GIF path.
        fps (int): Frames per second.
        frames_dir (str): Directory for intermediate frame PNGs.
    """
    roll = run_episode(env, Q=Q, policy=policy)
    frames = render_episode_frames(env, roll["trajectory"], out_dir=frames_dir, prefix="ep")
    imgs = [imageio.imread(p) for p in frames]
    imageio.mimsave(gif_path, imgs, duration=1.0 / fps)


def greedy_policy_from_V(V: np.ndarray, env: SlipperyGridWorld, gamma: float):
    """Return greedy policy derived from value function V(s).

    Args:
        V (np.ndarray): Array of values for each state.
        env (SlipperyGridWorld): Initialized environment.
        gamma (float): Discount factor (0 < gamma < 1).

    Returns:
        np.ndarray: pi(s)
    """
    policy = np.zeros(len(V))
    for state in range(len(V)):
        q_a = [-np.inf] * len(ACTIONS)
        for a in ACTIONS:
            q = 0.0
            for p, s_next in env.get_transition_distribution(state, a):
                r = env.reward(state, a, s_next)
                if env.is_terminal_state(s_next):
                    q += p * r
                else:
                    q += p * (r + gamma * V[s_next])
            q_a[a] = q
        policy[state] = int(np.argmax(q_a))
    return policy


def run_episode(
    env: SlipperyGridWorld,
    Q: Optional[np.ndarray] = None,
    policy: Optional[np.ndarray] = None,
    seed: int = None,
) -> Dict:
    """Roll out a single episode.

    Args:
        env (SlipperyGridWorld): environment.
        Q (Optional[np.ndarray]): Q(s,a). Defaults to None.
        policy (Optional[np.ndarray]): pi(s). Defaults to None.
        seed (int): random seed for env.rng. Defaults to None.

    Returns:
        Dict: episode stats with keys return, steps, success, trajectory.
              Each trajectory entry is a 7-tuple:
              (s, a_intended, a_executed, reward, s_next, done, info).
    """
    assert (Q is not None) or (policy is not None), "Provide Q or policy"

    s = env.reset()
    if seed is not None:
        env.rng = random.Random(seed)
    done = False
    total_return = 0.0
    steps = 0
    traj = []

    while not done:
        a = int(policy[s]) if policy is not None else int(np.argmax(Q[s]))
        s_next, r, done, info = env.step(a)
        traj.append((s, a, info.get("executed_action", a), r, s_next, done, info))
        total_return += float(r)
        s = s_next
        steps += 1

    success = env.is_terminal_state(s)
    return {
        "return": total_return,
        "steps": steps,
        "success": bool(success),
        "trajectory": traj,
    }


def evaluate(
    env: SlipperyGridWorld,
    Q: Optional[np.ndarray] = None,
    policy: Optional[np.ndarray] = None,
    n_episodes: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Evaluate Q(s,a) or a deterministic policy over multiple episodes.

    Args:
        env (SlipperyGridWorld): Initialized environment.
        Q (Optional[np.ndarray]): Q(s,a). Defaults to None.
        policy (Optional[np.ndarray]): pi(s). Defaults to None.
        n_episodes (int): Number of evaluation episodes. Defaults to 200.
        seed (int): Master random seed. Defaults to 0.

    Returns:
        Dict[str, float]: avg_return, std_return, success_rate, avg_steps.
    """
    rng = np.random.default_rng(seed)
    returns, steps, success = [], [], []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 1_000_000))
        res = run_episode(env, Q=Q, policy=policy, seed=ep_seed)
        returns.append(res["return"])
        steps.append(res["steps"])
        success.append(1.0 if res["success"] else 0.0)

    return {
        "avg_return": float(np.round(np.mean(returns), decimals=4)),
        "std_return": float(np.round(np.std(returns), decimals=4)),
        "success_rate": float(np.round(np.mean(success), decimals=4)),
        "avg_steps": float(np.round(np.mean(steps), decimals=4)),
    }
