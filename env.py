from __future__ import annotations
from collections import defaultdict
import random
from typing import List, Optional, Tuple


# Action mapping: 0=Up, 1=Right, 2=Down, 3=Left
ACTIONS = (0, 1, 2, 3)
ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (0, +1),
    2: (+1, 0),
    3: (0, -1),
}


def _perpendicular_actions(a: int) -> Tuple[int, int]:
    if a in (0, 2):   # Up or Down
        return (3, 1) # Left, Right
    else:             # Left or Right
        return (0, 2) # Up, Down

class SlipperyGridWorld:
    """
    Simple tabular GridWorld with slippery actions.

    State space: integers 0..(rows*cols-1), row-major.
    Actions: 0=Up, 1=Right, 2=Down, 3=Left.

    Slippery dynamics:
      - with prob (1 - slip_prob): take intended action
      - with prob slip_prob: take one of the two perpendicular actions (uniformly)

    Optional modifications (disabled by default, enable via constructor):
      - Moving obstacles  : obstacles move every `obstacle_move_every` steps;
                            stepping on one adds `obstacle_penalty` to the reward.
      - Moving goal       : the primary goal drifts every `goal_move_every` steps.
      - Multiple goals    : extra goal positions; reaching any of them ends the episode.
      - Corner obstacles  : `corner_obstacle_count` obstacles confined to each of the
                            bottom-left and top-right corner squares of side `corner_size`.
                            They share `obstacle_penalty` and `obstacle_move_every`.
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (3, 3),
        slip_prob: float = 0.2,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        wall_penalty: float = 0.0,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        # --- Modification 1: moving obstacles ---
        obstacles: Optional[List[Tuple[int, int]]] = None,
        obstacle_penalty: float = -10.0,
        obstacle_move_every: int = 1,
        # --- Modification 1b: corner-constrained obstacles ---
        corner_size: int = 0,
        corner_no: int = 2,
        corner_obstacle_count: int = 0,
        # --- Modification 2: moving goal ---
        moving_goal: bool = False,
        goal_move_every: int = 5,
        # --- Modification 3: multiple goals ---
        goals: Optional[List[Tuple[int, int]]] = None,

    ):
        assert rows > 0 and cols > 0
        assert 0.0 <= slip_prob <= 1.0

        self.rows = rows
        self.cols = cols
        self.start_row_column = start
        self.goal_row_column = goal
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
        self.max_steps = max_steps

        self.rng = random.Random(seed)
        self._steps = 0
        self._agent_row_column = start

        self.num_states = rows * cols
        self.nA = len(ACTIONS)

        # Modification 1: moving obstacles
        self._initial_obstacles: List[Tuple[int, int]] = list(obstacles) if obstacles else []
        self._obstacles: List[Tuple[int, int]] = list(self._initial_obstacles)
        self.obstacle_penalty = obstacle_penalty
        self.obstacle_move_every = obstacle_move_every

        # Modification 2: moving goal
        self.moving_goal = moving_goal
        self.goal_move_every = goal_move_every
        self._initial_goal: Tuple[int, int] = goal

        # Modification 3: multiple goals (extra positions on top of primary goal)
        self._extra_goals: set = set(goals) if goals else set()

        # Modification 4: corner-constrained obstacles
        self.corner_size = corner_size
        self.corner_no = corner_no
        self.corner_obstacle_count = corner_obstacle_count
        self._initial_bl_obstacles: List[Tuple[int, int]] = []
        self._initial_tr_obstacles: List[Tuple[int, int]] = []
        self._bl_obstacles: List[Tuple[int, int]] = []
        self._tr_obstacles: List[Tuple[int, int]] = []
        if corner_size > 0 and corner_obstacle_count > 0:
            self._place_corner_obstacles()

    @property
    def _all_goals(self) -> set:
        """All currently active goal positions."""
        return {self.goal_row_column} | self._extra_goals

    # --- helpers ---
    def row_column_to_state(self, r: int, c: int) -> int:
        return r * self.cols + c

    def state_to_row_column(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.cols)

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _apply_action(self, r: int, c: int, a: int) -> Tuple[int, int]:
        dr, dc = ACTION_TO_DELTA[a]
        nr, nc = r + dr, c + dc
        if self._in_bounds(nr, nc):
            return nr, nc
        return r, c

    def _sample_action_with_slip(self, intended: int) -> int:
        if self.rng.random() >= self.slip_prob:
            return intended
        left, right = _perpendicular_actions(intended)
        return left if self.rng.random() < 0.5 else right

    def _apply_action_stateless(self, state:int, action: int):
        r, c = self.state_to_row_column(state)
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = r + dr, c + dc
        if self._in_bounds(nr, nc):
            return self.row_column_to_state(nr, nc)
        return self.row_column_to_state(r, c)

    def _move_obstacles(self) -> None:
        """Move each obstacle one step in a random valid direction (or stay)."""
        forbidden = self._all_goals | {self.start_row_column}
        new_positions: List[Tuple[int, int]] = []
        taken: set = set()
        for obs in self._obstacles:
            r, c = obs
            candidates = [
                (r + dr, c + dc)
                for dr, dc in ACTION_TO_DELTA.values()
                if self._in_bounds(r + dr, c + dc)
                   and (r + dr, c + dc) not in forbidden
                   and (r + dr, c + dc) not in taken
            ]
            if obs not in taken and obs not in forbidden:
                candidates.append(obs)
            new_pos = self.rng.choice(candidates) if candidates else obs
            new_positions.append(new_pos)
            taken.add(new_pos)
        self._obstacles = new_positions

    def _all_obstacle_positions(self) -> List[Tuple[int, int]]:
        """All current obstacle positions (free-moving + corner-constrained)."""
        return self._obstacles + self._bl_obstacles + self._tr_obstacles

    def _place_corner_obstacles(self) -> None:
        """Randomly place corner_obstacle_count obstacles in each corner square."""
        forbidden = self._all_goals | {self.start_row_column} | set(self._obstacles)

        bl_cells = [
            (r, c)
            for r in range(self.rows - self.corner_size, self.rows)
            for c in range(0, self.corner_size)
            if (r, c) not in forbidden
        ]
        tr_cells = [
            (r, c)
            for r in range(0, self.corner_size)
            for c in range(self.cols - self.corner_size, self.cols)
            if (r, c) not in forbidden
        ]

        n = self.corner_obstacle_count
        self._initial_bl_obstacles = self.rng.sample(bl_cells, min(n, len(bl_cells)))
        self._initial_tr_obstacles = (
            self.rng.sample(tr_cells, min(n, len(tr_cells)))
            if self.corner_no >= 2 else []
        )
        self._bl_obstacles = list(self._initial_bl_obstacles)
        self._tr_obstacles = list(self._initial_tr_obstacles)

    def _move_corner_obstacles(self) -> None:
        """Move corner obstacles within their respective corner regions."""
        bl_region = frozenset(
            (r, c)
            for r in range(self.rows - self.corner_size, self.rows)
            for c in range(0, self.corner_size)
        )
        tr_region = frozenset(
            (r, c)
            for r in range(0, self.corner_size)
            for c in range(self.cols - self.corner_size, self.cols)
        )
        forbidden = self._all_goals | {self.start_row_column}

        def _move_in_region(
            obstacles: List[Tuple[int, int]], region: frozenset
        ) -> List[Tuple[int, int]]:
            new_positions: List[Tuple[int, int]] = []
            taken: set = set()
            for obs in obstacles:
                r, c = obs
                candidates = [
                    (r + dr, c + dc)
                    for dr, dc in ACTION_TO_DELTA.values()
                    if (r + dr, c + dc) in region
                       and (r + dr, c + dc) not in forbidden
                       and (r + dr, c + dc) not in taken
                ]
                if obs not in taken and obs not in forbidden:
                    candidates.append(obs)
                new_pos = self.rng.choice(candidates) if candidates else obs
                new_positions.append(new_pos)
                taken.add(new_pos)
            return new_positions

        self._bl_obstacles = _move_in_region(self._bl_obstacles, bl_region)
        if self.corner_no >= 2:
            self._tr_obstacles = _move_in_region(self._tr_obstacles, tr_region)

    def _move_goal(self) -> None:
        """Move the primary goal one step in a random valid direction (or stay)."""
        r, c = self.goal_row_column
        forbidden = {self.start_row_column} | set(self._all_obstacle_positions()) | self._extra_goals
        candidates = [(r, c)]
        for dr, dc in ACTION_TO_DELTA.values():
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc) and (nr, nc) not in forbidden:
                candidates.append((nr, nc))
        self.goal_row_column = self.rng.choice(candidates)

    # --- public API ---
    def reset(self, start: Optional[Tuple[int, int]] = None) -> int:
        """Reset environment to start state specified (optional).

        Args:
            start (Optional[Tuple[int, int]], optional): If not specified, 
            takes start state from environment initialization. 
            Defaults to None.

        Returns:
            int: Reset agent's state.
        """
        if start is not None:
            self.start_row_column = start
        self._agent_row_column = self.start_row_column
        self._steps = 0
        if self.moving_goal:
            self.goal_row_column = self._initial_goal
        if self._initial_obstacles:
            self._obstacles = list(self._initial_obstacles)
        if self._initial_bl_obstacles:
            self._bl_obstacles = list(self._initial_bl_obstacles)
        if self._initial_tr_obstacles:
            self._tr_obstacles = list(self._initial_tr_obstacles)
        return self.row_column_to_state(*self._agent_row_column)

    def get_transition_distribution(self, state: int, action: int) -> list[tuple[float, int]]:
        """Returns env transition probability distribution for given (s,a) and respective next states (s').
            Because the environment is slippery, attempting one action may lead to several possible next states.

        Args:
            state (int): Current state (s).
            action (int): Action to attempt (a).

        Returns:
            List of (probability, next_state) pairs.
        """
        assert action in ACTIONS, f"Invalid action {action}. Use 0=U,1=R,2=D,3=L."
        probs = [0]*len(ACTION_TO_DELTA)
        probs[action] = 1 - self.slip_prob
        act_s_1, act_s_2 = _perpendicular_actions(action)
        probs[act_s_1] = self.slip_prob/2
        probs[act_s_2] = self.slip_prob/2
        state_probs = defaultdict(float)

        for a_real, prob in enumerate(probs):
            if prob == 0:
                continue

            next_state = self._apply_action_stateless(state, a_real)
            state_probs[next_state] += prob

        return [(p, s_next) for s_next, p in state_probs.items()]
    
    def is_terminal_state(self, state: int) -> bool:
        """Check if the current state is terminal in the environment."""
        return self.state_to_row_column(state) in self._all_goals

    def step(self, action: int):
        """Perform one step in the environment.

        Args:
            action (int): Action to perform [0, 1, 2, 3].

        Returns:
            Next state, reward, flag done, info dictionary
        """
        assert action in ACTIONS, f"Invalid action {action}. Use 0=U,1=R,2=D,3=L."
        self._steps += 1

        intended = action
        executed = self._sample_action_with_slip(intended)

        r, c = self._agent_row_column
        nr, nc = self._apply_action(r, c, executed)
        hit_wall = (nr, nc) == (r, c)
        self._agent_row_column = (nr, nc)

        at_goal = self._agent_row_column in self._all_goals
        all_obs = self._all_obstacle_positions()
        on_obstacle = bool(all_obs) and self._agent_row_column in set(all_obs)

        done = at_goal
        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True

        if at_goal:
            reward = self.goal_reward
        else:
            reward = self.step_reward
            if hit_wall:
                reward += self.wall_penalty
            if on_obstacle:
                reward += self.obstacle_penalty

        # Environment dynamics after agent acts
        if self._steps % self.obstacle_move_every == 0:
            if self._obstacles:
                self._move_obstacles()
            if self._bl_obstacles or self._tr_obstacles:
                self._move_corner_obstacles()
        if self.moving_goal and self._steps % self.goal_move_every == 0:
            self._move_goal()

        info = {
            "intended_action": intended,
            "executed_action": executed,
            "steps": self._steps,
            "obstacles": self._all_obstacle_positions(),
            "goal": self.goal_row_column,
        }
        return self.row_column_to_state(*self._agent_row_column), reward, done, info

    def set_goal(self, goal: Tuple[int, int]):
        """Specify the goal state

        Args:
            goal (Tuple[int, int]): (row, column)
        """
        self.goal_row_column = goal
        self._initial_goal = goal

    def reward(self, state: int, action: int, next_state: int) -> float:
        """Return the reward R(s, a, s') for a transition.
        
        In this simplified GridWorld, reward depends only on the next state

        Args:
            state (int): State for which the reward should be retrieved.
            action (int): Attempted action.
            next_state (int): Next state after action in state.

        Returns:
            float: reward from the environment.
        """
        if self.state_to_row_column(state) in self._all_goals:
            return 0.0
        next_rc = self.state_to_row_column(next_state)
        if next_rc in self._all_goals:
            return self.goal_reward
        r = self.step_reward
        if state == next_state:
            r += self.wall_penalty
        all_obs = self._all_obstacle_positions()
        if all_obs and next_rc in set(all_obs):
            r += self.obstacle_penalty
        return r

    def set_size(self, rows: int, cols: int, start: Optional[Tuple[int, int]] = None, goal: Optional[Tuple[int, int]] = None) -> None:
        """Set enviroment grid size.

        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.
            start (Optional[Tuple[int, int]], optional): Start state (row, column). Defaults to None.
            goal (Optional[Tuple[int, int]], optional): Goal state (row, column). Defaults to None.
        """
        assert rows > 0 and cols > 0
        self.rows, self.cols = rows, cols
        self.num_states = rows * cols
        if start is not None:
            self.start_row_column = start
        if goal is not None:
            self.goal_row_column = goal
        r, c = self._agent_row_column
        self._agent_row_column = (min(max(r, 0), rows - 1), min(max(c, 0), cols - 1))