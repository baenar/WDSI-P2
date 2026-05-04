from __future__ import annotations
import random
import numpy as np


class DynaQ:
    """Dyna-Q: Q-learning with model-based planning (Sutton 1990)."""

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        episodes: int = 8000,
        n_planning_steps: int = 10,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.n_planning_steps = n_planning_steps
        self.seed = seed

    def _epsilon_greedy(self, Q: np.ndarray, state: int, n_actions: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, n_actions - 1)
        return int(np.argmax(Q[state]))

    def train(self, env) -> np.ndarray:
        """Train on env, return learned Q(s, a) table."""
        if self.seed is not None:
            random.seed(self.seed)

        epsilon = self.epsilon_start
        Q = np.zeros((env.num_states, env.nA))
        model: dict[tuple[int, int], tuple[int, float]] = {}

        for _ in range(self.episodes):
            state = env.reset()
            done = False
            while not done:
                action = self._epsilon_greedy(Q, state, env.nA, epsilon)
                next_state, reward, done, _ = env.step(action)

                # Q-learning update (off-policy)
                td_target = reward + self.gamma * np.max(Q[next_state]) * (not done)
                Q[state, action] += self.alpha * (td_target - Q[state, action])

                # Store transition in model
                model[(state, action)] = (next_state, reward)

                # Planning: simulate n steps from the learned model
                visited = list(model.keys())
                for _ in range(self.n_planning_steps):
                    s_sim, a_sim = visited[random.randint(0, len(visited) - 1)]
                    s_next_sim, r_sim = model[(s_sim, a_sim)]
                    done_sim = env.is_terminal_state(s_next_sim)
                    td_sim = r_sim + self.gamma * np.max(Q[s_next_sim]) * (not done_sim)
                    Q[s_sim, a_sim] += self.alpha * (td_sim - Q[s_sim, a_sim])

                state = next_state
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

        return Q
