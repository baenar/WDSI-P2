from __future__ import annotations
import random
import numpy as np


class QLearning:
    """Off-policy TD control (Q-learning, Watkins 1989)."""

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        episodes: int = 8000,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.seed = seed

    def train(self, env) -> np.ndarray:
        """Train on env, return learned Q(s, a) table."""
        if self.seed is not None:
            random.seed(self.seed)

        epsilon = self.epsilon_start
        Q = np.zeros((env.num_states, env.nA))

        for _ in range(self.episodes):
            state = env.reset()
            done = False
            while not done:
                if random.random() < epsilon:
                    action = random.randint(0, env.nA - 1)
                else:
                    action = int(np.argmax(Q[state]))

                next_state, reward, done, _ = env.step(action)

                best_next = int(np.argmax(Q[next_state]))
                td_target = reward + self.gamma * Q[next_state, best_next] * (not done)
                Q[state, action] += self.alpha * (td_target - Q[state, action])

                state = next_state
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

        return Q
