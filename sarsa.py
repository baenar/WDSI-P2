from __future__ import annotations
import random
import numpy as np


class SARSA:
    """On-policy TD control (SARSA)."""

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

        for _ in range(self.episodes):
            state = env.reset()
            action = self._epsilon_greedy(Q, state, env.nA, epsilon)
            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = self._epsilon_greedy(Q, next_state, env.nA, epsilon)

                td_target = reward + self.gamma * Q[next_state, next_action] * (not done)
                Q[state, action] += self.alpha * (td_target - Q[state, action])

                state = next_state
                action = next_action
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

        return Q
