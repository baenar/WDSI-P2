from __future__ import annotations
import numpy as np

class VIAlgorithm:
    """Value Iteration algorithm for solving MDPs."""

    def __init__(
        self,
        max_iter: int = 100,
        gamma: float = 0.99,
        theta: float = 1e-5
    ):
        self.max_iter = max_iter
        self.gamma = gamma
        self.theta = theta

    def run(self, env):
        """Run on env. return V(s) table."""
        num_states = env.rows * env.cols
        V = np.zeros(num_states)
        for i in range(self.max_iter):
            delta = 0
            for s in range(num_states):
                if(env.is_terminal_state(s)):
                    continue
                v = V[s]
                action_values = []
                for a in range(env.nA):
                    action_value = 0
                    transitions = env.get_transition_distribution(s, a)
                    for prob, next_s in transitions:
                        reward = env.reward(s, a, next_s)
                        action_value += prob * (reward + self.gamma * V[next_s])
                    action_values.append(action_value)
                V[s] = max(action_values)
                delta = max(delta, abs(v - V[s]))
            if delta < self.theta:
                break
        return V