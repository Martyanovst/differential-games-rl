import numpy as np


class OptimalAgent:
    def get_action(self, state):
        t, x1, x2 = state
        u = -x1 * x2
        return np.array([u])
