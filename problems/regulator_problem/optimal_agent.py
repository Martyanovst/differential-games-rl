import numpy as np


class OptimalAgent:
    def get_action(self, state):
        u = np.array([-0.9243, 0.1711, 0.0161, 0.0392, 0.2644]) @ state
        return np.array([u])
