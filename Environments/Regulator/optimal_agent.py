import numpy as np


class OptimalAgent:
    def get_action(self, state):
        u = -np.array([0.54986356, 0.53058622, 1.61575641, 1.44154412, 8.16303748]).dot(state[np.newaxis].T)
        return u
