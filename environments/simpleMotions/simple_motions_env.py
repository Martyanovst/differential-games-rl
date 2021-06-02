from numpy import hstack
import numpy as np


class SimpleMotions:
    def __init__(self, x0=1, terminal_t=2, dt=0.01):
        self.terminal_t = terminal_t
        self.x0 = x0
        self.state_dim = 2
        self.action_dim = 1
        self.action_min = np.array([-1])
        self.action_max = np.array([+1])
        self.state = hstack((0, self.x0))
        self.done = False
        self.dt = dt
        self.r = 0.5
        self.g = [1]
        self.beta = 0.5

    def reset(self):
        self.state = hstack((0, self.x0))
        self.done = False
        return self.state

    def step(self, u_action):
        u = u_action[0]
        t, x = self.state
        x = x + u * self.dt
        reward = -0.5 * (u ** 2) * self.dt
        t += self.dt
        if t >= self.terminal_t:
            self.done = True
            reward -= (x ** 2)
        self.state = hstack((t, x))
        return self.state, reward, int(self.done), None
