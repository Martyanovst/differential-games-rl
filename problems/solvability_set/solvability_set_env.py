import math

import numpy as np


class SolvabilitySet:

    def __init__(self, initial_x=0, initial_y=0, dt=0.04, terminal_time=7.8, u_action_max=0.25, v_action_max=0.5):
        self.done = False
        self.u_action_max = u_action_max
        self.v_action_max = v_action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.steps_count = int(terminal_time / dt)
        self.state = self.reset()

    def reset(self):
        self.done = False
        self.initial_x = np.random.uniform(-4, 4, 1)[0]
        self.initial_y = np.random.uniform(-4, 4, 1)[0]
        self.state = np.array([0, self.initial_x, self.initial_y])
        return self.state

    def step(self, u_action, v_action):
        t, x, y = self.state
        x = x + v_action[0] * self.dt
        y = y + u_action[0] * self.dt
        t += self.dt
        self.state = np.array([t, x, y])
        reward = 0
        if abs(t - self.terminal_time) < 0.0001:
            self.done = True
            delta = math.sqrt(x ** 2 + y ** 2)
            reward = delta - 4

        return self.state, reward, int(self.done), None
