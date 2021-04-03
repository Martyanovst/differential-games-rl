import numpy as np


class DubinsCar:
    def __init__(self, initial_state=np.array([0, 0, 0, 0], dtype=np.float64), dt=0.1, terminal_time=2 * np.pi, inner_step_n=20,
                 action_min=-0.5, action_max=1):
        self.initial_state = initial_state
        self.state = initial_state
        self.state_dim = 4
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        t, x, y, phi = self.state
        t += self.dt
        x += np.cos(phi) * self.dt
        y += np.cos(phi) * self.dt
        phi += action[0] * self.dt

        reward = 0.01 * (action[0] ** 2) * self.dt
        done = False
        if t >= self.terminal_time:
            reward += np.abs(x - 4) + np.abs(y) + np.abs(phi - 0.75 * np.pi)
            done = True
        self.state = np.array([t, x, y, phi])
        return self.state, reward, done, None
