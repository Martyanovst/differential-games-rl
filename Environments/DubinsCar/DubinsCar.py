import numpy as np


class DubinsCar:
    def __init__(self, initial_state=np.array([0, 0, 0, 0]), dt=0.1, terminal_time=2 * np.pi, inner_step_n=20,
                 action_min=np.array([-0.5]), action_max=np.array([1])):
        self.state_dim = 4
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            self.state = self.state + np.array(
                [1, np.cos(self.state[3]), np.sin(self.state[3]), action[0]]) * self.inner_dt

        reward = - 0.01 * (action[0] ** 2) * self.dt
        done = False
        if self.state[0] >= self.terminal_time:
            reward -= np.abs(self.state[1] - 4) + np.abs(self.state[2]) + np.abs(self.state[3] - 0.75 * np.pi)
            done = True

        return self.state, reward, done, None


class DubinsCar_SymmetricActionInterval:
    def __init__(self, initial_state=np.array([0, 0, 0, 0]), dt=0.1, terminal_time=2 * np.pi, inner_step_n=20,
                 action_min=np.array([-1]), action_max=np.array([1])):
        self.state_dim = 4
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.beta = 0.01
        self.r = 0.01
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)
        action = action * 0.75 + 0.25

        for _ in range(self.inner_step_n):
            self.state = self.state + np.array(
                [1, np.cos(self.state[3]), np.sin(self.state[3]), action[0]]) * self.inner_dt

        if self.state[0] >= self.terminal_time:
            reward = -np.abs(self.state[1] - 4) - np.abs(self.state[2]) - np.abs(self.state[3] - 0.75 * np.pi)
            done = True
        else:
            reward = -self.r * (action[0] ** 2) * self.dt
            done = False

        return self.state, reward, done, None
