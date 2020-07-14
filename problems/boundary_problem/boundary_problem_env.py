from numpy import hstack


class BoundaryProblem:
    def __init__(self, x1=-1, x2=-1, dt=0.005):
        self.state = hstack((0, x1, x2))
        self.done = False
        self.initial_x1 = x1
        self.initial_x2 = x2
        self.terminal_t = 1
        self.dt = dt

    def reset(self):
        self.state = hstack((0, self.initial_x1, self.initial_x2))
        self.done = False
        return self.state

    def step(self, u_action):
        u = u_action[0]
        t, x1, x2 = self.state
        x1 = x1 + x2 * self.dt
        x2 = x2 + u * self.dt
        reward = (x1 ** 2 + u ** 2) * self.dt
        t = t + self.dt
        if t >= self.terminal_t:
            reward += x1 + x2
            self.done = True
        self.state = hstack((t, x1, x2))
        return self.state, reward, int(self.done), None
