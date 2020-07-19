from numpy import hstack


class NonlinearProblem:
    def __init__(self, x1=1, x2=1, dt=0.005):
        self.state = hstack((0, x1, x2))
        self.done = False
        self.initial_x1 = x1
        self.initial_x2 = x2
        self.dt = dt
        self.optimal_v = 0.5 * (x1 ** 2) + x2 ** 2

    def reset(self):
        self.state = hstack((0, self.initial_x1, self.initial_x2))
        self.done = False
        return self.state

    def step(self, u_action):
        u = u_action[0]
        t, x1, x2 = self.state
        x1 = x1 + (-x1 + x2) * self.dt
        x2 = x2 + (-0.5 * (x1 + x2) + 0.5 * (x1 ** 2) * x2 + x1 * u) * self.dt
        reward = (x1 ** 2 + x2 ** 2 + u ** 2) * self.dt
        t = t + self.dt
        self.state = hstack((t, x1, x2))
        return self.state, reward, int(self.done), None
