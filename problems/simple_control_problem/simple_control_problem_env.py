from numpy import hstack


class SimpleControlProblem:
    def __init__(self, x0=1, terminal_t=2, dt=0.005):
        self.terminal_t = terminal_t
        self.x0 = x0
        self.state = hstack((0, self.x0))
        self.done = False
        self.dt = dt

    def reset(self):
        self.state = hstack((0, self.x0))
        self.done = False
        return self.state

    def step(self, u_action):
        u = u_action[0]
        t, x = self.state
        x = x + u * self.dt
        reward = 0.5 * (u ** 2) * self.dt
        t += self.dt
        if t >= self.terminal_t:
            self.done = True
            reward += (x ** 2)
        self.state = hstack((t, x))
        return self.state, reward, int(self.done), None
