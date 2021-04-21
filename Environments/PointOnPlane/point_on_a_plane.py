from numpy import array, dot, exp, hstack
from numpy.linalg import norm


class PointOnAPlane:
    def __init__(self, initial_x=array([0, 2, 2, 1]), terminal_t=4, dt=0.005):
        self.initial_x = initial_x
        self.terminal_t = terminal_t
        self.dt = dt
        self.u_action_max = 1
        self.v_action_max = 1
        self.done = False
        self.state = self.reset()

    def A(self, t):
        return array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [-4 * exp(t / 5), 0, -0.1 * exp(t / 5), 0],
                      [0, -4 * exp(t / 5), 0, -0.1 * exp(t / 5)]])

    def B(self, t):
        return array([[0, 0],
                      [0, 0],
                      [-8, 0],
                      [0, -8]])

    def C(self, t):
        return array([[0, 0],
                      [0, 0],
                      [2.4 * exp(t / 5), 0],
                      [0, 2.4 * exp(t / 5)]])

    def action_scaling(self, action, a, b):
        n = norm(action)
        if n <= 1:
            return array([a, b]) * action
        else:
            return array([a, b]) * action / n

    def reset(self):
        self.state = hstack((0, self.initial_x))
        self.done = False
        return self.state

    def step(self, u_action, v_action):
        u = self.action_scaling(u_action, 6, 2)
        v = self.action_scaling(v_action, 2, 4)
        t, x = self.state[0], self.state[1:]
        x = x + (dot(self.A(t), x) + dot(self.B(t), u) + dot(self.C(t), v)) * self.dt
        t = t + self.dt
        self.state = hstack((t, x))

        reward = 0
        self.done = False
        if t >= self.terminal_t:
            reward = norm(array([x[0], x[1]]))
            self.done = True

        return self.state, reward, self.done, None
