# a = 0.1664682790
# b = 0.4986113866
# c = 0.9916694222
# d = 0.9583581328
# teta1 = -2.35653*(0.155067 + 0.40471* x1 + 0.197997 * x2)
# teta2 = -9.27761*(0.0969914 + 0.0502915*x1 + 0.033317 * x2)
import math

import numpy as np


class OptimalAgent:
    def __init__(self, env):
        super().__init__()
        self.x1 = env.initial_x1
        self.x2 = env.initial_x2
        self.teta1 = -2.35653 * (0.155067 + 0.40471 * self.x1 + 0.197997 * self.x2)
        self.teta2 = -9.27761 * (0.0969914 + 0.0502915 * self.x1 + 0.033317 * self.x2)

        self.f = lambda t: (1 / math.sqrt(2)) * (
                math.sin(t / math.sqrt(2)) * math.cosh(t / math.sqrt(2)) - math.cos(t / math.sqrt(2)) * math.sinh(
            t / math.sqrt(2)))

        self.df = lambda t: math.sin(t / math.sqrt(2)) * math.sinh(t / math.sqrt(2))

        self.ddf = lambda t: (1 / math.sqrt(2)) * (
                math.sin(t / math.sqrt(2)) * math.cosh(t / math.sqrt(2)) + math.cos(t / math.sqrt(2)) * math.sinh(
            t / math.sqrt(2)))

        self.dddf = lambda t: math.cos(t / math.sqrt(2)) * math.cosh(t / math.sqrt(2))

    def get_action(self, state):
        t, x1, x2 = state
        u = self.df(t) * self.x1 - self.f(t) * self.x2 - self.ddf(t) * self.teta1 + self.dddf(t) * self.teta2
        return np.array([u])
