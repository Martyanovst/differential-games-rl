import numpy as np


class OptimalUAgent:
    def __init__(self, env):
        super().__init__()
        self.u_action_max = env.u_action_max
        self.terminal_time = env.terminal_time

    def get_action(self, state):
        t, x = state
        if x < t - self.terminal_time:
            return self.u_action_max
        elif x > self.terminal_time - t:
            return - self.u_action_max
        else:
            return 0


class OptimalVAgent:
    def __init__(self, env):
        super().__init__()
        self.v_action_max = env.v_action_max
        self.terminal_time = env.terminal_time

    def get_action(self, state):
        t, x = state
        if x < t - self.terminal_time:
            return self.v_action_max
        elif x > self.terminal_time - t:
            return - self.v_action_max
        else:
            return 0
