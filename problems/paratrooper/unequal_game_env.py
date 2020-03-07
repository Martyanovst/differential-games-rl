import numpy as np
import pygame


class UnequalGame:

    def __init__(self, initial_x=1, dt=0.005, terminal_time=2, u_action_max=2, v_action_max=1):
        self.done = False
        self.u_action_max = u_action_max
        self.v_action_max = v_action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_x = initial_x
        self.steps_count = int(terminal_time / dt)
        self.state = self.reset()
        # self.init_rendering()

    def reset(self):
        self.done = False
        self.state = np.array([0, self.initial_x])
        return self.state

    def step(self, u_action, v_action):
        t, x = self.state
        x = x + (u_action - v_action) * self.dt
        t += self.dt
        self.state = np.array([t, x])
        # self.screen.blit(self.point, (x + 300, int(t * 100)))
        # pygame.display.update()
        # pygame.time.delay(100)
        reward = 0
        if t >= self.terminal_time:
            reward = x ** 2
            self.done = True

        return self.state, reward, int(self.done), None

    def init_rendering(self):
        self.screen = pygame.display.set_mode((640, 480))
        self.point = pygame.image.load('point.png')
