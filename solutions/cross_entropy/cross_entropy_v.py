import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class Agent:

    def __init__(self, env, v_model, batch_size=200, gamma=0.9999):
        self.v = v_model
        self.opt = torch.optim.Adam(self.v.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()
        self.v_target = deepcopy(self.v)
        self.tau = 1e-2
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.dt = env.dt
        self.f = env.f
        self.g = env.g
        self.beta = 1e-2
        self.epsilon = 1 / 2
        self.epsilon_decrease = 0.99
        self.U = [-2, 2]
        self.V = [-1, 1]
        self.batch_size = batch_size

    def min_U(self, states):
        res = []
        for state in states:
            t = state[0]
            x = state[1]
            v_value = min(map(lambda u: self.v(torch.tensor([t, x + self.f(t, x, u) * self.dt])), self.U))
            res.append(v_value)
        return torch.tensor(res).reshape(self.batch_size,1)

    def max_V(self, states):
        res = []
        for state in states:
            t = state[0]
            x = state[1]
            v_value = max(map(lambda v: self.v(torch.tensor([t, x + self.g(t, x, v) * self.dt])), self.V))
            res.append(v_value)
        return torch.tensor(res).reshape(self.batch_size,1)

    def get_u_action(self, state):
        t = state[0]
        x = state[1]
        x1 = torch.tensor([t, x - self.beta], dtype=torch.float32)
        x2 = torch.tensor([t, x + self.beta], dtype=torch.float32)
        if self.v(x1) < self.v(x2):
            y = x - x1[1]
        else:
            y = x - x2[1]
        if y * self.f(t, x, -2) < y * self.f(t, x, 2):
            probs = np.array([1 - self.epsilon, self.epsilon])
            return np.random.choice(self.U, p=probs)
        else:
            probs = np.array([self.epsilon, 1 - self.epsilon])
            return np.random.choice(self.U, p=probs)

    def get_v_action(self, state):
        t = state[0]
        x = state[1]
        x1 = torch.tensor([t, x - self.beta], dtype=torch.float32)
        x2 = torch.tensor([t, x + self.beta], dtype=torch.float32)
        if self.v(x1) > self.v(x2):
            y = x - x1[1]
        else:
            y = x - x2[1]
        if y * self.g(t, x, -1) > y * self.g(t, x, 1):
            probs = np.array([1 - self.epsilon, self.epsilon])
            return np.random.choice(self.V, p=probs)
        else:
            probs = np.array([self.epsilon, 1 - self.epsilon])
            return np.random.choice(self.V, p=probs)

    def state_dict(self):
        return self.v.state_dict()

    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * original_param.data)

    def get_batch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, rewards, dones, next_states = map(np.array, zip(*minibatch))
        states = torch.tensor(states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.from_numpy(dones.astype(np.float32))
        next_states = torch.tensor(next_states, dtype=torch.float32)
        return states, rewards, dones, next_states

    def fit(self, state, reward, done, next_state):
        self.memory.append([state, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            states, rewards, dones, next_states = self.get_batch()
            self.opt.zero_grad()
            ts = states[:, :1]
            xs = states[:, 1:]
            dx = torch.cat((ts + self.dt, xs), 1)
            target = rewards.reshape(self.batch_size, 1) + (1 / 3) * (-(dones - 1)).reshape(self.batch_size, 1) * (
                    self.v(dx).detach() + self.min_U(states).detach() + self.max_V(states).detach())
            loss = self.loss(self.v(states), target)
            loss.backward()
            self.opt.step()
            self.update_targets(self.v_target, self.v)

    def noise_decrease(self):
        self.epsilon *= self.epsilon_decrease
