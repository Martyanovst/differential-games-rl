import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class v_model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        state = torch.tensor(np.array([t, x]), dtype=torch.float32)
        return self.model(state)

class NAFAgent:

    def __init__(self, v_model,
                 noise, batch_size=200, gamma=0.9999):
        self.v = v_model
        self.opt = torch.optim.Adam(self.v.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()
        self.v_target = deepcopy(self.v)
        self.tau = 1e-2
        self.memory = deque(maxlen=100000)
        self.gamma = gamma

        self.batch_size = batch_size
        self.noise = noise

    def get_action(self, state, with_noise=True):
        state = torch.tensor(state, dtype=torch.float)
        mu_value = self.Q.mu(state).detach().data.numpy() * self.action_max
        noise = self.noise.noise() * with_noise
        action = mu_value + noise
        return np.clip(action, - self.action_max, self.action_max)

    def state_dict(self):
        return self.Q_target.state_dict()

    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * original_param.data)

    def get_batch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = map(np.array, zip(*minibatch))
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.from_numpy(dones.astype(np.float32))
        next_states = torch.tensor(next_states, dtype=torch.float32)
        return states, actions, rewards, dones, next_states

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            states, actions, rewards, dones, next_states = self.get_batch()
            self.opt.zero_grad()
            target = (1 / 3) * (v)
            loss = self.loss(self.Q(states, actions), target)
            loss.backward()
            self.opt.step()
            self.update_targets(self.Q_target, self.Q)
