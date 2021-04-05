import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class Q_model(nn.Module):
    def __init__(self, mu_model, p_model,
                 v_model, action_shape, action_max=1):
        super().__init__()
        self.P = p_model
        self.mu = mu_model
        self.v = v_model
        self.action_max = action_max
        self.action_shape = action_shape
        self.tril_mask = torch.tril(torch.ones(
            action_shape, action_shape), diagonal=-1).unsqueeze(0)
        self.diag_mask = torch.diag(torch.diag(
            torch.ones(action_shape, action_shape))).unsqueeze(0)

    def forward(self, state, action):
        L = self.P(state).view(-1, self.action_shape, self.action_shape)
        L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
        P = torch.bmm(L, L.transpose(2, 1))
        mu = self.mu(state) * self.action_max
        action_mu = (action - mu).unsqueeze(2)
        A = -0.5 * \
            torch.bmm(torch.bmm(action_mu.transpose(2, 1), P),
                      action_mu)[:, :, 0]
        return A + self.v(state)


class NAFAgent:

    def __init__(self, mu_model, p_model, v_model,
                 noise, state_shape, action_shape,
                 action_max, batch_size=200, gamma=0.9999, action_min=None, learning_n_per_fit=8):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_max = action_max
        self.learning_n_per_fit = learning_n_per_fit
        if action_min:
            self.action_min = action_min
        else:
            self.action_min = -action_max
        self.Q = Q_model(mu_model, p_model, v_model, action_shape)
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()
        self.Q_target = deepcopy(self.Q)
        self.tau = 1e-3
        self.memory = deque(maxlen=100000)
        self.gamma = gamma

        self.batch_size = batch_size
        self.noise = noise

    def get_action(self, state, with_noise=True):
        state = torch.tensor(state, dtype=torch.float)
        mu_value = self.Q.mu(state).detach().data.numpy() * self.action_max
        noise = self.noise.noise() * with_noise
        action = mu_value + noise
        return np.clip(action, self.action_min, self.action_max)

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
            for _ in range(self.learning_n_per_fit):
                states, actions, rewards, dones, next_states = self.get_batch()
                self.opt.zero_grad()
                target = rewards.reshape(self.batch_size, 1) + (1 - dones).reshape(self.batch_size,
                                                                                   1) * self.gamma * self.Q_target.v(
                    next_states).detach()
                loss = self.loss(self.Q(states, actions), target)
                loss.backward()
                self.opt.step()
                self.update_targets(self.Q_target, self.Q)
