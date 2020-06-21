import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class QModel(nn.Module):
    def __init__(self, mu_model, p_model, action_shape, action_max=1):
        super().__init__()
        self.P = p_model
        self.mu = mu_model
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
        A = torch.bmm(torch.bmm(action_mu.transpose(2, 1), P),
                      action_mu)[:, :, 0]
        return 1 / 2 * A


class CentralizedQModel(nn.Module):
    def __init__(self, u_model, v_model, v_network):
        super().__init__()
        self.u_model = u_model
        self.v_model = v_model
        self.v = v_network

    def forward(self, state, u_action, v_action):
        return self.v(state) + self.u_model(state, u_action) - self.v_model(state, v_action)


class CentralizedNafAgent:

    def __init__(self, u_model, v_model, value_model, noise, state_shape, action_shape, u_max, v_max, batch_size=200):

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.u_max = u_max
        self.v_max = v_max

        self.Q = CentralizedQModel(u_model, v_model, value_model)
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        self.Q_target = deepcopy(self.Q)
        self.tau = 1e-2
        self.memory = deque(maxlen=20000)
        self.gamma = 0.9999
        self.batch_size = batch_size
        self.noise = noise
        self.reward_normalize = 1

    def get_action(self, state):
        return self.get_u_action(state)

    def get_u_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        mu_value = self.Q.u_model.mu(state).detach().data.numpy() * self.u_max
        noise = self.noise.noise()
        action = mu_value + noise
        return np.clip(action, - self.u_max, self.u_max)

    def get_v_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        mu_value = self.Q.v_model.mu(state).detach().data.numpy() * self.v_max
        noise = self.noise.noise()
        action = mu_value + noise
        return np.clip(action, - self.v_max, self.v_max)

    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * original_param.data)

    def get_batch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, u_actions, v_actions, rewards, dones, next_states = map(np.array, zip(*minibatch))
        states = torch.tensor(states, dtype=torch.float32)
        u_actions = torch.tensor(u_actions, dtype=torch.float32)
        v_actions = torch.tensor(v_actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        return states, u_actions, v_actions, rewards, dones, next_states

    def fit(self, state, u_action, v_action, reward, done, next_state):
        self.memory.append([state, u_action, v_action, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            states, u_actions, v_actions, rewards, dones, next_states = self.get_batch()
            self.opt.zero_grad()
            target = self.reward_normalize * rewards.reshape(self.batch_size, 1) + (
                    1 - dones) * self.gamma * self.Q_target.v(
                next_states).detach()
            loss = torch.mean((self.Q(states, u_actions, v_actions) - target) ** 2)
            loss.backward()
            self.opt.step()
            self.update_targets(self.Q_target, self.Q)
