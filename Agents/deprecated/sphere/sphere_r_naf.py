import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class Q_model(nn.Module):
    def __init__(self, phi_model,
                 v_model, action_shape, beta, dt, action_max=1):
        super().__init__()
        self.v = v_model
        self.phi = phi_model
        self.beta = beta
        self.dt = dt
        self.action_max = action_max
        self.action_shape = action_shape
        self.tril_mask = torch.tril(torch.ones(
            action_shape, action_shape), diagonal=-1).unsqueeze(0)
        self.diag_mask = torch.diag(torch.diag(
            torch.ones(action_shape, action_shape))).unsqueeze(0)

    def forward(self, state, action):
        phi = self.phi(state)
        phi_clipped = nn.functional.tanh(phi)
        action_phi = (action - phi_clipped).unsqueeze(2)
        phi_delta = (phi - phi_clipped).unsqueeze(2)

        A = -self.dt * self.beta * \
            torch.bmm(action_phi.transpose(2, 1),
                      action_phi)[:, :, 0] + \
            2 * self.dt * self.beta * torch.bmm(phi_delta.transpose(2, 1),
                                                action_phi)[:, :, 0]
        return A + self.v(state)


class Sphere_R_NAF:

    def __init__(self, phi_model, v_model,
                 noise, state_shape, action_shape, beta, dt,
                 action_max, batch_size=200, gamma=0.9999):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_max = action_max

        self.Q = Q_model(phi_model, v_model, action_shape, beta, dt, action_max)
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=1e-4)
        self.loss = nn.MSELoss()
        self.Q_target = deepcopy(self.Q)
        self.tau = 1e-3
        self.memory = deque(maxlen=100000)
        self.gamma = gamma

        self.batch_size = batch_size
        self.noise = noise

    def get_action(self, state, with_noise=True):
        state = torch.tensor(state, dtype=torch.float)
        mu_value = nn.functional.tanh(self.Q.phi(state)).detach().data.numpy() * self.action_max
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
            target = rewards.reshape(self.batch_size, 1) + (
                    1 - dones).reshape(self.batch_size, 1) * self.gamma * self.Q_target.v(
                next_states).detach()
            loss = self.loss(self.Q(states, actions), target)
            loss.backward()
            self.opt.step()
            self.update_targets(self.Q_target, self.Q)
