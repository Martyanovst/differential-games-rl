import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from models.linear_transformations import transform_interval


class CentralizedNAF:
    def __init__(self, u_action_dim, v_action_dim, q_model, noise,
                 batch_size=128, gamma=1, tau=1e-3, q_model_lr=1e-3):
        self.u_action_min = u_action_dim[0]
        self.u_action_max = u_action_dim[1]
        self.v_action_min = v_action_dim[0]
        self.v_action_max = v_action_dim[1]
        self.q_model = q_model
        self.opt = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)
        self.loss = nn.MSELoss()
        self.lr = q_model_lr
        self.q_target = deepcopy(self.q_model)
        self.tau = tau
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.batch_size = batch_size
        self.u_noise = noise
        self.v_noise = noise

    def save(self, path):
        torch.save({
            'q-model': self.q_model.state_dict(),
            'u_noise': self.u_noise.state_dict(),
            'v_noise': self.v_noise.state_dict(),
            'u_action_min': self.u_action_min,
            'u_action_max': self.u_action_max,
            'v_action_min': self.v_action_min,
            'v_action_max': self.v_action_max,
            'tau': self.tau,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size
        }, path)

    def train(self):
        self.noise.threshold = 1

    def eval(self):
        self.noise.threshold = 0

    def get_u_action(self, state):
        state = torch.FloatTensor(state)
        state.requires_grad = True
        mu_value = self.q_model.u_mu_model(state).detach().numpy()
        noise = self.u_noise.noise()
        action = mu_value + noise
        action = transform_interval(action, self.u_action_min, self.u_action_max)
        return np.clip(action, self.u_action_min, self.u_action_max)

    def get_v_action(self, state):
        state = torch.FloatTensor(state)
        state.requires_grad = True
        mu_value = self.q_model.v_mu_model(state).detach().numpy()
        noise = self.v_noise.noise()
        action = mu_value + noise
        action = transform_interval(action, self.v_action_min, self.v_action_max)
        return np.clip(action, self.v_action_min, self.v_action_max)

    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * original_param.data)

    def add_to_memory(self, step):
        self.memory.append(step)

    def fit(self, step):
        self.add_to_memory(step)

        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, u_actions, v_actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            states.requires_grad = True
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)

            target = rewards + (1 - dones) * self.gamma * self.q_target.v_model(next_states).detach()
            q_values = self.q_model(states, u_actions, v_actions)
            loss = self.loss(q_values, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.update_targets(self.q_target, self.q_model)
