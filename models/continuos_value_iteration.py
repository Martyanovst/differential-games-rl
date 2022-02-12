import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from models.linear_transformations import transform_interval
from models.q_models import CVI_VModel


class CVI:
    def __init__(self, action_min, action_max, v_model, noise,
                 batch_size=128, gamma=1, tau=1e-3, v_model_lr=1e-3):
        self.action_max = action_max
        self.action_min = action_min
        self.v_model = v_model
        self.opt = torch.optim.Adam(self.v_model.parameters(), lr=v_model_lr)
        self.loss = nn.MSELoss()
        self.lr = v_model_lr
        self.v_target = deepcopy(self.v_model)
        self.tau = tau
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise = noise
        self.learning_n_per_fit = 16

    def save(self, path):
        torch.save({
            'v-model': self.v_model.state_dict(),
            'noise': self.noise.state_dict(),
            'action_min': self.action_min,
            'action_max': self.action_max,
            'tau': self.tau,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size
        }, path)

    def train(self):
        self.noise.threshold = 1
        self.memory.clear()
        self.v_model.train()

    def eval(self):
        self.noise.threshold = 0
        self.v_model.eval()

    def get_action(self, state):
        state = torch.FloatTensor(state)
        state.requires_grad = True
        mu_value = self.v_model(state).detach().numpy()
        noise = self.noise.noise()
        action = mu_value + noise
        action = transform_interval(action, self.action_min, self.action_max)
        return np.clip(action, self.action_min, self.action_max)

    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * original_param.data)

    def add_to_memory(self, step):
        self.memory.append(step)

    def fit(self, step):
        self.add_to_memory(step)

        if len(self.memory) >= self.batch_size:
            batch = list(zip(*random.sample(self.memory, self.batch_size)))
            states = torch.FloatTensor(np.array(batch[0]))
            actions = torch.FloatTensor(np.array(batch[1]))
            rewards = torch.FloatTensor(np.array(batch[2]))
            dones = torch.FloatTensor(np.array(batch[3]))
            next_states = torch.FloatTensor(np.array(batch[4]))
            states.requires_grad = True
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)

            target = rewards + (1 - dones) * self.gamma * \
                self.v_target(next_states).detach()
            v_values = self.v_model(states)
            loss = self.loss(v_values, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.update_targets(self.v_target, self.v_model)
