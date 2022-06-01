import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from models.linear_transformations import transform_interval


class CVI:
    def __init__(self, action_min, action_max, v_model, virtual_step, noise,
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
        self.virtual_step = virtual_step
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
        mu_value = self.v_target.get_max_value(state).detach().numpy()
        noise = self.noise.noise()
        action = mu_value + noise
        return action

    def get_action_without_noise(self, state):
        mu_value = self.v_model.batch_max_value(state).detach().numpy()
        return mu_value

    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * original_param.data)

    def add_to_memory(self, state):
        self.memory.append([state])

    def fit(self, step):
        state, _, _, done, _ = step
        if not done:
            self.add_to_memory(state)

        if len(self.memory) >= self.batch_size:
            batch = list(zip(*random.sample(self.memory, self.batch_size)))
            init_states = np.array(batch[0])
            states_numpy = init_states.copy()
            states = torch.FloatTensor(states_numpy)
            states.requires_grad = True
            rewards = np.zeros(states.shape[0])
            for i in range(4):
                states = torch.FloatTensor(states_numpy)
                states.requires_grad = True
                actions = self.get_action_without_noise(states)
                next_states, rewards_step, dones, _ = self.virtual_step(states_numpy.copy(), actions)
                rewards += rewards_step
                states_numpy = next_states
            init_states = torch.FloatTensor(init_states)
            next_states = torch.FloatTensor(next_states)
            rewards = torch.FloatTensor(rewards).reshape(self.batch_size, 1)
            dones = torch.FloatTensor(dones).reshape(self.batch_size, 1)
            target = rewards + (1 - dones) * self.gamma * \
                     self.v_target(next_states).detach()
            v_values = self.v_model(init_states)
            loss = self.loss(v_values, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.update_targets(self.v_target, self.v_model)
