import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque

from models.linear_transformations import transform_interval


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Qmodel(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear_1 = nn.Linear(state_dim + action_dim, 400)
        self.linear_2 = nn.Linear(400, 300)
        self.linear_3 = nn.Linear(300, 1)
        self.relu = nn.ReLU()
        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.linear_1.weight.data = fanin_init(self.linear_1.weight.data.size())
        self.linear_2.weight.data = fanin_init(self.linear_2.weight.data.size())
        self.linear_3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        hidden = self.linear_1(torch.cat((state, action), dim=1))
        hidden = self.relu(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu(hidden)
        Q_value = self.linear_3(hidden)
        return Q_value


class Numodel(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.linear_1 = nn.Linear(state_dim, 400)
        self.linear_2 = nn.Linear(400, 300)
        self.linear_3 = nn.Linear(300, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.linear_1.weight.data = fanin_init(self.linear_1.weight.data.size())
        self.linear_2.weight.data = fanin_init(self.linear_2.weight.data.size())
        self.linear_3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        hidden = self.linear_1(state)
        hidden = self.relu(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu(hidden)
        hidden = self.linear_3(hidden)
        nu_value = self.tanh(hidden)
        return nu_value

class DDPG:

    def __init__(self, state_dim, action_dim, action_min, action_max, q_model, pi_model, noise,
                 q_model_lr=1e-3, pi_model_lr=1e-4, gamma=0.99, batch_size=64, tau=1e-3,
                 memory_len=6000000, learning_iter_per_fit=1, convex_comb_for_actions=False):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = torch.FloatTensor(action_min)
        self.action_max = torch.FloatTensor(action_max)
        self.q_model = q_model
        self.pi_model = pi_model
        self.noise = noise

        self.q_model_lr = q_model_lr
        self.pi_model_lr = pi_model_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.memory = deque(maxlen=memory_len)
        self.learning_iter_per_fit = learning_iter_per_fit
        self.convex_comb_for_actions = convex_comb_for_actions

        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_model_lr)
        self.q_target_model = deepcopy(self.q_model)
        self.pi_target_model = deepcopy(self.pi_model)
        return None

    def get_action(self, state):
        state = torch.FloatTensor(state)
        if self.convex_comb_for_actions:
            action = (1 - self.noise.threshold) * self.pi_model(state) + torch.FloatTensor(self.noise.get())
        else:
            action = self.pi_model(state) + torch.FloatTensor(self.noise.get())
        action = transform_interval(action, self.action_min, self.action_max)
        return np.clip(action.detach().numpy(), self.action_min.numpy(), self.action_max.numpy())

    def update_target_model(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        return None

    def add_to_memory(self, sessions):
        for session in sessions:
            session_len = len(session['actions'])
            for i in range(session_len):
                self.memory.append([session['states'][i],
                                    session['actions'][i],
                                    session['rewards'][i],
                                    session['dones'][i],
                                    session['states'][i + 1]])

    def fit(self, sessions):
        self.add_to_memory(sessions)

        if len(self.memory) >= self.batch_size:
            for _ in range(self.learning_iter_per_fit):
                batch = random.sample(self.memory, self.batch_size)
                states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
                rewards = rewards.reshape(self.batch_size, 1)
                dones = dones.reshape(self.batch_size, 1)

                pred_next_actions = transform_interval(self.pi_target_model(next_states),
                                                       self.action_min, self.action_max)
                next_states_and_pred_next_actions = torch.cat((next_states, pred_next_actions), dim=1)
                targets = rewards + (1 - dones) * self.gamma * self.q_target_model(next_states_and_pred_next_actions)
                states_and_actions = torch.cat((states, actions), dim=1)
                q_loss = torch.mean((self.q_model(states_and_actions) - targets.detach()) ** 2)
                self.update_target_model(self.q_target_model, self.q_model, self.q_optimizer, q_loss)

                pred_actions = transform_interval(self.pi_model(states),
                                                  self.action_min, self.action_max)
                states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
                pi_loss = - torch.mean(self.q_model(states_and_pred_actions))
                self.update_target_model(self.pi_target_model, self.pi_model, self.pi_optimizer, pi_loss)

        return None