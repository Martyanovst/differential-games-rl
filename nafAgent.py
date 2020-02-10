import numpy as np
import torch
from torch import nn
import random
import gym
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.autograd import Variable
from noises import UniformNoise, OUNoise, ZeroNoise
from linearNetwork import LinearNetwork, Identical

class NAF_Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        middleware_out_dim = 32
        self.middleware = LinearNetwork(layers=[input_dim, 100,  100, middleware_out_dim],
                                hidden_activation=nn.Sigmoid(),
                                output_activation=nn.Sigmoid())
        self.mu = LinearNetwork(layers=[middleware_out_dim, 50,  50, output_dim],
                                hidden_activation=nn.ReLU(),
                                output_activation=nn.Tanh())
        self.P = LinearNetwork(layers=[middleware_out_dim,  100, 100, output_dim ** 2],
                               hidden_activation=nn.ReLU(),
                               output_activation=Identical())
        self.v = LinearNetwork(layers=[middleware_out_dim, 32, 32, 1],
                               hidden_activation=nn.ReLU(), output_activation=Identical())

        self.tril_mask = torch.tril(torch.ones(
            output_dim, output_dim), diagonal=-1).unsqueeze(0)

        self.diag_mask = torch.diag(torch.diag(
            torch.ones(output_dim, output_dim))).unsqueeze(0)

    def _forward_(self, tensor):
        tensor = self.middleware(tensor)
        mu = self.mu(tensor)
        L = self.P(tensor)
        v = self.v(tensor)
        return mu * 2, L, v

    def forward(self, tensor, action):
        mu, L_vec, v = self._forward_(tensor)
        # --------------------------------
        L = L_vec.view(-1, self.output_dim, self.output_dim)
        L = L * self.tril_mask.expand_as(L) + \
            torch.exp(L) * self.diag_mask.expand_as(L)
        P = torch.bmm(L, L.transpose(2, 1))
        action_mu = (action - mu).unsqueeze(2)
        A = -0.5 * \
            torch.bmm(torch.bmm(action_mu.transpose(2, 1), P),
                      action_mu)[:, :, 0]
        # --------------------------------
        return A + v

    def maximum_q_value(self, tensor):
        tensor = self.middleware(tensor)
        return self.v(tensor)

    def argmax_action(self, tensor):
        tensor = self.middleware(tensor)
        return 2 * self.mu(tensor)

class DQNAgent(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = action_dim.high[0]
        self.action_min =  action_dim.low[0]
        self.gamma = 0.99
        self.memory_size = 20000
        self.memory = []
        self.batch_size = 200
        self.learning_rate = 1e-3
        self.tau = 1e-2
        self.reward_normalize = 1
        self.loss = nn.MSELoss()

        self.action_exploration = OUNoise(action_dim.shape[0])
        # self.action_exploration = ZeroNoise()
        # self.action_exploration = UniformNoise(action_dim.shape[0], threshold=self.action_max)
        self.init_naf_networks()

    def init_naf_networks(self):
        self.Q = NAF_Network(self.state_dim, self.action_dim.shape[0])
        self.q_target = NAF_Network(self.state_dim, self.action_dim.shape[0])
        self.soft_update(1)
        self.opt = torch.optim.Adam(
            self.Q.parameters(), lr=self.learning_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action = self.Q.argmax_action(state).detach().data.numpy()
        action_exploration = self.action_exploration.noise()
        return np.clip(action + action_exploration, self.action_min, self.action_max)

    def soft_update(self, tau):
        for new_parameter, old_parameter in zip(self.q_target.parameters(), self.Q.parameters()):
            new_parameter.data.copy_(
                (tau) * old_parameter.data + (1 - tau) * new_parameter.data)

    def get_batch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = map(
            np.array, zip(*minibatch))
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones.astype(np.float32), dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        return states, actions, rewards, dones, next_states

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        if len(self.memory) >= self.batch_size:
            states, actions, rewards, dones, next_states = self.get_batch()
            target = self.reward_normalize * rewards + (self.gamma * \
                (1 - dones) * self.q_target.maximum_q_value(next_states).detach())
            loss = self.loss(self.Q(states, actions), target)
            self.opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1)
            self.opt.step()
            self.soft_update(self.tau)

            self.action_exploration.decrease()
            return float(loss), self.action_exploration.threshold
        return None, self.action_exploration.threshold