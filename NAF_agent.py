import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
from copy import deepcopy
from collections import deque
from noises import UniformNoise


class P_model(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.dense_1 = nn.Sequential(nn.Linear(state_shape[0], 100), nn.ReLU())
        self.dense_2 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
        self.dense_3 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
        
        self.dense_4 = nn.Linear(100, action_shape ** 2)

        self.action_shape = action_shape
        self.tril_mask = torch.tril(torch.ones(
            action_shape, action_shape), diagonal=-1).unsqueeze(0)

        self.diag_mask = torch.diag(torch.diag(
            torch.ones(action_shape, action_shape))).unsqueeze(0)

    def forward(self, state):
        hidden = self.dense_1(state)
        hidden = self.dense_1(state)
        hidden = self.dense_2(hidden)
        hidden = self.dense_3(hidden)
        L_vec = self.dense_4(hidden)
        L = L_vec.view(-1, self.action_shape, self.action_shape)
        L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
        P = torch.bmm(L, L.transpose(2, 1))
        return P

class nu_model(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.dense_1 = nn.Sequential(nn.Linear(state_shape[0], 50), nn.ReLU())
        self.dense_2 = nn.Sequential(nn.Linear(50, 50), nn.ReLU())
        self.dense_3 = nn.Sequential(nn.Linear(50, 50), nn.ReLU())
        self.dense_4 = nn.Sequential(nn.Linear(50, 2), nn.Tanh())

    def forward(self, state):
        hidden = self.dense_1(state)
        hidden = self.dense_1(state)
        hidden = self.dense_2(hidden)
        hidden = self.dense_3(hidden)

        nu_value = self.dense_4(hidden)
        return nu_value


class V_model(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.dense_1 = nn.Sequential(nn.Linear(state_shape[0], 50), nn.ReLU())
        self.dense_2 = nn.Sequential(nn.Linear(50, 50), nn.ReLU())
        self.dense_3 = nn.Sequential(nn.Linear(50, 50), nn.ReLU())
        self.dense_4 = nn.Linear(50, 1)

    def forward(self, state):
        hidden = self.dense_1(state)
        hidden = self.dense_1(state)
        hidden = self.dense_2(hidden)
        hidden = self.dense_3(hidden)
        V_value = self.dense_4(hidden)
        return V_value


class Q_model(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.P = P_model(state_shape, action_shape)
        self.nu = nu_model(state_shape)
        self.V = V_model(state_shape)

    def forward(self, state, action):
        action_mu = (action - self.nu(state)).unsqueeze(2)
        A = -0.5 * \
            torch.bmm(torch.bmm(action_mu.transpose(2, 1), self.P(state)),
                      action_mu)[:, :, 0]
        # --------------------------------
        return A + self.V(state)

class NAFAgent():
    
    def __init__(self, state_shape, action_shape, action_max):

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_max = action_max
        
        self.Q = Q_model(state_shape, action_shape[0])
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        self.Q_target = deepcopy(self.Q)
        self.tau = 1e-3
        self.memory = deque(maxlen=200000)
        self.gamma = 0.99
        self.noise_threshold = 1
        self.noise_threshold_decrease = 0.000001
        self.noise_threshold_min = 0.001
        self.batch_size = 200
        # self.noise = UniformNoise(action_shape[0])
        self.noise = OUNoise(action_shape[0])
        self.reward_normalize = 1
    
    def get_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float)
        #state[:, 2] /= 8
        nu_value = self.Q.nu(state).detach().data.numpy()[0]
        noise = self.noise.noise()
        action = self.action_max * (nu_value + self.noise_threshold * noise)
        return np.clip(action, - self.action_max, self.action_max)
    
    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * original_param.data)
    
    def get_batch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = map(np.array, zip(*minibatch))
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        return states, actions, rewards, next_states
    
    def fit(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])
        
        if len(self.memory) >= self.batch_size:
            states, actions, rewards, next_states = self.get_batch()
            self.opt.zero_grad()
            target = self.reward_normalize * rewards.reshape(self.batch_size, 1) + self.gamma * self.Q_target.V(next_states).detach()
            loss = torch.mean((self.Q(states, actions) - target) ** 2)
            loss.backward()
            self.opt.step()
            self.update_targets(self.Q_target, self.Q)
            
            # self.noise.decrease()

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, threshold=1, threshold_min=0.01, threshold_decrease=0.000002):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.threshold = threshold
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def decrease(self):
        pass
        if self.threshold > self.threshold_min:
            self.threshold -= self.threshold_decrease   


#