import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
from copy import deepcopy
from collections import deque


class P_model(nn.Module):
	def __init__(self, state_shape):
		super().__init__()
		self.dense_1 = nn.Sequential(nn.Linear(state_shape[0], 32), nn.ReLU())
		self.dense_2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
		self.dense_3 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
		self.dense_4 = nn.Linear(32, 1)

	def forward(self, state):
		hidden = self.dense_1(state)
		hidden = self.dense_2(hidden)
		hidden = self.dense_3(hidden)
		P_value = self.dense_4(hidden) ** 2
		return P_value


class nu_model(nn.Module):
	def __init__(self, state_shape):
		super().__init__()
		self.dense_1 = nn.Sequential(nn.Linear(state_shape[0], 16), nn.ReLU())
		self.dense_2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
		self.dense_3 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
		self.dense_4 = nn.Sequential(nn.Linear(16, 1), nn.Tanh())

	def forward(self, state):
		hidden = self.dense_1(state)
		hidden = self.dense_2(hidden)
		hidden = self.dense_3(hidden)
		nu_value = self.dense_4(hidden)
		return nu_value


class V_model(nn.Module):
	def __init__(self, state_shape):
		super().__init__()
		self.dense_1 = nn.Sequential(nn.Linear(state_shape[0], 16), nn.ReLU())
		self.dense_2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
		self.dense_3 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
		self.dense_4 = nn.Linear(16, 1)

	def forward(self, state):
		hidden = self.dense_1(state)
		hidden = self.dense_2(hidden)
		hidden = self.dense_3(hidden)
		V_value = self.dense_4(hidden)
		return V_value


class Q_model(nn.Module):
	def __init__(self, state_shape):
		super().__init__()
		self.P = P_model(state_shape)
		self.nu = nu_model(state_shape)
		self.V = V_model(state_shape)

	def forward(self, state, action):
		return self.V(state) + self.P(state) * (self.nu(state) - action) ** 2


class NAFAgent(nn.Module):
	
	def __init__(self, state_shape, action_shape, action_max):
		super().__init__()
		self.state_shape = state_shape
		self.action_shape = action_shape
		self.action_max = action_max
		
		self.Q = Q_model(state_shape)
		self.opt = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
		self.Q_target = deepcopy(self.Q)
		self.tau = 1e-1
		self.memory = deque(maxlen=100000)
		self.gamma = 0.99
		self.noise_threshold = 1
		self.noise_threshold_decrease = 0.0001
		self.noise_threshold_min = 0.001
		self.batch_size = 256
		self.noise = OUNoise(action_shape[0])
		self.reward_normalize = 0.01
		# self.nu_network = Network([2,16,16,2])

	def get_action(self, state):
		state = torch.tensor(np.array([state]), dtype=torch.float)
		nu_value = self.Q.nu(state).detach().data.numpy()[0]
		noise = self.noise.noise()
		action = self.action_max * nu_value + self.noise_threshold * noise
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
			target = self.reward_normalize * rewards + self.gamma * self.Q.V(next_states).detach()
			loss = torch.mean((self.Q(states, actions) - target) ** 2)
			loss.backward()
			self.opt.step()
			self.update_targets(self.Q_target, self.Q)
			
			if self.noise_threshold > self.noise_threshold_min:
				self.noise_threshold -= self.noise_threshold_decrease

			

class OUNoise:
	"""docstring for OUNoise"""
	def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
		self.action_dimension = action_dimension
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.state = np.ones(self.action_dimension) * self.mu
		self.reset()

	def reset(self):
		self.state = np.ones(self.action_dimension) * self.mu

	def noise(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
		self.state = x + dx
		return self.state
