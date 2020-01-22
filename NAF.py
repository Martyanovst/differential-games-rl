import numpy as np
import torch
from torch import nn
import random
import gym
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.autograd import Variable

class LinearNetwork(nn.Module):
    def __init__(self, layers, hidden_activation, output_activation):
        super().__init__()
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layers_count = len(layers)
        self.output_layer = nn.Linear(
            layers[self.layers_count - 2], layers[self.layers_count - 1])
        self.init_hidden_layers_(layers)

    def init_hidden_layers_(self, layers):
        self.hidden_layers = []
        for i in range(1, len(layers) - 1):
            previous_layer = layers[i - 1]
            current_layear = layers[i]
            linear = nn.Linear(previous_layer, current_layear)
            self.hidden_layers.append(linear)

    def forward(self, tensor):
        hidden = tensor
        for layer in self.hidden_layers:
            hidden = self.hidden_activation(layer(hidden))
        output = self.output_activation(self.output_layer(hidden))
        return output


class NAF_Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.mu = LinearNetwork(layers=[input_dim, 64, 64, 32, output_dim],
                                hidden_activation=nn.ReLU(),
                                output_activation=nn.Tanh())
        self.P = LinearNetwork(layers=[input_dim, 64, 64, 32, output_dim ** 2],
            hidden_activation=nn.ReLU(),
            output_activation=nn.ReLU())
        self.v = LinearNetwork(layers=[input_dim, 32, 16, 1], hidden_activation=nn.ReLU(
        ), output_activation=nn.ReLU())

        self.tril_mask = Variable(torch.tril(torch.ones(
            output_dim, output_dim), diagonal=-1).unsqueeze(0))

        self.diag_mask = Variable(torch.diag(torch.diag(
            torch.ones(output_dim, output_dim))).unsqueeze(0))

    def _forward_(self, tensor):
        mu = self.mu(tensor)
        L = self.P(tensor)
        v = self.v(tensor)
        return mu, L, v

    def forward(self, tensor, action):
        mu, L_vec, v = self._forward_(tensor)
        # --------------------------------
        L = L_vec.view(-1, self.output_dim, self.output_dim)
        L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
        P = torch.bmm(L, L.transpose(2, 1))
        action_mu = (action - mu).unsqueeze(2)
        A = -0.5 * torch.bmm(torch.bmm(action_mu.transpose(2, 1), P), action_mu)[:, :, 0]
        # --------------------------------
        return A + v

    def maximum_q_value(self, tensor):
        return self.v(tensor)

    def argmax_action(self, tensor):
        return self.mu(tensor)


class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, threshold=1, threshold_min=0.001, threshold_decrease=0.0001):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.threshold = threshold
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def decrease(self):
        if self.threshold > self.threshold_min:
            self.threshold -= self.threshold_decrease


class DQNAgent(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.memory_size = 200000
        self.memory = []
        self.batch_size = 512
        self.learinig_rate = 1e-2
        self.tau = 1e-1
        self.reward_normalize = 0.01

        self.action_exploration = OUNoise(action_dim.shape[0])
        self.init_naf_networks()

    def init_naf_networks(self):
        self.Q = NAF_Network(self.state_dim, self.action_dim.shape[0])
        self.q_target = deepcopy(self.Q)
        self.opt = torch.optim.Adam(
            self.Q.parameters(), lr=self.learinig_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action = self.q_target.argmax_action(state).detach().data.numpy()
        action_exploration = self.action_exploration.noise()
        return np.clip(action + action_exploration, -1, 1)

    def soft_update(self, tau):
        for new_parameter, old_parameter in zip(self.q_target.parameters(), self.Q.parameters()):
            new_parameter.data.copy_(
                (tau) * old_parameter + (1 - tau) * new_parameter)

    def get_batch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = map(
            np.array, zip(*minibatch))
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        return states, actions, rewards, dones, next_states

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            states, actions, rewards, dones, next_states = self.get_batch()
            self.opt.zero_grad()
            target = self.reward_normalize * rewards + self.gamma * \
                (1 - dones) * self.Q.maximum_q_value(next_states).detach()
            loss = torch.mean((self.Q(states, actions) - target) ** 2)
            loss.backward()
            self.opt.step()
            self.soft_update(self.tau)

            self.action_exploration.decrease()


env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space
agent = DQNAgent(state_dim, action_dim)
episode_n = 500
rewards = []
for episode in range(episode_n):
    state = env.reset()
    total_reward = 0
    for t in range(10000):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        agent.fit(state, action, reward, done, next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    print(str(episode) + ' : ' + str(total_reward))
    rewards.append(total_reward)

plt.plot(range(episode_n), rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
