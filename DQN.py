import numpy as np
import torch
from torch import nn
import random
import gym
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init(self, layers, hidden_activation, output_activation):
        super().__init__()
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layers_count = len(layers)
        self.output_layer = nn.Linear(layers[self.layers_count - 2], self.layers_count - 1])
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
        return hidden


class NAF_Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mu = Network([input_dim, 64, 64, 32, output_dim], nn.Relu(), nn.Tahn())
        self.P = Network([input_dim, 64, 64, 32, int(output_dim * (output_dim + 1) / 2)], nn.Relu(), nn.Relu())
        self.v = Network([input_dim, 32, 16, 1], nn.Relu(), nn.Tahn())        

    def forward(self, tensor):
        mu = self.mu(tensor)
        L = self.P(tensor)
        v = self.v(tensor)
        return mu, L, v

    def Q_value(self, tensor, action):
        mu, L, v = self.forward(tensor)
        P = L @ L.T
        A = -1/2 * (action - mu).T @ P @ (action - mu)
        return A + v
    
    def maximum_q_value(self, tensor):
        return self.v(tensor)

    def argmax_action(self, tensor):
        return self.mu(tensor)
    
    def Q_value(self, state, action):
        mu, P, V = self.forward(state)
        A = - 1/2 * (action - mu).T * P * (action - mu) 
        return A + V


class DQNAgent(nn.Module):

    def __init__(self, state_dim, action_space):
        super().__init__()
        self.state_dim = state_dim
        self.action_space = action_space

        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_desc = 0.99
        self.epsilon_min = 0.1
        self.memory_size = 200000
        self.memory = []
        self.batch_size = 512
        self.learinig_rate = 1e-2

        self.q_normalized = Network(self.state_dim, self.action_space.shape[0])
        self.tau = 0.8
        self.q_target = Network(self.state_dim, self.action_space.shape[0])
        self.update_weights(1)
        self.optimazer = torch.optim.Adam(
            self.q_normalized.parameters(), lr=self.learinig_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        mu, P, V = self.q_target(state)

        # argmax_action = torch.argmax().data.numpy()
        action_exploration = np.random.normal(0, 0.25 * self.epsilon, 2)
        action = np.clip(mu.data.numpy() + action_exploration, -1, 1)
        return action

    def update_weights(self, tau):
        for new_parameter, old_parameter in zip(self.q_target.parameters(), self.q_normalized.parameters()):
            new_parameter.data.copy_(
                (tau) * old_parameter + (1 - tau) * new_parameter)

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        if len(self.memory) > self.batch_size:
            for _ in range(5):
                batch = random.sample(self.memory, self.batch_size)

                states, actions, rewards, dones, next_states = list(
                    zip(*batch))
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                tensor = zip(states, actions)
                
                q_values = list(map(lambda inpt: self.q_normalized.Q_value(inpt[0], inpt[1]), tensor))
                next_states = torch.FloatTensor(next_states)
                next_q_values = self.q_target(next_states)
                targets = q_values.clone()
                for i in range(self.batch_size):
                    targets[i][actions[i]] = rewards[i] + self.gamma * \
                        (1 - dones[i]) * max(next_q_values[i])
                loss = torch.mean((targets.detach() - q_values) ** 2)

                loss.backward()
                self.optimazer.step()
                self.optimazer.zero_grad()
                self.update_weights(self.tau)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_desc


env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0]
action_space = env.action_space
agent = DQNAgent(state_dim, action_space)
episode_n = 300
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
