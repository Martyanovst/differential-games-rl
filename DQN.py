import numpy as np
import torch
from torch import nn
import random
import gym
import matplotlib.pyplot as plt


class Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, 32)
        self.linear_2 = nn.Linear(32, 32)
        self.linear_3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.relu(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu(hidden)
        output = self.linear_3(hidden)
        return self.tanh(output)


class DQNAgent(nn.Module):

    def __init__(self, state_dim, action_space):
        super().__init__()
        self.state_dim = state_dim
        self.action_space = action_space

        self.gamma = 0.95
        self.epsilon = 1
        self.memory_size = 100000
        self.memory = []
        self.batch_size = 256
        self.learinig_rate = 1e-2

        self.q_normalized = Network(self.state_dim, self.action_space.shape[0])
        self.tau = 0.8
        self.q_target = Network(self.state_dim, self.action_space.shape[0])
        self.update_weights(1)
        self.optimazer = torch.optim.Adam(
            self.q_normalized.parameters(), lr=self.learinig_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        argmax_action = torch.argmax(self.q_target(state)).data.numpy()
        action_exploration = np.random.normal(0, 1, 2)
        action = np.clip(argmax_action + action_exploration, -1, 1)
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
            for _ in range(10):
                batch = random.sample(self.memory, self.batch_size)

                states, actions, rewards, dones, next_states = list(
                    zip(*batch))
                states = torch.FloatTensor(states)
                q_values = self.q_normalized(states)
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


env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0]
action_space = env.action_space
agent = DQNAgent(state_dim, action_space)
episode_n = 30
rewards = []
for episode in range(episode_n):
    state = env.reset()
    total_reward = 0
    for t in range(1000):
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
