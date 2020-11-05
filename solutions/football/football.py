from copy import deepcopy

import tensorflow as tf
import seaborn as sns
import gym
import gfootball
import gfootball.env as football_env
import sys
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque

sns.set()
# Training keras models in a loop with eager execution on causes memory leaks and terrible performance.
tf.compat.v1.disable_eager_execution()
sys.path.append("/home/ubuntu/football/kaggle-football/")


class Seq_Network(nn.Module):
    def __init__(self, layers, hidden_activation, output_activation=None):
        super().__init__()
        hidden_layers = layers[:-1]
        network = [nn.Sequential(nn.Linear(i, o), hidden_activation) for i, o in
                   zip(hidden_layers, hidden_layers[1:])]
        network.append(nn.Linear(layers[-2], layers[-1]))
        if output_activation:
            network.append(output_activation)
        self.network = nn.Sequential(*network)
        self.apply(self._init_weights_)

    def forward(self, tensor):
        return self.network(tensor)

    def _init_weights_(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


class DQNAgent(nn.Module):

    def __init__(self, state_shape, actions_n):
        super().__init__()

        self.state_shape = state_shape
        self.actions_n = actions_n

        self.action_network = Seq_Network([state_shape[0], 80, 40, actions_n], nn.ReLU())
        self.target_network = deepcopy(self.action_network)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()
        self.memory = deque(maxlen=500000)
        self.gamma = 0.999
        self.tau = 1e-3
        self.epsilon = 1
        self.epsilon_decrease = 0.99
        self.epsilon_min = 0.1
        self.learning_steps_n = 1
        self.batch_size = 128

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(list(range(self.actions_n)))
        else:
            state = torch.tensor(state, dtype=torch.float)
            qvalues = self.action_network(state).data.numpy()
            return np.argmax(qvalues)

    def to_one_hot(self, tensor, n_dims=None):
        tensor = tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else self.actions_n
        # n_dims = n_dims if n_dims is not None else int(torch.max(tensor)) + 1
        one_hot = torch.zeros(tensor.size()[0], n_dims).scatter_(1, tensor, 1)
        return one_hot

    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * original_param.data)

    def get_batch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def eps_decrease(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decrease

    def fit(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, float(done)])

        if len(self.memory) > self.batch_size * self.learning_steps_n:

            for _ in range(self.learning_steps_n):
                states, actions, rewards, next_states, dones = self.get_batch()
                qvalues = self.action_network(states)
                qvalues_for_actions = torch.sum(qvalues * self.to_one_hot(actions), dim=1)
                next_qvalues = self.target_network(next_states)
                target = rewards + self.gamma * (1 - dones) * torch.max(next_qvalues, dim=-1)[0]
                loss = self.loss(qvalues_for_actions, target.detach())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_targets(self.target_network, self.action_network)


def play_and_learn(t_max=3000):
    total_reward = 0
    state = env.reset()
    for t in range(t_max):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state
        total_reward += reward
        agent.fit(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    agent.eps_decrease()
    return total_reward


n_episodes = 100
# (115,)
env = gym.make("GFootball-11_vs_11_kaggle-simple115v2-v0")
# (72,96,4)
# env = football_env.create_environment(env_name="11_vs_11_kaggle", stacked=False, logdir='/tmp/football',
#                                       write_goal_dumps=False, write_full_episode_dumps=False, render=False)
agent = DQNAgent(env.observation_space.shape, env.action_space.n)

for episode in range(n_episodes):
    reward = play_and_learn()
    print("episode=%.0f, epsilon=%.3f, reward=%.3f" % (episode, agent.epsilon, reward))

torch.save(agent.action_network.state_dict(), 'weights')

# For testing agent
# env = gym.make("LunarLander-v2")
# agent = DQNAgent(env.observation_space.shape, env.action_space.n)
#
# for episode in range(n_episodes):
#     reward = play_and_learn()
#     print("episode=%.0f, epsilon=%.3f, reward=%.3f" % (episode, agent.epsilon, reward))
