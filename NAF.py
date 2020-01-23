import numpy as np
import torch
from torch import nn
import random
import gym
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.autograd import Variable
from nafAgent import DQNAgent

# env = gym.make('Pendulum-v0')
# env = gym.make('MountainCarContinuous-v0')
env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space
agent = DQNAgent(state_dim, action_dim)
episode_n = 200
rewards = []
for episode in range(episode_n):
    state = env.reset()
    total_reward = 0
    loss = 0
    for t in range(10000):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        if episode > 50:
            env.render()
        loss = agent.fit(state, action, reward, done, next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    print(str(episode) + ' : ' + str(total_reward) + ' loss: ' +
          str(loss) + ' memory: ' + str(len(agent.memory)))
    rewards.append(total_reward)

plt.plot(range(episode_n), rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
