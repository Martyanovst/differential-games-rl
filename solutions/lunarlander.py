import numpy as np
import matplotlib.pyplot as plt
import gym
import torch.nn as nn
from utilities.sequentialNetwork import Seq_Network
from models.naf import NAFAgent

env = gym.make("LunarLanderContinuous-v2").env
state_shape = env.observation_space.shape
action_shape = env.action_space.shape
action_max = 1

mu_model = Seq_Network([state_shape[0], 50, 50, 50, action_shape[0]], nn.ReLU(), nn.Tanh())
p_model = Seq_Network([state_shape[0], 100, 100, 100, action_shape[0] ** 2], nn.ReLU())
v_model = Seq_Network([state_shape[0], 50, 50, 50, 1], nn.ReLU())
agent = NAFAgent(mu_model, p_model, v_model, state_shape, action_shape[0], action_max)


def play_and_learn(t_max=200, learn=True):
    total_reward = 0
    state = env.reset()

    for t in range(t_max):

        action = np.nan_to_num(agent.get_action(state))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if learn:
            agent.fit(state, action, reward, done, next_state)
        env.render()

        state = next_state

        if done:
            break

    return total_reward


episodes_n = 1100
rewards = np.zeros(episodes_n)
mean_rewards = np.zeros(episodes_n)

for episode in range(episodes_n):

    if episode < 1000:
        rewards[episode] = play_and_learn()
    else:
        agent.noise_threshold = 0
        rewards[episode] = play_and_learn(learn=False)
    mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
    mean_rewards[episode] = mean_reward
    print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f" % (
        episode, agent.noise.threshold, rewards[episode], mean_reward))

plt.plot(range(episodes_n), mean_rewards)
plt.show()
