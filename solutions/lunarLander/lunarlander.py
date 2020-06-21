import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import torch
import torch.nn as nn

from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network
from models.naf import NAFAgent

env_to_wrap = gym.make("LunarLanderContinuous-v2").env
env = wrappers.Monitor(env_to_wrap, './videos/' + 'lunarLander' + '/', force=True)
state_shape = env.observation_space.shape
action_shape = env.action_space.shape
action_max = 1

mu_model = Seq_Network([state_shape[0], 50, 50, 50, action_shape[0]], nn.ReLU(), nn.Tanh())
p_model = Seq_Network([state_shape[0], 100, 100, 100, action_shape[0] ** 2], nn.ReLU())
v_model = Seq_Network([state_shape[0], 50, 50, 50, 1], nn.ReLU())
noise = OUNoise(action_shape, threshold=1, threshold_min=0.001, threshold_decrease=0.000005)
batch_size = 200
agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape[0], action_max, batch_size, 0.99)


# agent.Q.load_state_dict(torch.load('./result'))

def play_and_learn(env, learn=True):
    total_reward = 0
    state = env.reset()
    done = False
    step = 0
    while not done:
        action = np.nan_to_num(agent.get_action(state))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        if learn:
            agent.fit(state, action, reward, done, next_state)
            agent.noise.decrease()
        state = next_state
        step += 1
        if step > 500 and learn:
            break
    return total_reward


episodes_n = 600
episodes_to_save = [1, 50, 100, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
rewards = np.zeros(episodes_n)
mean_rewards = np.zeros(episodes_n)

for episode in range(episodes_n):
    if episode in episodes_to_save:
        threshold = agent.noise.threshold
        agent.noise.threshold = 0
        rewards[episode] = play_and_learn(env, learn=False)
        agent.noise.threshold = threshold
    else:
        rewards[episode] = play_and_learn(env_to_wrap)
    mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
    mean_rewards[episode] = mean_reward
    print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f" % (
        episode, agent.noise.threshold, rewards[episode], mean_reward))

plt.plot(range(episodes_n), mean_rewards)
plt.show()
play_and_learn(env, learn=False)
play_and_learn(env, learn=False)
play_and_learn(env, learn=False)
play_and_learn(env, learn=False)
play_and_learn(env, learn=False)
env.close()
env_to_wrap.close()
torch.save(agent.state_dict(), './result')
