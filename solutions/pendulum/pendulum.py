import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn

from models.naf import NAFAgent
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

episodes_n = 200
state_shape = 3
action_shape = 1
action_max = 2


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


def play_and_learn(env, agent, delta):
    total_reward = 0
    state = env.reset()
    step = 0
    while step < 200:
        th, thdot = env.state
        action = agent.get_action(state)
        next_state, _, _, _ = env.step(action)
        u = np.clip(action, -action_max, action_max)[0]
        reward = angle_normalize(th) ** 2 + .1 * thdot ** 2
        total_reward += -reward
        reward += delta * (u ** 2)
        agent.fit(state, action, -reward, False, next_state)
        state = next_state
        step += 1
    agent.noise.decrease()
    return total_reward


def play(env, agent):
    total_reward = 0
    state = env.reset()
    step = 0
    while step < 200:
        th, thdot = env.state
        env.render()
        action = agent.get_action(state)
        next_state, _, _, _ = env.step(action)
        u = np.clip(action, -action_max, action_max)[0]
        reward = angle_normalize(th) ** 2 + .1 * thdot ** 2
        total_reward += -reward
        reward += delta * (u ** 2)
        state = next_state
        step += 1
    agent.noise.decrease()
    return total_reward


def fit(env, delta):
    mu_model = Seq_Network([state_shape, 50, 50, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 50, 50, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 50, 50, 1], nn.ReLU())
    noise = OUNoise(action_shape, threshold=1, threshold_min=0.0001, threshold_decrease=0.005)
    batch_size = 128
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size, 0.99)
    rewards = np.zeros(episodes_n)
    mean_rewards = np.zeros(episodes_n)
    for episode in range(episodes_n):
        reward = play_and_learn(env, agent, delta)
        rewards[episode] = reward
        mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
        mean_rewards[episode] = mean_reward
        print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f, delta=%.4f" % (
            episode, agent.noise.threshold, rewards[episode], mean_reward, delta))
    plt.plot(range(episodes_n), mean_rewards)
    # torch.save(agent.Q.state_dict(), './result13')
    # while True:
    #     play(env,agent)


env = gym.make('Pendulum-v0')
deltas = [0, 0.000001, 0.00001, 0.001, 0.01, 1]
# deltas = np.arange(11) * 10
# deltas = [0]
print(deltas)
for delta in deltas:
    fit(env, delta)
plt.legend(deltas)
plt.show()
