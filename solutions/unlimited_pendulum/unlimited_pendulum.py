import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from models.double_naf import DoubleNAFAgent
from models.unlimited_naf import UnlimitedNAFAgent
from problems.unlimited_pendulum.unlimited_pendulum_env import UnlimitedPendulumEnv
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

episodes_n = 500
state_shape = 3
action_shape = 1
action_max = 2


def play_and_learn(env, agent):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        agent.fit(state, action, reward, False, next_state)
        state = next_state
    agent.noise.decrease()
    return total_reward


def play(env, agent):
    total_reward = 0
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward


def fit(env):
    mu_model = Seq_Network([state_shape, 50, 50, action_shape], nn.ReLU())
    v_model = Seq_Network([state_shape, 50, 50, 1], nn.ReLU())
    noise = OUNoise(action_shape, threshold=1, threshold_min=0.0001, threshold_decrease=1 / episodes_n)
    batch_size = 128
    agent = UnlimitedNAFAgent(mu_model, v_model, noise, state_shape, action_shape, batch_size, 0.99)
    rewards = np.zeros(episodes_n)
    mean_rewards = np.zeros(episodes_n)
    for episode in range(episodes_n):
        reward = play_and_learn(env, agent)
        rewards[episode] = reward
        mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
        mean_rewards[episode] = mean_reward
        print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f" % (
            episode, agent.noise.threshold, rewards[episode], mean_reward))
    plt.plot(range(episodes_n), mean_rewards)


env = UnlimitedPendulumEnv()
fit(env)
plt.show()
