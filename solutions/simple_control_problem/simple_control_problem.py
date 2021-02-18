import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.naf import NAFAgent
from problems.simple_control_problem.simple_control_problem_env import SimpleControlProblem
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

env = SimpleControlProblem()
state_shape = 2
action_shape = 1
action_max = 1
episodes_n = 100
epsilon_min = 0.00001
epsilon = 1

mu_model = Seq_Network([state_shape, 100, 50, action_shape], nn.ReLU(), nn.Tanh())
p_model = Seq_Network([state_shape, 100, 50, action_shape ** 2], nn.ReLU())
v_model = Seq_Network([state_shape, 100, 50, 1], nn.ReLU())
noise = OUNoise(action_shape, threshold=epsilon, threshold_min=epsilon_min,
                threshold_decrease=(epsilon_min / epsilon) ** (1 / episodes_n))
batch_size = 100
agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size, 0.999)


def play_and_learn(env):
    total_reward = 0
    state = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        agent.fit(state, action, -reward, done, next_state)
        state = next_state
        step += 1
    _, x = env.state
    agent.noise.decrease()
    return total_reward, x


def agent_play(env, agent):
    state = env.reset()
    total_reward = 0
    ts = []
    us = []
    done = False
    while not done:
        ts.append(state[0])
        us.append(state[1])
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    plt.plot(ts, us)
    return total_reward


rewards = np.zeros(episodes_n)
mean_rewards = np.zeros(episodes_n)
for episode in range(episodes_n):
    reward, x = play_and_learn(env)
    rewards[episode] = reward
    mean_reward = np.mean(rewards[max(0, episode - 10):episode + 1])
    mean_rewards[episode] = mean_reward
    print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f, x0=%.5f" % (
        episode, agent.noise.threshold, rewards[episode], mean_reward, x))

agent.noise.threshold = 0
reward = agent_play(env, agent)
plt.title('track')
plt.legend(['NAF'])
plt.show()
plt.plot(range(episodes_n), mean_rewards)
plt.title('Динамика показателя качества в процессе обучения')
plt.legend(['NAF'])
plt.xlabel('Эпизод')
plt.ylabel('Показатель качества')
plt.show()
plt.show()
torch.save(agent.Q.state_dict(), './test/naf')
np.save('naf_fit', mean_rewards)
