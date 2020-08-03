import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.simple_naf import SimpleNaf
from models.unlimited_naf import UnlimitedNAFAgent
from problems.nonlinear_problem.nonlinear_problem_env import NonlinearProblem
from problems.nonlinear_problem.optimal_agent import OptimalAgent
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

env = NonlinearProblem()
state_shape = 2
action_shape = 1
episodes_n = 500

mu_model = Seq_Network([state_shape, 100, 100, 100, action_shape], nn.Sigmoid())
p_model = Seq_Network([state_shape, 100, 100, 100, action_shape ** 2], nn.Sigmoid())
v_model = Seq_Network([state_shape, 100, 100, 100, 1], nn.Sigmoid())
noise = OUNoise(action_shape, threshold=1, threshold_min=0.001, threshold_decrease=0.002)
batch_size = 200
agent = SimpleNaf(mu_model, v_model, noise, state_shape, action_shape, batch_size, 1, env.dt)


def play_and_learn(env):
    total_reward = 0
    state = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        done = step >= 500 and total_reward >= env.optimal_v
        agent.fit(state, action, -reward, done, next_state)
        state = next_state
        step += 1
    x1, x2 = env.state
    t = env.t
    agent.noise.decrease()
    return total_reward, t, x1, x2


def agent_play(env, agent):
    state = env.reset()
    total_reward = 0
    ts = []
    us = []
    terminal_time = 20
    done = False
    step = 0
    while not done:
        action = agent.get_action(state)
        ts.append(env.t)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        us.append(action[0])
        state = next_state
        done = env.t >= terminal_time
        step += 1
    plt.plot(ts, us)
    return total_reward


rewards = np.zeros(episodes_n)
mean_rewards = np.zeros(episodes_n)
times = np.zeros(episodes_n)
mean_times = np.zeros(episodes_n)
for episode in range(episodes_n):
    reward, t, x1, x2 = play_and_learn(env)
    rewards[episode] = reward
    times[episode] = t
    mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
    mean_time = np.mean(times[max(0, episode - 50):episode + 1])
    mean_rewards[episode] = mean_reward
    mean_times[episode] = mean_time
    print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f, x1=%.3f, x2=%.3f, t=%.3f" % (
        episode, agent.noise.threshold, rewards[episode], mean_reward, x1, x2, mean_time))

agent.noise.threshold = 0
reward = agent_play(env, agent)
optimal_reward = agent_play(env, OptimalAgent())
plt.title('track')
plt.legend(['NAF', 'Optimal'])
plt.show()
print('optimal', optimal_reward)
print('fitted', reward)
plt.plot(range(episodes_n), mean_rewards)
plt.plot(range(episodes_n), env.optimal_v * np.ones(episodes_n))
plt.title('fit')
plt.legend(['NAF', 'Optimal'])
plt.show()
plt.plot(range(episodes_n), mean_times)
plt.title('times')
plt.legend(['NAF'])
plt.show()
torch.save(agent.Q.state_dict(), './result')
