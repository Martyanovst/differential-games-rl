import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from models.naf import NAFAgent
from problems.boundary_problem.boundary_problem_env import BoundaryProblem
from problems.boundary_problem.optimal_agent import OptimalAgent
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

env = BoundaryProblem(-1, 3)
state_shape = 3
action_max = 1.5
action_shape = 1
episodes_n = 500

mu_model = Seq_Network([state_shape, 100, 100, action_shape], nn.Sigmoid(), nn.Tanh())
p_model = Seq_Network([state_shape, 100, 100, action_shape ** 2], nn.Sigmoid())
v_model = Seq_Network([state_shape, 100, 100, 1], nn.Sigmoid())
noise = OUNoise(action_shape, threshold=1, threshold_min=0.001, threshold_decrease=0.00001)
batch_size = 200
agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size, 0.999)

def play_and_learn(env, learn=True):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        if learn:
            agent.fit(state, action, -reward, done, next_state)
            agent.noise.decrease()
        state = next_state
    t, x1, x2 = env.state
    return total_reward, x1, x2

def agent_play(env, agent):
    state = env.reset()
    total_reward = 0
    max_action = 0
    ts = []
    us = []
    while not env.done:
        action = agent.get_action(state)
        if abs(action[0]) > max_action:
            max_action = abs(action[0])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        ts.append(state[0])
        us.append(action[0])
        state = next_state
    plt.plot(ts, us)
    return total_reward


rewards = np.zeros(episodes_n)
mean_rewards = np.zeros(episodes_n)
for episode in range(episodes_n):
    reward, x1, x2 = play_and_learn(env)
    rewards[episode] = reward
    mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
    mean_rewards[episode] = mean_reward
    print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f, x1=%.3f, x2=%.3f" % (
        episode, agent.noise.threshold, rewards[episode], mean_reward, x1, x2))

agent.noise.threshold = 0
reward = agent_play(env, agent)
optimal_reward = agent_play(env, OptimalAgent(env))
plt.show()
print('optimal', optimal_reward)
print('fitted', reward)
plt.plot(range(episodes_n), mean_rewards)
plt.plot(range(episodes_n), np.ones(episodes_n) * optimal_reward)
plt.title('fit')
plt.legend(['NAF', 'Optimal'])
plt.show()
play_and_learn(env, learn=False)
