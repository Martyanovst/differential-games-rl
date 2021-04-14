from collections import deque

import numpy as np
import torch.nn as nn

from models.bounded.bounded_naf import Bounded_NAF
from problems.dubins_car.dubins_car_env import DubinsCar
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

env = DubinsCar(inner_step_n=100)
state_shape = env.state_dim
action_shape = env.action_dim
action_max = env.action_max
action_min = env.action_min
episodes_n = 200
epsilon_min = 0.00001
batch_size = 256
epsilon = 1


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
    agent.noise.decrease()
    return total_reward


def agent_play(env, agent):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward


max_iterations = episodes_n * 3


def fit_with_dt(agent, dt):
    global idx, mean_rewards, rewards
    env = DubinsCar(dt=dt, inner_step_n=100)
    agent.noise.threshold = epsilon
    agent.memory = deque(maxlen=100000)
    for episode in range(episodes_n):
        reward = play_and_learn(env)
        rewards[idx] = reward
        mean_reward = np.mean(rewards[max(0, idx - 25):idx + 1])
        mean_rewards[idx] = mean_reward
        print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f" % (
            idx, agent.noise.threshold, reward, mean_reward))
        idx += 1


for i in range(5):
    print('-----------------------------------  ' + str(i) + '  -------------------------------------------')
    rewards = np.zeros(max_iterations)
    mean_rewards = np.zeros(max_iterations)
    idx = 0
    mu_model = Seq_Network([state_shape, 256, 128, action_shape], nn.ReLU(), nn.Tanh())
    phi_model = Seq_Network([state_shape, 256, 128, action_shape], nn.ReLU())
    p_model = Seq_Network([state_shape, 256, 128, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 256, 128, 1], nn.ReLU())
    noise = OUNoise(action_shape, threshold=epsilon, threshold_min=epsilon_min,
                    threshold_decrease=(epsilon_min / epsilon) ** (1 / episodes_n))
    agent = Bounded_NAF(mu_model, p_model, v_model, phi_model, noise, state_shape, action_shape, action_max,
                        action_min=action_min,
                        batch_size=batch_size,
                        gamma=1,
                        learning_n_per_fit=8)
    fit_with_dt(agent, 2)
    fit_with_dt(agent, 0.5)
    fit_with_dt(agent, 0.25)
    np.save('./test/bounded_test/' + str(i), mean_rewards)
