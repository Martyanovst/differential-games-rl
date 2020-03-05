import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from models.naf import NAFAgent
from problems.paratrooper.optimal_agents import OptimalVAgent, DummyVAgent
from problems.paratrooper.unequal_game_env import UnequalGame
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

state_shape = 2
action_shape = 1
episode_n = 300


def init_u_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 16, 16, 1], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 16, 16, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 16, 16, 1], nn.ReLU())
    noise = OUNoise(1, threshold=1, threshold_min=0.0000001, threshold_decrease=0.000005)
    # noise = OUNoise(1)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size)
    return agent


def init_v_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 16, 16, 1], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 16, 16, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 16, 16, 1], nn.ReLU())
    noise = OUNoise(1, threshold=1, threshold_min=0.0000001, threshold_decrease=0.000005)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size)
    return agent


def get_state(t, position):
    return np.append([t], position)


def fit_agents(env, episode_n, u_agent, v_agent, learning_q=1):
    rewards = []
    for episode in range(episode_n):
        state = get_state(*env.reset())
        total_reward = 0
        while not env.done:
            u_action = u_agent.get_action(state)
            v_action = v_agent.get_action(state)
            # v_action = 0
            next_state, reward, done, _ = env.step(u_action, v_action)
            next_state = get_state(*next_state)
            reward = float(reward)
            total_reward += reward
            u_agent.fit(state, u_action, -reward, done, next_state)
            v_agent.fit(state, v_action, reward, done, next_state)
            state = next_state
        print("episode=%.0f, total reward=%.3f, u-threshold=%0.3f" % (episode, total_reward, u_agent.noise.threshold))
        rewards.append(total_reward)
    return rewards


def play(u_agent, v_agent):
    env = UnequalGame()
    state = get_state(*env.reset())
    total_reward = 0
    while not env.done:
        u_action = u_agent.get_action(state)
        v_action = v_agent.get_action(state)
        next_state, reward, done, _ = env.step(u_action, v_action)
        next_state = get_state(*next_state)
        reward = float(reward)
        total_reward += reward
        state = next_state
    return total_reward


def test_agents(u_agent, v_agent, tests_count, title):
    rewards = []
    for i in range(tests_count):
        reward = play(u_agent, v_agent)
        rewards.append(reward)
    plt.plot(range(tests_count), rewards)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    env = UnequalGame()
    u_agent = init_u_agent(state_shape, action_shape, env.u_action_max, 200)
    v_agent = init_v_agent(state_shape, action_shape, env.v_action_max, 200)
    rewards = fit_agents(env, episode_n, u_agent, v_agent)
    # rewards = fit_agents(env, episode_n, u_agent, OptimalVAgent(env))
    plt.plot(range(episode_n), rewards)
    plt.title('fit')
    plt.show()
    test_agents(u_agent, OptimalVAgent(env), 10, 'Optimal V-agent')
    test_agents(u_agent, DummyVAgent(0), 10, 'Constant 0 value agent')
    test_agents(u_agent, DummyVAgent(0.5), 10, 'Constant 0.5 value agent')
    test_agents(u_agent, DummyVAgent(1), 10, 'Constant 1 value agent')
