import numpy as np
import torch
import torch.nn as nn

from solutions.cross_entropy.cross_entropy_v import Agent
from solutions.cross_entropy.optimal_agents import OptimalVAgent, DummyVAgent
from solutions.cross_entropy.unequal_game_env import UnequalGame
from utilities.sequentialNetwork import Seq_Network
import matplotlib.pyplot as plt

state_shape = 2
action_shape = 1
episode_n = 100


def init_v_model(state_shape):
    v_model = Seq_Network([state_shape, 25, 25, 1], nn.ReLU())
    return v_model


def fit_agent(env, episode_n, agent, title='fit'):
    rewards = np.zeros(episode_n)
    mean_rewards = np.zeros(episode_n)
    for episode in range(episode_n):
        state = env.reset()
        total_reward = 0
        while not env.done:
            u_action = agent.get_u_action(state)
            v_action = agent.get_v_action(state)
            next_state, reward, done, _ = env.step(u_action, v_action)
            reward = float(reward)
            total_reward += reward
            agent.fit(state, reward, done, next_state)
            state = next_state
        agent.noise_decrease()
        rewards[episode] = total_reward
        mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
        mean_rewards[episode] = mean_reward
        print("episode=%.0f, total reward=%.3f, threshold=%0.3f" % (
            episode, total_reward, agent.epsilon))
    plt.plot(range(episode_n), mean_rewards)
    plt.title(title)
    plt.show()
    return agent


def play(env, u_agent, v_agent):
    state = env.reset()
    total_reward = 0
    while not env.done:
        u_action = u_agent.get_u_action(state)
        v_action = v_agent.get_action(state)
        next_state, reward, done, _ = env.step(u_action, v_action)
        reward = float(reward)
        total_reward += reward
        state = next_state
    return total_reward


def test_agents(env, u_agent, v_agent, title):
    reward = play(env, u_agent, v_agent)
    print(reward, title)


if __name__ == '__main__':
    env = UnequalGame()
    v_model = init_v_model(state_shape)
    agent = Agent(env, v_model, batch_size=64)
    fit_agent(env, episode_n, agent)
    torch.save(agent.state_dict(), './result')
    agent.epsilon = 0

    test_agents(env, agent, OptimalVAgent(env), 'Optimal V-agent')
    test_agents(env, agent, DummyVAgent(0), 'Constant 0 value agent')
    test_agents(env, agent, DummyVAgent(0.5), 'Constant 0.5 value agent')
    test_agents(env, agent, DummyVAgent(1), 'Constant 1 value agent')
