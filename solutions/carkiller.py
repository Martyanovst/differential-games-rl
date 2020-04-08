import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from models.naf import NAFAgent
from problems.carkiller.optimal_agents import OptimalUAgent, OptimalVAgent
from problems.carkiller.other_agents import OptimalConstantCounterVAgent, SinCosUAgent, ConstantVAgent, SinVAgent
from problems.carkiller.two_points_on_parallel_lines_env import TwoPointsOnParallelLines
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

state_shape = 5
action_shape = 1
episode_n = 1000


def init_u_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 50, 50, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 50, 50, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 50, 50, 1], nn.ReLU())
    noise = OUNoise(1, threshold=action_max, threshold_min=0.0000001, threshold_decrease=0.000007)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size)
    return agent


def init_v_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 20, 20, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 20, 20, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 20, 20, 1], nn.ReLU())
    noise = OUNoise(1, threshold=action_max, threshold_min=0.000001, threshold_decrease=0.00001)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size)
    return agent


def get_state(t, position):
    return np.append([t], position)


def fit_agents(env, episode_n, u_agent, v_agent):
    rewards = np.zeros(episode_n)
    mean_rewards = np.zeros(episode_n)
    counter = 0
    for episode in range(episode_n):
        # state = get_state(*env.reset())
        state = env.reset()
        total_reward = 0
        while not env.done:
            u_action = u_agent.get_action(state)
            v_action = v_agent.get_action(state)
            next_state, reward, done, _ = env.step(u_action[0], v_action[0])
            next_state = next_state
            reward = float(reward)
            total_reward += -reward
            if counter // 100 % 2 == 0:
                u_agent.fit(state, u_action, -reward, done, next_state)
                v_agent.memory.append([state, v_action, reward, done, next_state])
            else:
                v_agent.fit(state, v_action, reward, done, next_state)
                u_agent.memory.append([state, u_action, -reward, done, next_state])
            counter += 1
            state = next_state
        rewards[episode] = total_reward
        mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
        mean_rewards[episode] = mean_reward
        print("episode=%.0f, total reward=%.3f, u-threshold=%0.3f" % (episode, mean_reward, u_agent.noise.threshold))
    return mean_rewards


def fit_v_agent(env, episode_n, u_agent, v_agent):
    rewards = np.zeros(episode_n)
    mean_rewards = np.zeros(episode_n)
    for episode in range(episode_n):
        state = get_state(*env.reset())
        total_reward = 0
        while not env.done:
            u_action = u_agent.get_action(state)
            v_action = v_agent.get_action(state)
            next_state, reward, done, _ = env.step(u_action[0], v_action[0])
            next_state = get_state(*next_state)
            reward = float(reward)
            total_reward += reward
            v_agent.fit(state, v_action, reward, done, next_state)
            state = next_state
        rewards[episode] = total_reward
        mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
        mean_rewards[episode] = mean_reward
        print("episode=%.0f, total reward=%.3f, u-threshold=%0.3f" % (episode, mean_reward, v_agent.noise.threshold))
    return mean_rewards


def play(u_agent, v_agent, env):
    # state = get_state(*env.reset())
    state = env.reset()
    total_reward = 0
    while not env.done:
        u_action = u_agent.get_action(state)
        v_action = v_agent.get_action(state)
        next_state, reward, done, _ = env.step(u_action[0], v_action[0])
        next_state = next_state
        reward = float(reward)
        total_reward += reward
        state = next_state
    return total_reward


def test_agents(u_agent, v_agent, env):
    reward = play(u_agent, v_agent, env)
    print(-reward)


if __name__ == '__main__':
    env = TwoPointsOnParallelLines()

    u_agent = init_u_agent(state_shape, action_shape, env.u_action_max, 128)
    v_agent = init_v_agent(state_shape, action_shape, env.v_action_max, 128)
    rewards = fit_agents(env, episode_n, u_agent, v_agent)
    # rewards = fit_agents(env, episode_n, u_agent, OptimalVAgent(env))
    u_agent.noise.threshold = 0
    plt.plot(range(episode_n), rewards)
    plt.title('fit')
    plt.show()

    print(OptimalConstantCounterVAgent(env, u_agent).get_beta())
    test_agents(OptimalUAgent(env), OptimalVAgent(env), env)
    test_agents(u_agent, ConstantVAgent(env, 0), env)
    test_agents(u_agent, ConstantVAgent(env, 0.5), env)
    test_agents(u_agent, ConstantVAgent(env, 1), env)
    test_agents(u_agent, SinVAgent(env), env)
    u_agent = OptimalUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).get_beta())
    u_agent = SinCosUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).get_beta())

    u_agent = OptimalUAgent(env)
    test_agents(u_agent, ConstantVAgent(env, 0), env)
    test_agents(u_agent, ConstantVAgent(env, 0.5), env)
    test_agents(u_agent, ConstantVAgent(env, 1), env)
    test_agents(u_agent, SinVAgent(env), env)
