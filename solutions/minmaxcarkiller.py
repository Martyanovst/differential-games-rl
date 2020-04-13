import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import models.minmaxnaf
from problems.carkiller.optimal_agents import OptimalVAgent, OptimalUAgent
from problems.carkiller.other_agents import ConstantVAgent, SinVAgent, OptimalConstantCounterVAgent, SinCosUAgent
from problems.carkiller.two_points_on_parallel_lines_env import TwoPointsOnParallelLines
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

state_shape = 5
action_shape = 1
episode_n = 1000


def init_u_agent(state_shape, action_shape, action_max):
    mu_model = Seq_Network([state_shape, 50, 50, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 50, 50, action_shape ** 2], nn.ReLU())
    model = models.minmaxnaf.QModel(mu_model, p_model, action_shape, action_max)
    return model


def init_v_agent(state_shape, action_shape, action_max):
    mu_model = Seq_Network([state_shape, 50, 50, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 50, 50, action_shape ** 2], nn.ReLU())
    model = models.minmaxnaf.QModel(mu_model, p_model, action_shape, action_max)
    return model


def init_agent(state_shape, action_shape, u_max, v_max, batch_size):
    u_model = init_u_agent(state_shape, action_shape, u_max)
    v_model = init_v_agent(state_shape, action_shape, v_max)
    v_network = Seq_Network([state_shape, 16, 16, 1], nn.ReLU())
    noise = OUNoise(1, threshold=1, threshold_min=0.002, threshold_decrease=0.000001)
    agent = models.minmaxnaf.MinmaxNafAgent(u_model, v_model, v_network, noise, state_shape, action_shape, u_max, v_max,
                                            batch_size)
    return agent


def get_state(t, position):
    return np.append([t], position)


def fit_agent(env, episode_n, agent):
    rewards = np.zeros(episode_n)
    mean_rewards = np.zeros(episode_n)
    for episode in range(episode_n):
        state = env.reset()
        total_reward = 0
        while not env.done:
            u_action = agent.get_u_action(state)
            v_action = agent.get_v_action(state)
            next_state, reward, done, _ = env.step(u_action[0], v_action[0])
            reward = float(reward)
            total_reward += -reward
            agent.fit(state, u_action, v_action, reward, done, next_state)
            state = next_state
        rewards[episode] = total_reward
        mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
        mean_rewards[episode] = mean_reward
        print("episode=%.0f, total reward=%.3f, u-threshold=%0.3f" % (episode, mean_reward, agent.noise.threshold))
    return mean_rewards


def play(u_agent, v_agent, env):
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


def test_agents(u_agent, v_agent, env, title):
    reward = play(u_agent, v_agent, env)
    print(-reward, title)


if __name__ == '__main__':
    env = TwoPointsOnParallelLines()
    agent = init_agent(state_shape, action_shape, env.u_action_max, env.v_action_max, 128)
    rewards = fit_agent(env, episode_n, agent)
    agent.noise.threshold = 0
    plt.plot(range(episode_n), rewards)
    plt.title('fit')
    plt.show()

    print(OptimalConstantCounterVAgent(env, agent).beta, "test fitted agent")
    test_agents(agent, OptimalVAgent(env), env, "fitted vs. optimal")
    test_agents(OptimalUAgent(env), OptimalVAgent(env), env, "optimal vs. optimal")
    test_agents(agent, ConstantVAgent(env, 0), env, "fitted vs. 0")
    test_agents(agent, ConstantVAgent(env, 0.5), env, "fitted vs. 0.5")
    test_agents(agent, ConstantVAgent(env, 1), env, "fitted vs. 1")
    test_agents(agent, ConstantVAgent(env, -0.5), env, "fitted vs. -0.5")
    test_agents(agent, ConstantVAgent(env, -1), env, "fitted vs. -1")
    test_agents(agent, SinVAgent(env), env, "fitted vs. sin")
    u_agent = OptimalUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).beta, "test optimal")
    u_agent = SinCosUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).beta, "test sincos")

    u_agent = OptimalUAgent(env)
    test_agents(u_agent, ConstantVAgent(env, 0), env, "optimal vs. 0")
    test_agents(u_agent, ConstantVAgent(env, 0.5), env, "optimal vs. 0.5")
    test_agents(u_agent, ConstantVAgent(env, 1), env, "optimal vs. 1")
    test_agents(u_agent, ConstantVAgent(env, -0.5), env, "optimal vs. -0.5")
    test_agents(u_agent, ConstantVAgent(env, -1), env, "optimal vs. -1")
    test_agents(u_agent, SinVAgent(env), env, "optimal vs. sin")
