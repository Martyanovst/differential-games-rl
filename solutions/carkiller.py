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
    mu_model = Seq_Network([state_shape, 100, 100, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 100, 100, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 50, 50, 1], nn.ReLU())
    noise = OUNoise(1, threshold=1, threshold_min=0.2, threshold_decrease=0.002)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size)
    return agent


def init_v_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 20, 20, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 20, 20, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 16, 16, 1], nn.ReLU())
    noise = OUNoise(1, threshold=1, threshold_min=0.2, threshold_decrease=0.002)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size)
    return agent


def get_state(t, position):
    return np.append([t], position)


def fit_agents(env, episode_n, u_agent, v_agent):
    rewards = np.zeros(episode_n)
    mean_rewards = np.zeros(episode_n)
    is_u_agent_fit = False
    for episode in range(episode_n):
        state = env.reset()
        total_reward = 0
        while not env.done:
            u_action = u_agent.get_action(state, is_u_agent_fit)
            v_action = v_agent.get_action(state, not is_u_agent_fit)
            next_state, reward, done, _ = env.step(u_action[0], v_action[0])
            reward = float(reward)
            total_reward += -reward
            if is_u_agent_fit:
                u_agent.fit(state, u_action, -reward, done, next_state)
            else:
                v_agent.fit(state, v_action, reward, done, next_state)
            state = next_state
        if is_u_agent_fit:
            u_agent.noise.decrease()
        else:
            v_agent.noise.decrease()
        is_u_agent_fit = episode // 50 % 2 == 1
        rewards[episode] = total_reward
        mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
        mean_rewards[episode] = mean_reward
        print("episode=%.0f, total reward=%.3f, u-threshold=%0.3f, v-threshold=%0.3f" % (episode, mean_reward,
                                                                                         u_agent.noise.threshold,
                                                                                         v_agent.noise.threshold))
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

    u_agent = init_u_agent(state_shape, action_shape, env.u_action_max, 64)
    v_agent = init_v_agent(state_shape, action_shape, env.v_action_max, 64)

    rewards = fit_agents(env, episode_n, u_agent, v_agent)
    u_agent.noise.threshold = 0
    plt.plot(range(episode_n), rewards)
    plt.title('fit')
    plt.show()

    print(OptimalConstantCounterVAgent(env, u_agent).get_beta(), "test fitted agent")
    test_agents(u_agent, OptimalVAgent(env), env, "fitted vs. optimal")
    test_agents(OptimalUAgent(env), OptimalVAgent(env), env, "optimal vs. optimal")
    test_agents(u_agent, ConstantVAgent(env, 0), env, "fitted vs. 0")
    test_agents(u_agent, ConstantVAgent(env, 0.5), env, "fitted vs. 0.5")
    test_agents(u_agent, ConstantVAgent(env, 1), env, "fitted vs. 1")
    test_agents(u_agent, ConstantVAgent(env, -0.5), env, "fitted vs. -0.5")
    test_agents(u_agent, ConstantVAgent(env, -1), env, "fitted vs. -1")
    test_agents(u_agent, SinVAgent(env), env, "fitted vs. sin")
    u_agent = OptimalUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).get_beta(), "test optimal")
    u_agent = SinCosUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).get_beta(), "test sincos")

    u_agent = OptimalUAgent(env)
    test_agents(u_agent, ConstantVAgent(env, 0), env, "optimal vs. 0")
    test_agents(u_agent, ConstantVAgent(env, 0.5), env, "optimal vs. 0.5")
    test_agents(u_agent, ConstantVAgent(env, 1), env, "optimal vs. 1")
    test_agents(u_agent, ConstantVAgent(env, -0.5), env, "optimal vs. -0.5")
    test_agents(u_agent, ConstantVAgent(env, -1), env, "optimal vs. -1")
    test_agents(u_agent, SinVAgent(env), env, "optimal vs. sin")
