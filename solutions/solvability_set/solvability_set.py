import torch.nn as nn
import torch

from models.double_naf import DoubleNAFAgent
from problems.solvability_set.solvability_set_env import SolvabilitySet
from utilities.DiffGamesResolver import DiffGamesResolver
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network
import matplotlib.pyplot as plt

state_shape = 3
action_shape = 1
episode_n = 500


def init_u_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 100, 100, action_shape], nn.Sigmoid(), nn.Tanh())
    p_model = Seq_Network([state_shape, 100, 100, action_shape ** 2], nn.Sigmoid())
    v_model = Seq_Network([state_shape, 100, 100, 1], nn.Sigmoid())
    noise = OUNoise(1, threshold=action_max, threshold_min=0.001, threshold_decrease=2 * action_max / episode_n)
    agent = DoubleNAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size,
                           gamma=0.99)
    return agent


def init_v_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 100, 100, action_shape], nn.Sigmoid(), nn.Tanh())
    p_model = Seq_Network([state_shape, 100, 100, action_shape ** 2], nn.Sigmoid())
    v_model = Seq_Network([state_shape, 100, 100, 1], nn.Sigmoid())
    noise = OUNoise(1, threshold=action_max, threshold_min=0.001, threshold_decrease=2 * action_max / episode_n)
    agent = DoubleNAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size,
                           gamma=0.99)
    return agent


def test_agent(env, u_agent, v_agent):
    state = env.reset()
    total_reward = 0
    xs = []
    us = []
    vs = []
    ys = []
    steps = 0
    while not env.done:
        u_action = u_agent.get_action(state)
        v_action = v_agent.get_action(state)
        us.append(u_action[0])
        vs.append(v_action[0])
        next_state, reward, done, _ = env.step(u_action, v_action)
        next_state = next_state
        reward = float(reward)
        total_reward += reward
        state = next_state
        xs.append(state[1])
        ys.append(state[2])
        steps += 1

    figure, axes = plt.subplots()
    M = plt.Circle((0, 0), 4, color='r', fill=False)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    axes.set_aspect(1)
    plt.axvline(x=state[1], ymin=-5, ymax=5, color='g')
    axes.add_artist(M)
    plt.plot(xs, ys)
    plt.show()
    plt.plot(range(steps), us)
    plt.title("u actions")
    plt.show()
    plt.plot(range(steps), vs)
    plt.title("v actions")
    plt.show()
    return total_reward


if __name__ == '__main__':
    env = SolvabilitySet(initial_x=0, initial_y=1 / 2)
    resolver = DiffGamesResolver()
    u_agent = init_u_agent(state_shape, action_shape, env.u_action_max, 128)
    v_agent = init_v_agent(state_shape, action_shape, env.v_action_max, 128)
    # u_agent.Q.load_state_dict(torch.load('./u_result' + str(env.initial_x) + str(env.initial_y)))
    # v_agent.Q.load_state_dict(torch.load('./v_result00'))
    rewards = resolver.fit_agents(env, episode_n, u_agent, v_agent, fit_step=10)
    u_agent.noise.threshold = 0
    v_agent.noise.threshold = 0
    test_agent(env, u_agent, v_agent)
    # torch.save(u_agent.Q.state_dict(), './u_result' + str(env.initial_x) + str(env.initial_y))
    # torch.save(v_agent.Q.state_dict(), './v_result' + str(env.initial_x) + str(env.initial_y))

    #
    #
    # u_agent = init_u_agent(state_shape, action_shape, env.u_action_max, 128)
    # v_agent = init_v_agent(state_shape, action_shape, env.v_action_max, 64)
    # resolver.fit_v_agent(env, episode_n, u_agent, v_agent)

    # v_agent.noise.threshold = 0
    # test_agent(env, u_agent, v_agent)
