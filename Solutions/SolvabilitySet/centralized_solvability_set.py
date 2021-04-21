import torch
import torch.nn as nn

from models.centrilized_naf import CentralizedNafAgent, QModel
from models.naf import NAFAgent
from problems.solvability_set.solvability_set_env import SolvabilitySet
from utilities.DiffGamesCentralizedResolver import DiffGamesCentralizedResolver
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network
import matplotlib.pyplot as plt

state_shape = 3
action_shape = 1
episode_n = 500


def init_u_agent(state_shape, action_shape, action_max):
    mu_model = Seq_Network([state_shape, 50, 50, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 50, 50, action_shape ** 2], nn.ReLU())
    model = QModel(mu_model, p_model, action_shape, action_max)
    return model


def init_test_u_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 100, 100, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 100, 100, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 100, 100, 1], nn.ReLU())
    # noise = OUNoise(1, threshold=action_max, threshold_min=0.001, threshold_decrease=action_max / episode_n)
    noise = OUNoise(1, threshold=1, threshold_min=0.001, threshold_decrease=1 / episode_n)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size, gamma=1)
    return agent


def init_v_agent(state_shape, action_shape, action_max):
    mu_model = Seq_Network([state_shape, 50, 50, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 50, 50, action_shape ** 2], nn.ReLU())
    model = QModel(mu_model, p_model, action_shape, action_max)
    return model


def init_test_v_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 100, 100, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 100, 100, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 100, 100, 1], nn.ReLU())
    # noise = OUNoise(1, threshold=action_max, threshold_min=0.001, threshold_decrease=action_max / episode_n)
    noise = OUNoise(1, threshold=1, threshold_min=0.001, threshold_decrease=1 / episode_n)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size, gamma=1)
    return agent


def init_agent(state_shape, action_shape, u_max, v_max, batch_size):
    u_model = init_u_agent(state_shape, action_shape, u_max)
    v_model = init_v_agent(state_shape, action_shape, v_max)
    v_network = Seq_Network([state_shape, 50, 50, 1], nn.ReLU())
    noise = OUNoise(1, threshold=1, threshold_min=0.001, threshold_decrease=1 / episode_n)
    agent = CentralizedNafAgent(u_model, v_model, v_network, noise, state_shape, action_shape,
                                       u_max, v_max,
                                       batch_size)
    return agent


def test_agent(env, agent):
    state = env.reset()
    total_reward = 0
    xs = []
    us = []
    vs = []
    ys = []
    steps = 0
    while not env.done:
        u_action = agent.get_u_action(state)
        v_action = agent.get_v_action(state)
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
    plt.show()
    plt.plot(range(steps), vs)
    plt.show()
    return total_reward


if __name__ == '__main__':
    env = SolvabilitySet(initial_x=0, initial_y=0)
    resolver = DiffGamesCentralizedResolver()
    agent = init_agent(state_shape, action_shape, env.u_action_max, env.v_action_max, 128)
    resolver.fit_agent(env, episode_n, agent)
    agent.noise.threshold = 0
    # agent.Q.load_state_dict(torch.load('./result02'))
    torch.save(agent.Q.state_dict(), './result02')
    test_agent(env, agent)
    # u_agent = init_test_u_agent(state_shape, action_shape, env.u_action_max, 64)
    # v_agent = init_test_v_agent(state_shape, action_shape, env.v_action_max, 128)
    # resolver.fit_v_agent(env, episode_n, agent, v_agent)
