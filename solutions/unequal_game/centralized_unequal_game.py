import torch.nn as nn

import models.centrilized_naf
from problems.unequal_game.optimal_agents import OptimalVAgent, DummyVAgent
from problems.unequal_game.unequal_game_env import UnequalGame
from utilities.DiffGamesCentralizedResolver import DiffGamesCentralizedResolver
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

state_shape = 2
action_shape = 1
episode_n = 150


def init_u_agent(state_shape, action_shape, action_max):
    mu_model = Seq_Network([state_shape, 32, 32, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 32, 32, action_shape ** 2], nn.ReLU())
    model = models.centrilized_naf.QModel(mu_model, p_model, action_shape, action_max)
    return model


def init_v_agent(state_shape, action_shape, action_max):
    mu_model = Seq_Network([state_shape, 32, 32, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 32, 32, action_shape ** 2], nn.ReLU())
    model = models.centrilized_naf.QModel(mu_model, p_model, action_shape, action_max)
    return model


def init_agent(state_shape, action_shape, u_max, v_max, batch_size):
    u_model = init_u_agent(state_shape, action_shape, u_max)
    v_model = init_v_agent(state_shape, action_shape, v_max)
    v_network = Seq_Network([state_shape, 16, 16, 1], nn.ReLU())
    noise = OUNoise(1, threshold=1, threshold_min=0.0000001, threshold_decrease=0.000005)
    agent = models.centrilized_naf.CentralizedNafAgent(u_model, v_model, v_network, noise, state_shape, action_shape,
                                                       u_max, v_max,
                                                       batch_size)
    return agent


if __name__ == '__main__':
    env = UnequalGame()
    resolver = DiffGamesCentralizedResolver()
    agent = init_agent(state_shape, action_shape, env.u_action_max, env.v_action_max, 64)
    resolver.fit_agent(env, episode_n, agent)
    agent.noise.threshold = 0
    resolver.test_agents(env, agent, OptimalVAgent(env), 'Optimal V-agent')
    resolver.test_agents(env, agent, DummyVAgent(0), 'Constant 0 value agent')
    resolver.test_agents(env, agent, DummyVAgent(0.5), 'Constant 0.5 value agent')
    resolver.test_agents(env, agent, DummyVAgent(1), 'Constant 1 value agent')
    resolver.test_agents(env, agent, DummyVAgent(-0.5), 'Constant -0.5 value agent')
    resolver.test_agents(env, agent, DummyVAgent(-1), 'Constant -1 value agent')
