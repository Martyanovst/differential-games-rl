import torch.nn as nn
import torch
from models.naf import NAFAgent
from problems.solvability_set.solvability_set_env import SolvabilitySet
from problems.unequal_game.optimal_agents import OptimalVAgent, DummyVAgent, DummyUAgent, OptimalUAgent
from problems.unequal_game.unequal_game_env import UnequalGame
from utilities.DiffGamesResolver import DiffGamesResolver
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

state_shape = 3
action_shape = 1
episode_n = 500


def init_u_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 100, 100, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 100, 100, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 100, 100, 1], nn.ReLU())
    noise = OUNoise(1, threshold=action_max, threshold_min=0.001, threshold_decrease=0.001 * action_max)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size, gamma=1)
    return agent


def init_v_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 100, 100, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 100, 100, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 100, 100, 1], nn.ReLU())
    noise = OUNoise(1, threshold=action_max, threshold_min=0.001, threshold_decrease=0.002 * action_max)
    agent = NAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size, gamma=1)
    return agent


if __name__ == '__main__':
    env = SolvabilitySet()
    resolver = DiffGamesResolver()
    u_agent = init_u_agent(state_shape, action_shape, env.u_action_max, 128)
    # u_agent.Q.load_state_dict(torch.load('./result'))
    v_agent = init_v_agent(state_shape, action_shape, env.v_action_max, 128)
    # rewards = resolver.fit_agents(env, episode_n, u_agent, v_agent)
    # torch.save(u_agent.Q.state_dict(), './result')

    u_agent.noise.threshold = 0
    #
    # resolver.test_agents(env, u_agent, OptimalVAgent(env), 'Optimal V-agent')
    # resolver.test_agents(env, u_agent, DummyVAgent(0), 'Constant 0 value agent')
    # resolver.test_agents(env, u_agent, DummyVAgent(0.5), 'Constant 0.5 value agent')
    # resolver.test_agents(env, u_agent, DummyVAgent(1), 'Constant 1 value agent')
    #
    # v_agent = init_v_agent(state_shape, action_shape, env.v_action_max, 64)
    # resolver.test_agents(env, DummyUAgent(env), v_agent, 'Versus dummy U-agent before learning')
    # resolver.test_agents(env, OptimalUAgent(env), v_agent, 'Versus optimal U-agent before learning')
    resolver.fit_v_agent(env, episode_n, u_agent, v_agent)
    #
    # v_agent.noise.threshold = 0
    #
    # resolver.test_agents(env, DummyUAgent(env), v_agent, 'Versus dummy U-agent after learning')
    # resolver.test_agents(env, OptimalUAgent(env), v_agent, 'Versus optimal U-agent after learning')
