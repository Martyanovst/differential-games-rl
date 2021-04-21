import torch.nn as nn

from models.double_naf import DoubleNAFAgent
from problems.two_points_on_parallel_lines.optimal_agents import OptimalUAgent, OptimalVAgent
from problems.two_points_on_parallel_lines.other_agents import OptimalConstantCounterVAgent, SinCosUAgent, \
    ConstantVAgent, SinVAgent
from problems.two_points_on_parallel_lines.two_points_on_parallel_lines_env import TwoPointsOnParallelLines
from utilities.DiffGamesResolver import DiffGamesResolver
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

state_shape = 5
action_shape = 1
episode_n = 300


def init_u_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 32, 32, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 32, 32, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 32, 32, 1], nn.ReLU())
    noise = OUNoise(1, threshold=action_max, threshold_min=0.2, threshold_decrease=0.01 * action_max)
    agent = DoubleNAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size, gamma=0.9999)
    return agent


def init_v_agent(state_shape, action_shape, action_max, batch_size):
    mu_model = Seq_Network([state_shape, 32, 32, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 32, 32, action_shape ** 2], nn.ReLU())
    v_model = Seq_Network([state_shape, 32, 32, 1], nn.ReLU())
    noise = OUNoise(1, threshold=action_max, threshold_min=0.2, threshold_decrease=0.005 * action_max)
    agent = DoubleNAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, action_max, batch_size, gamma=0.9999)
    return agent


if __name__ == '__main__':
    env = TwoPointsOnParallelLines()
    resolver = DiffGamesResolver()
    u_agent = init_u_agent(state_shape, action_shape, env.u_action_max, 128)
    v_agent = init_v_agent(state_shape, action_shape, env.v_action_max, 128)

    rewards = resolver.fit_agents(env, episode_n, u_agent, v_agent)
    u_agent.noise.threshold = 0

    print(OptimalConstantCounterVAgent(env, u_agent).get_beta(), "test fitted agent")
    resolver.test_agents(env, u_agent, OptimalVAgent(env), "fitted vs. optimal")
    resolver.test_agents(env, OptimalUAgent(env), OptimalVAgent(env), "optimal vs. optimal")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, 0), "fitted vs. 0")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, 0.5), "fitted vs. 0.5")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, 1), "fitted vs. 1")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, -0.5), "fitted vs. -0.5")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, -1), "fitted vs. -1")
    resolver.test_agents(env, u_agent, SinVAgent(env), "fitted vs. sin")
    u_agent = OptimalUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).get_beta(), "test optimal")
    u_agent = SinCosUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).get_beta(), "test sincos")

    u_agent = OptimalUAgent(env)
    resolver.test_agents(env, u_agent, ConstantVAgent(env, 0), "optimal vs. 0")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, 0.5), "optimal vs. 0.5")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, 1), "optimal vs. 1")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, -0.5), "optimal vs. -0.5")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, -1), "optimal vs. -1")
    resolver.test_agents(env, u_agent, SinVAgent(env), "optimal vs. sin")
    v_agent = init_v_agent(state_shape, action_shape, env.v_action_max, 128)
    resolver.fit_v_agent(env, episode_n, u_agent, v_agent, 'fit v-agent')
