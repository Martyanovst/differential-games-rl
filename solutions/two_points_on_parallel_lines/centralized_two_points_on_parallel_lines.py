import torch.nn as nn

import models.centrilized_naf
from problems.two_points_on_parallel_lines.optimal_agents import OptimalVAgent, OptimalUAgent
from problems.two_points_on_parallel_lines.other_agents import ConstantVAgent, SinVAgent, OptimalConstantCounterVAgent, \
    SinCosUAgent
from problems.two_points_on_parallel_lines.two_points_on_parallel_lines_env import TwoPointsOnParallelLines
from utilities.DiffGamesCentralizedResolver import DiffGamesCentralizedResolver
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

state_shape = 5
action_shape = 1
episode_n = 300


def init_u_agent(state_shape, action_shape, action_max):
    mu_model = Seq_Network([state_shape, 50, 50, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 50, 50, action_shape ** 2], nn.ReLU())
    model = models.centrilized_naf.QModel(mu_model, p_model, action_shape, action_max)
    return model


def init_v_agent(state_shape, action_shape, action_max):
    mu_model = Seq_Network([state_shape, 50, 50, action_shape], nn.ReLU(), nn.Tanh())
    p_model = Seq_Network([state_shape, 50, 50, action_shape ** 2], nn.ReLU())
    model = models.centrilized_naf.QModel(mu_model, p_model, action_shape, action_max)
    return model


def init_agent(state_shape, action_shape, u_max, v_max, batch_size):
    u_model = init_u_agent(state_shape, action_shape, u_max)
    v_model = init_v_agent(state_shape, action_shape, v_max)
    v_network = Seq_Network([state_shape, 16, 16, 1], nn.ReLU())
    noise = OUNoise(1, threshold=1, threshold_min=0.002, threshold_decrease=0.002)
    agent = models.centrilized_naf.CentralizedNafAgent(u_model, v_model, v_network, noise, state_shape, action_shape,
                                                       u_max, v_max,
                                                       batch_size)
    return agent


if __name__ == '__main__':
    env = TwoPointsOnParallelLines()
    resolver = DiffGamesCentralizedResolver()
    agent = init_agent(state_shape, action_shape, env.u_action_max, env.v_action_max, 64)
    rewards = resolver.fit_agent(env, episode_n, agent)
    agent.noise.threshold = 0

    print(OptimalConstantCounterVAgent(env, agent).beta)
    resolver.test_u_agents(env, agent, OptimalVAgent(env), title="fitted vs. optimal")
    resolver.test_agents(env, OptimalUAgent(env), OptimalVAgent(env), title="optimal vs. optimal")
    resolver.test_u_agents(env, agent, ConstantVAgent(env, 0), title="fitted vs. 0")
    resolver.test_u_agents(env, agent, ConstantVAgent(env, 0.5), title="fitted vs. 0.5")
    resolver.test_u_agents(env, agent, ConstantVAgent(env, 1), title="fitted vs. 1")
    resolver.test_u_agents(env, agent, ConstantVAgent(env, -0.5), title="fitted vs. -0.5")
    resolver.test_u_agents(env, agent, ConstantVAgent(env, -1), title="fitted vs. -1")
    resolver.test_u_agents(env, agent, SinVAgent(env), title="fitted vs. sin")
    u_agent = OptimalUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).beta)
    u_agent = SinCosUAgent(env)
    print(OptimalConstantCounterVAgent(env, u_agent).beta)

    u_agent = OptimalUAgent(env)
    resolver.test_agents(env, u_agent, ConstantVAgent(env, 0), title="optimal vs. 0")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, 0.5), title="optimal vs. 0.5")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, 1), title="optimal vs. 1")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, -0.5), title="optimal vs. -0.5")
    resolver.test_agents(env, u_agent, ConstantVAgent(env, -1), title="optimal vs. -1")
    resolver.test_agents(env, u_agent, SinVAgent(env), title="optimal vs. sin")
