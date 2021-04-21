import torch.nn as nn

import models.centrilized_naf
from models.centralized_double_naf import CentralizedDoubleNafAgent
from problems.point_on_a_plane.point_on_a_plane import PointOnAPlane
from utilities.DiffGamesCentralizedResolver import DiffGamesCentralizedResolver
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

state_shape = 5
action_shape = 2
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
    v_network = Seq_Network([state_shape, 50, 50, 1], nn.ReLU())
    noise = OUNoise(2, threshold=1, threshold_min=0.002, threshold_decrease=0.003)
    agent = CentralizedDoubleNafAgent(u_model, v_model, v_network, noise, state_shape,
                                      action_shape,
                                      u_max, v_max,
                                      batch_size)
    return agent


if __name__ == '__main__':
    env = PointOnAPlane()
    resolver = DiffGamesCentralizedResolver()
    agent = init_agent(state_shape, action_shape, env.u_action_max, env.v_action_max, 128)
    resolver.fit_agent(env, episode_n, agent)
    # torch.save(u_agent.Q.state_dict(), './resultU')
    # torch.save(v_agent.Q.state_dict(), './resultV')
