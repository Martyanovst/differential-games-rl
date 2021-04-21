import numpy as np
import gym
import torch.nn as nn
from Agents.NAF import NAF, QModel, QModel_SphereCase
from Agents.Utilities.SequentialNetwork import Seq_Network
from Agents.Utilities.Noises import OUNoise
from Resolvers import OneAgentSolver as solver
from Environments.DubinsCar.DubinsCar import DubinsCar_SymmetricActionInterval as DubinsCar
from Environments.DubinsCar.DubinsCarVisualizer import DubinsCarVisualizer

env = DubinsCar(dt=1, inner_step_n=100)

nu_model = Seq_Network([env.state_dim, 256, 128, env.action_dim], nn.ReLU())
v_model = Seq_Network([env.state_dim, 256, 128, 1], nn.ReLU())
p_model = Seq_Network([env.state_dim, 256, env.action_dim ** 2], nn.ReLU())
q_model = QModel_SphereCase(env.action_dim, env.action_min, env.action_max, nu_model, v_model, p_model)
noise = OUNoise(env.action_dim, threshold_decrease=1 / 500)

agent = NAF(env.state_dim, env.action_dim, env.action_min, env.action_max, q_model, noise,
            batch_size=128, gamma=1, tau=1e-3, q_model_lr=1e-3, learning_n_per_fit=16)

visualizer = DubinsCarVisualizer(waiting_for_show=100)
solver.go(env, agent, episode_n=501, session_n=1, session_len=1000, show=visualizer.show)
