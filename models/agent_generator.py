from models.naf import NAF

from models.ou_noise import OUNoise
import torch.nn as nn

from models.q_models import QModel, QModel_Bounded, QModel_Bounded_RewardBased, QModel_Bounded_GradientBased
from models.sequential_network import Seq_Network


class AgentGenerator:
    def __init__(self, env, batch_size, epoch_num=100, gamma=1, lr=1e-3):
        self.dt = env.dt
        self.g = env.g
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_max = env.action_max
        self.action_min = env.action_min
        self.beta = env.beta
        self.r = env.r
        self.lr = lr
        self.gamma = gamma
        self.noise_min = 1e-3

    def _naf_(self, q_model):
        noise = OUNoise(self.action_dim, threshold_min=self.noise_min,
                        threshold_decrease=self.noise_min ** (1 / self.epoch_num))
        return NAF(self.action_min, self.action_max, q_model, noise,
                   batch_size=128, gamma=1, tau=1e-3, q_model_lr=self.lr)

    def generate_naf(self):
        mu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        p_model = Seq_Network([self.state_dim, 256, self.action_dim ** 2], nn.ReLU())
        q_model = QModel(self.action_dim, self.action_min, self.action_max, mu_model, p_model, v_model)
        return self._naf_(q_model)

    def generate_b_naf(self):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        p_model = Seq_Network([self.state_dim, 256, self.action_dim ** 2], nn.ReLU())
        q_model = QModel_Bounded(self.action_dim, self.action_min, self.action_max, nu_model, v_model, p_model)
        return self._naf_(q_model)

    def generate__b_naf_reward_based(self):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_Bounded_RewardBased(self.action_dim, self.action_min, self.action_max, nu_model, v_model,
                                             self.beta, self.dt)
        return self._naf_(q_model)

    def generate_naf_bounded_gradient_based(self):
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_Bounded_GradientBased(self.action_dim, self.action_min, self.action_max, v_model, r=self.r,
                                               g=self.g, dt=self.dt)
        return self._naf_(q_model)
