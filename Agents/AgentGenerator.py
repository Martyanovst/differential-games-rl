from Agents.NAF import NAF
from Agents.QModels import QModel_SphereCase, QModel, QModel_BoundedCase, QModel_SphereCase_RBased, QModel_RBased, \
    QModel_BoundedCase_RBased, QModel_BoundedCase_RG_Based, QModel_RG_Based
from Agents.Utilities.Noises import OUNoise
import torch.nn as nn
from Agents.Utilities.SequentialNetwork import Seq_Network


class AgentGenerator:
    def __init__(self, env, batch_size, episode_n=100, noise_min=1e-3):
        self.dt = env.dt
        self.g = env.g
        self.episode_n = episode_n
        self.batch_size = batch_size
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_max = env.action_max
        self.action_min = env.action_min
        self.beta = env.beta
        self.r = env.r
        self.noise_min = noise_min

    def _naf_(self, q_model):
        noise = OUNoise(self.action_dim, threshold=1, threshold_min=self.noise_min,
                        threshold_decrease=self.noise_min ** (1 / self.episode_n))
        return NAF(self.state_dim, self.action_dim, self.action_min, self.action_max, q_model, noise,
                   batch_size=128, gamma=1, tau=1e-3, q_model_lr=1e-3, learning_n_per_fit=16)

    def generate_naf(self):
        mu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        p_model = Seq_Network([self.state_dim, 256, self.action_dim ** 2], nn.ReLU())
        q_model = QModel(self.action_dim, self.action_min, self.action_max, mu_model, p_model, v_model)
        return self._naf_(q_model)

    def generate_naf_r_based(self):
        mu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_RBased(self.action_dim, self.action_min, self.action_max, mu_model,
                                v_model,
                                self.r, self.dt)
        return self._naf_(q_model)

    def generate_naf_sphere_case(self):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        p_model = Seq_Network([self.state_dim, 256, self.action_dim ** 2], nn.ReLU())
        q_model = QModel_SphereCase(self.action_dim, self.action_min, self.action_max, nu_model, v_model, p_model)
        return self._naf_(q_model)

    def generate_naf_bounded_case(self):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        mu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        p_model = Seq_Network([self.state_dim, 256, self.action_dim ** 2], nn.ReLU())
        q_model = QModel_BoundedCase(self.action_dim, self.action_min, self.action_max, nu_model, mu_model, p_model,
                                     v_model)
        return self._naf_(q_model)

    def generate_naf_sphere_case_r_based(self):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_SphereCase_RBased(self.action_dim, self.action_min, self.action_max, nu_model, v_model,
                                           self.beta, self.dt)
        return self._naf_(q_model)

    def generate_naf_bounded_case_r_based(self):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        mu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_BoundedCase_RBased(self.action_dim, self.action_min, self.action_max, nu_model, mu_model,
                                            v_model,
                                            self.r, self.dt)
        return self._naf_(q_model)

    def generate_naf_bounded_case_rg_based(self):
        mu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_BoundedCase_RG_Based(self.action_dim, self.action_min, self.action_max, mu_model,
                                              v_model,
                                              r=self.r,
                                              g=self.g,
                                              dt=self.dt,
                                              batch_size=self.batch_size)
        return self._naf_(q_model)

    def generate_naf_rg_based(self):
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_RG_Based(self.action_dim, self.action_min, self.action_max,
                                  v_model,
                                  r=self.r,
                                  g=self.g,
                                  dt=self.dt,
                                  batch_size=self.batch_size)
        return self._naf_(q_model)

    def generate_naf_sphere_case_rg_based(self):
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_RG_Based(self.action_dim, self.action_min, self.action_max,
                                  v_model,
                                  r=self.r,
                                  g=self.g,
                                  dt=self.dt,
                                  batch_size=self.batch_size)
        return self._naf_(q_model)
