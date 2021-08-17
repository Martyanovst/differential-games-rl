import torch

from models.naf import NAF

from models.ou_noise import OUNoise, load_noise
import torch.nn as nn

from models.q_models import QModel, QModel_Bounded, QModel_Bounded_RewardBased, QModel_Bounded_GradientBased
from models.sequential_network import Seq_Network


class AgentGenerator:
    def __init__(self, env, train_cfg, model_cfg):
        self.dt = env.dt
        self.g = env.g
        self.epoch_num = train_cfg['epoch_num']
        self.batch_size = train_cfg['batch_size']
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_max = env.action_max
        self.action_min = env.action_min
        self.beta = env.beta
        self.r = env.r
        self.lr = model_cfg.get('lr', 1e-3)
        self.gamma = model_cfg.get('gamma', 1)
        self.model_type = model_cfg['model_name']
        self.noise_min = 1e-3

    def _naf_(self, q_model):
        noise = OUNoise(self.action_dim, threshold_min=self.noise_min,
                        threshold_decrease=self.noise_min ** (1 / self.epoch_num))
        return NAF(self.action_min, self.action_max, q_model, noise,
                   batch_size=self.batch_size, gamma=self.gamma, tau=1e-3, q_model_lr=self.lr)

    def _generate_naf(self):
        mu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        p_model = Seq_Network([self.state_dim, 256, self.action_dim ** 2], nn.ReLU())
        q_model = QModel(self.action_dim, self.action_min, self.action_max, mu_model, p_model, v_model, self.dt)
        return self._naf_(q_model)

    def _generate_b_naf(self):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        p_model = Seq_Network([self.state_dim, 256, self.action_dim ** 2], nn.ReLU())
        q_model = QModel_Bounded(self.action_dim, self.action_min, self.action_max, nu_model, v_model, p_model, self.dt)
        return self._naf_(q_model)

    def _generate__b_naf_reward_based(self):
        nu_model = Seq_Network([self.state_dim, 256, 128, self.action_dim], nn.ReLU())
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_Bounded_RewardBased(self.action_dim, self.action_min, self.action_max, nu_model,
                                             v_model,
                                             self.beta, self.dt)
        return self._naf_(q_model)

    def _generate_b_naf_gradient_based(self):
        v_model = Seq_Network([self.state_dim, 256, 128, 1], nn.ReLU())
        q_model = QModel_Bounded_GradientBased(self.action_dim, self.action_min, self.action_max, v_model, r=self.r,
                                               g=self.g, dt=self.dt)
        return self._naf_(q_model)

    def load(self, path):
        state_dict = torch.load(path)
        if state_dict['q-model']['model-name'] == 'q-model':
            model = self.generate_naf()
        elif state_dict['q-model']['model-name'] == 'q-model-bounded':
            model = self.generate_b_naf()
        elif state_dict['q-model']['model-name'] == 'q-model-bounded-reward-based':
            model = self.generate__b_naf_reward_based()
        else:
            model = self.generate_b_naf_gradient_based()

        model.q_model.load_state_dict(state_dict['q-model'])
        model.noise = load_noise(state_dict['noise'])
        model.action_min = state_dict['action_min']
        model.action_max = state_dict['action_max']
        model.tau = state_dict['tau']
        model.lr = state_dict['lr']
        model.gamma = state_dict['gamma']
        model.batch_size = state_dict['batch_size']
        return model

    def generate(self):
        if self.model_type == 'naf':
            return self._generate_naf()
        elif self.model_type == 'bnaf':
            return self._generate_b_naf()
        elif self.model_type == 'rb-bnaf':
            return self._generate__b_naf_reward_based()
        elif self.model_type == 'gb-bnaf':
            return self._generate_b_naf_gradient_based()
