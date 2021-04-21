import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from copy import deepcopy
from models.Utilities.LinearTransformations import transform_interval


class QModel(nn.Module):
    def __init__(self, action_dim, action_min, action_max,
                 mu_model, p_model, v_model):
        super().__init__()
        self.action_dim = action_dim
        self.action_min = torch.FloatTensor(action_min)
        self.action_max = torch.FloatTensor(action_max)
        self.p_model = p_model
        self.mu_model = mu_model
        self.v_model = v_model
        self.tril_mask = torch.tril(torch.ones(
            action_dim, action_dim), diagonal=-1).unsqueeze(0)
        self.diag_mask = torch.diag(torch.diag(
            torch.ones(action_dim, action_dim))).unsqueeze(0)

    def forward(self, state, action):
        L = self.p_model(state).view(-1, self.action_dim, self.action_dim)
        L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
        P = torch.bmm(L, L.transpose(2, 1))
        mu = transform_interval(self.mu_model(state), self.action_min, self.action_max)
        action_mu = (action - mu).unsqueeze(2)
        A = -0.5 * \
            torch.bmm(torch.bmm(action_mu.transpose(2, 1), P),
                      action_mu)[:, :, 0]
        return A + self.v_model(state)


class MuModel(nn.Module):
    def __init__(self, nu_model):
        super().__init__()
        self.nu_model = nu_model
        self.tanh = nn.Tanh()
        return None

    def forward(self, state):
        nu = self.nu_model(state)
        return self.tanh(nu)


class QModel_BoundedCase(nn.Module):
    def __init__(self, action_dim, action_min, action_max,
                 nu_model, mu_model, v_model, p_model):
        super().__init__()

        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max

        self.nu_model = nu_model
        self.mu_model = mu_model
        self.v_model = v_model
        self.p_model = p_model

    def forward(self, state, action):
        nu = self.nu_model(state)
        mu = self.mu_model(state)
        p = self.p_model(state)
        A = - 0.5 * torch.exp(p) * (action - mu) * (action + mu - 2 * nu)
        return A + self.v_model(state)


class QModel_SphereCase(nn.Module):
    def __init__(self, action_dim, action_min, action_max,
                 nu_model, v_model, p_model):
        super().__init__()

        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max

        self.nu_model = nu_model
        self.mu_model = MuModel(nu_model)
        self.v_model = v_model
        self.p_model = p_model

    def forward(self, state, action):
        nu = self.nu_model(state)
        mu = self.mu_model(state)
        p = self.p_model(state)
        A = - 0.5 * torch.exp(p) * (action - mu) * (action + mu - 2 * nu)
        return A + self.v_model(state)


class QModel_SphereCase_RBased(nn.Module):
    def __init__(self, action_dim, action_min, action_max,
                 nu_model, v_model, beta, dt):
        super().__init__()

        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max

        self.nu_model = nu_model
        self.mu_model = MuModel(nu_model)
        self.v_model = v_model
        self.beta = beta
        self.dt = dt

    def forward(self, state, action):
        nu = self.nu_model(state)
        mu = self.mu_model(state)
        A = - self.dt * self.beta * (action - mu) * (action + mu - 2 * nu)
        return A + self.v_model(state)


class NAF:
    def __init__(self, state_dim, action_dim, action_min, action_max, q_model, noise,
                 batch_size=200, gamma=0.9999, tau=1e-3, q_model_lr=1e-4, learning_n_per_fit=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = action_max
        self.action_min = action_min

        self.q_model = q_model
        self.opt = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)
        self.loss = nn.MSELoss()
        self.q_target = deepcopy(self.q_model)
        self.tau = tau
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.learning_n_per_fit = learning_n_per_fit

        self.batch_size = batch_size
        self.noise = noise

    def get_action(self, state):
        state = torch.FloatTensor(state)
        mu_value = self.q_model.mu_model(state).detach().numpy()
        noise = self.noise.noise()
        action = mu_value + noise
        action = transform_interval(action, self.action_min, self.action_max)
        return np.clip(action, self.action_min, self.action_max)

    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * original_param.data)

    def add_to_memory(self, sessions):
        for session in sessions:
            session_len = len(session['actions'])
            for i in range(session_len):
                self.memory.append([session['states'][i],
                                    session['actions'][i],
                                    session['rewards'][i],
                                    session['dones'][i],
                                    session['states'][i + 1]])

    def fit(self, sessions):
        self.add_to_memory(sessions)

        if len(self.memory) >= self.batch_size:
            for _ in range(self.learning_n_per_fit):
                batch = random.sample(self.memory, self.batch_size)
                states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
                rewards = rewards.reshape(self.batch_size, 1)
                dones = dones.reshape(self.batch_size, 1)

                target = rewards + (1 - dones) * self.gamma * self.q_target.v_model(next_states).detach()
                loss = self.loss(self.q_model(states, actions), target)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.update_targets(self.q_target, self.q_model)
