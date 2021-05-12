import torch
import torch.nn as nn
from Agents.Utilities.LinearTransformations import transform_interval


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


class QModel_RBased(nn.Module):
    def __init__(self, action_dim, action_min, action_max,
                 mu_model, v_model, r, dt):
        super().__init__()
        self.action_dim = action_dim
        self.action_min = torch.FloatTensor(action_min)
        self.action_max = torch.FloatTensor(action_max)
        self.mu_model = mu_model
        self.v_model = v_model
        self.dt = dt
        self.r = r

    def forward(self, state, action):
        mu = transform_interval(self.mu_model(state), self.action_min, self.action_max)
        action_mu = (action - mu)
        A = -self.dt * self.r * (action_mu ** 2)
        return A + self.v_model(state)


class MuModel(nn.Module):
    def __init__(self, nu_model):
        super().__init__()
        self.nu_model = nu_model
        self.tanh = nn.Tanh()

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


class QModel_BoundedCase_RBased(nn.Module):
    def __init__(self, action_dim, action_min, action_max,
                 nu_model, mu_model, v_model, r, dt):
        super().__init__()

        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.nu_model = nu_model
        self.mu_model = mu_model
        self.v_model = v_model
        self.dt = dt
        self.r = r

    def forward(self, state, action):
        nu = self.nu_model(state)
        mu = self.mu_model(state)
        A = - self.dt * self.r * (action - mu) * (action + mu - 2 * nu)
        return A + self.v_model(state)


class QModel_BoundedCase_RG_Based(nn.Module):
    def __init__(self, action_dim, action_min, action_max,
                 mu_model, v_model, r, g, dt, batch_size):
        super().__init__()
        self.mu_model = mu_model
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.v_model = v_model
        self.dt = dt
        self.r = r
        self.g = torch.FloatTensor(g).repeat(batch_size, 1).unsqueeze(1)
        self.action_max = action_max
        self.action_dim = action_dim

    def forward(self, state, action):
        mu = self.mu_model(state)
        v = self.v_model(state)
        v.backward(torch.ones((state.shape[0], 1)))
        dv = state.grad[:, 1:].detach().unsqueeze(2)
        nu = (0.5 * (1 / self.r) * torch.bmm(self.g,
                                             dv)[:, :, 0])
        A = - self.dt * self.r * (action - mu) * (action + mu - 2 * nu)
        self.v_model.zero_grad()
        return A + self.v_model(state)


class QModel_SphereCase_RG_Based(nn.Module):
    def __init__(self, action_dim, action_min, action_max,
                 mu_model, v_model, r, g, dt, batch_size):
        super().__init__()
        self.mu_model = mu_model
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.v_model = v_model
        self.dt = dt
        self.r = r
        self.g = torch.FloatTensor(g).repeat(batch_size, 1).unsqueeze(1)
        self.action_max = action_max
        self.action_dim = action_dim
        self.tril_mask = torch.tril(torch.ones(
            action_dim, action_dim), diagonal=-1).unsqueeze(0)
        self.diag_mask = torch.diag(torch.diag(
            torch.ones(action_dim, action_dim))).unsqueeze(0)

    def forward(self, state, action):
        mu = self.mu_model(state)
        action_mu = (action - mu).unsqueeze(2)
        v = self.v_model(state)
        v.backward(torch.ones((state.shape[0], 1)))
        dv = state.grad[:, 1:].detach().unsqueeze(2)
        phi = (0.5 * (1 / self.r) * torch.bmm(self.g,
                                              dv)[:, :, 0])
        action_phi = (phi - mu).unsqueeze(2)
        A = -self.dt * self.r * \
            torch.bmm(action_mu.transpose(2, 1),
                      action_mu)[:, :, 0] + \
            2 * self.dt * self.r * torch.bmm(action_phi.transpose(2, 1),
                                             action_mu)[:, :, 0]
        self.v_model.zero_grad()
        return A + self.v_model(state)


class QModel_RG_Based(nn.Module):
    def __init__(self, action_dim, action_min, action_max,
                 v_model, r, g, dt):
        super().__init__()
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.v_model = v_model
        self.dt = dt
        self.r = r
        self.g = torch.FloatTensor(g)
        self.action_max = action_max
        self.action_dim = action_dim
        self.tril_mask = torch.tril(torch.ones(
            action_dim, action_dim), diagonal=-1).unsqueeze(0)
        self.diag_mask = torch.diag(torch.diag(
            torch.ones(action_dim, action_dim))).unsqueeze(0)

    def forward(self, state, action):
        g = self.g.repeat(state.shape[0], 1).unsqueeze(1)
        v = self.v_model(state)
        v.backward(torch.ones((state.shape[0], 1)))
        dv = state.grad[:, 1:].detach().unsqueeze(2)
        mu = (0.5 * (1 / self.r) * torch.bmm(g, dv)[:, :, 0])
        action_mu = (action - mu).unsqueeze(2)
        A = -self.dt * self.r * \
            torch.bmm(action_mu.transpose(2, 1),
                      action_mu)[:, :, 0]
        self.v_model.zero_grad()
        return A + self.v_model(state)

    def mu_model(self, state):
        v = self.v_model(state)
        v.backward()
        dv = state.grad[1:].detach()
        mu = (0.5 * (1 / self.r) * torch.matmul(self.g,
                                                dv))
        return mu
