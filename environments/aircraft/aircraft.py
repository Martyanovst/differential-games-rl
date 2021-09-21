import numpy as np
from numpy.linalg import norm


class AircraftLanding:
    def __init__(self, action_radius=np.array([32.5, 10, 10, 10]),
                 initial_state=np.array([0, -1000, 67.13, 60, -3.13, 0, 0,
                                         2.94 * np.pi / 180, 0, 0, 0, 0, 0,
                                         124500, 0, 0, 0]),
                 terminal_time=15, dt=0.05, inner_step_n=10):

        self.state_dim = 17
        self.action_dim = 4
        self.action_radius = action_radius
        self.action_min = - self.action_radius
        self.action_max = + self.action_radius

        self.beta = 1
        self.r = 1
        self.g = 1

        self.initial_state = initial_state
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.state = self.reset()

        self.load_constants()

    def load_constants(self):
        '''
        Постоянные в модели посадки самолета
        '''
        self.m = 75000  # масса самолета, кг
        self.s = 201  # площадь крыла, м^2
        self.l = 37.55  # размах крыла, м
        self.b = 5.285  # средняя аэродинамическая хорда, м

        self.sigma = 1.72 / 180 * np.pi  # угол установки двигателей, рад
        self.delta_st = -1.26  # угол установки стабилизатора руля высоты на хвостовом оперении
        self.k = 4  # коэффициент усиления в динамике рулей самолета, c^-1

        # Моменты инерции, кг*м^2:
        self.Ix = 2.5e6
        self.Iy = 7.5e6
        self.Iz = 6.5e6
        self.Ixy = 0.5e6
        self.J = self.Ix * self.Iy - self.Ixy ** 2

        self.g = 9.81  # ускорение свободного падения, м/с^2
        self.rho = 1.207  # плотность воздуха, кг/м^3

    def f(self, state, u):
        t, x, Vx, y, Vy, z, Vz, theta, wz, psi, wy, gamma, wx, p, delta_e, delta_r, delta_a = state
        delta_ps, delta_es, delta_rs, delta_as = u + np.array([79.5, 0, 0, 0])

        w_xg, w_yg, w_zg = np.array([0, 0, 0])  # скорости ветра (постоянная помеха)

        # модуль (длина) вектора воздушной скорости
        V = np.sqrt((Vx - w_xg) ** 2 + (Vy - w_yg) ** 2 + (Vz - w_zg) ** 2)

        # скоростной напор
        q = 1 / 2 * self.rho * V ** 2

        # углы скольжения и атаки
        beta_rad = np.arcsin(
            ((Vx - w_xg) * (np.sin(psi) * np.cos(gamma) + np.cos(psi) * np.sin(theta) * np.sin(gamma)) -
             (Vy - w_yg) * np.cos(theta) * np.sin(gamma) +
             (Vz - w_zg) * (np.cos(psi) * np.cos(gamma) - np.sin(psi) * np.sin(theta) * np.sin(gamma))) / V)

        alpha_rad = np.arcsin(
            -((Vx - w_xg) * (np.sin(psi) * np.sin(gamma) - np.cos(psi) * np.sin(theta) * np.cos(gamma)) +
              (Vy - w_yg) * np.cos(theta) * np.cos(gamma) +
              (Vz - w_zg) * (np.cos(psi) * np.sin(gamma) + np.sin(psi) * np.sin(theta) * np.cos(gamma))) / (
                    V * np.cos(beta_rad)))

        alpha, beta = alpha_rad * 180 / np.pi, beta_rad * 180 / np.pi

        # расчет аэродинамических коэффициентов сил и моментов
        cx = (0.21 + 0.004 * alpha + 0.47e-3 * alpha ** 2) * np.cos(alpha_rad) - (
                0.65 + 0.09 * alpha + 0.003 * delta_e) * np.sin(alpha_rad)
        cy = (0.21 + 0.004 * alpha + 0.47e-3 * alpha ** 2) * np.sin(alpha_rad) + (
                0.65 + 0.09 * alpha + 0.003 * delta_e) * np.cos(alpha_rad)
        cz = -0.0115 * beta - (0.0034 - 6e-5 * alpha) * delta_r

        mx = (-0.0035 - 0.0001 * alpha) * beta + (
                -0.0005 + 0.00003 * alpha) * delta_r - 0.0004 * delta_a + self.l * np.pi / (2 * V * 180) * (
                     (-0.61 + 0.004 * alpha) * wx + (-0.3 - 0.012 * alpha) * wy)
        my = (-0.004 - 0.00005 * alpha) * beta + (-0.00135 + 0.000015 * alpha) * delta_r + self.l * np.pi / (
                2 * V * 180) * (0.015 * alpha * wx + (-0.21 - 0.005 * alpha) * wy)
        mz = 0.033 - 0.017 * alpha - 0.013 * delta_e + 0.047 * self.delta_st - 1.29 * wz

        # аэродинамические моменты
        Mx = q * self.s * self.l * mx
        My = q * self.s * self.l * my
        Mz = q * self.s * self.b * mz

        # собираем апдейт вектора состояния
        state_update = np.ones(17)

        sin_sigma = np.sin(self.sigma)
        cos_sigma = np.cos(self.sigma)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        sin_gamma = np.sin(gamma)
        cos_gamma = np.cos(gamma)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # динамика поступательного движения
        state_update[1] = Vx
        state_update[2] = ((p * cos_sigma - q * self.s * cx) * cos_psi * np.cos(theta) +
                           (p * sin_sigma + q * self.s * cy) * (sin_psi * sin_gamma - cos_psi * cos_gamma * sin_theta) +
                           q * self.s * cz * (sin_psi * cos_gamma + cos_psi * sin_theta * sin_gamma)) / self.m

        state_update[3] = Vy
        state_update[4] = ((p * cos_sigma - q * self.s * cx) * sin_theta +
                           (p * sin_sigma + q * self.s * cy) * cos_theta * cos_gamma -
                           q * self.s * cz * cos_theta * sin_gamma) / self.m - self.g

        state_update[5] = Vz
        state_update[6] = (-(p * cos_sigma - q * self.s * cx) * sin_psi * cos_theta +
                           (p * sin_sigma + q * self.s * cy) * (cos_psi * sin_gamma + sin_psi * cos_gamma * sin_theta) +
                           q * self.s * cz * (cos_psi * cos_gamma - sin_psi * sin_theta * sin_gamma)) / self.m

        # динамика вращательного движения
        state_update[7] = wz * cos_gamma + wy * sin_gamma  # тангаж
        state_update[8] = (self.Ixy * (wx ** 2 - wy ** 2) - (self.Iy - self.Ix) * wx * wy + Mz) / self.Iz

        state_update[9] = (wy * cos_gamma + wz * sin_gamma) / cos_theta  # рысканье
        state_update[10] = ((self.Iy - self.Iz) * self.Ixy * wy * wz + (self.Iy - self.Ix) * self.Ix * wx * wz +
                            self.Ix * My + self.Ixy * Mx + self.Ixy * wz * (self.Ix * wy - self.Ixy * wx)) / self.J

        state_update[11] = wx - (wy * cos_gamma - wz * sin_gamma) * np.tan(theta)  # крен
        state_update[12] = ((self.Iy - self.Iz) * self.Iy * wy * wz + (self.Iz - self.Ix) * self.Ixy * wx * wz +
                            self.Iy * Mx + self.Ixy * My + self.Ixy * wz * (self.Ixy * wy - self.Iy * wx)) / self.J

        # динамика тяги
        state_update[13] = -p + 3538 * (delta_ps - 41.3)

        # динамика рулевых органов
        state_update[14] = self.k * (delta_es - delta_e)  # руль высоты
        state_update[15] = self.k * (delta_rs - delta_r)  # руль направления
        state_update[16] = self.k * (delta_as - delta_a)  # руль элеронов

        return state_update

    def action_scaling(self, action, action_radius, action_min, action_max):
        action = np.clip(action, action_min, action_max)
        return action
        # return action*action_radius / max(norm(action/action_radius),1)

    def reset(self):
        self.state = self.initial_state
        return self.state

    def get_state_obs(self):
        return ""

    def step(self, action):
        action = self.action_scaling(action, self.action_radius, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            k1 = self.f(self.state, action)
            k2 = self.f(self.state + k1 * self.inner_dt / 2, action)
            k3 = self.f(self.state + k2 * self.inner_dt / 2, action)
            k4 = self.f(self.state + k3 * self.inner_dt, action)
            self.state = self.state + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6

        reward = 0
        done = False

        if self.state[0] >= self.terminal_time:
            reward = -norm(self.state[[1, 3]])
            done = True

        print(self.state)
        return self.state, reward, done, None
