import numpy as np

class UniformNoise:
    def __init__(self, action_dimension, threshold=1, threshold_min=0.001, threshold_decrease=0.0001):
        self.action_dimension = action_dimension
        self.threshold = threshold
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease

    def noise(self):
        return np.random.uniform(-self.threshold, self.threshold, self.action_dimension)

    def decrease(self):
        if self.threshold > self.threshold_min:
            self.threshold -= self.threshold_decrease

class ZeroNoise:
    def __init__(self):
        self.threshold = 0
    def noise(self):
        return 0

    def decrease(self):
        pass

class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, threshold=1, threshold_min=0.00001, threshold_decrease=0.00005):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.threshold = threshold
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.threshold

    def decrease(self):
        pass
        if self.threshold > self.threshold_min:
            self.threshold -= self.threshold_decrease