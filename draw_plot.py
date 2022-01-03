import matplotlib.pyplot as plt
import numpy as np

seeds = [2021, 2022, 2023, 2024, 2025]
path = './rewards/targetProblem/'
X = np.arange(1000)


def plot(model, color, label):
    data = np.array([np.load(path + model + str(seed) + '.npy') for seed in seeds])
    max = data.max(axis=0)
    min = data.min(axis=0)
    mean = data.mean(axis=0)
    plt.fill_between(X, min, max, color=color, alpha=0.1)
    plt.plot(X, mean, color=color, label=label)


plot('naf', 'b', 'naf')
plot('bnaf', 'r', 'bnaf')
plot('rb-bnaf', 'g', 'rb-bnaf')
plot('gb-bnaf', 'y', 'gb-bnaf')
plot('ddpg', 'm', 'ddpg')

# plt.ylim(-6)

plt.xlabel('episodes')
plt.ylabel('rewards')
# plt.title(task)
plt.legend(loc=4)
ax = plt.gca()
ax.set_facecolor('#eaeaf2')
plt.grid(color='white')
plt.show()