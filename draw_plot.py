import matplotlib.pyplot as plt
import numpy as np

seeds = [2021, 2022, 2023]
path = './rewards/targetProblem/dt0.1/'
X = np.arange(2000)


def plot(model, color, label):
    data = np.array([np.load(path + model + '_' + str(seed) + '.npy') for seed in seeds])
    max = data.max(axis=0)
    min = data.min(axis=0)
    mean = data.mean(axis=0)
    plt.fill_between(X, min, max, color=color, alpha=0.1)
    plt.plot(X, mean, color=color, label=label)


plot('naf', 'b', 'naf')
plot('bnaf', 'r', 'bnaf')
plot('rb-bnaf', 'g', 'rb-bnaf')
# plot('gb-bnaf', 'y', 'gb-bnaf')

# plt.ylim(-5)

# plot_best_by_last_value('NAF', 'b')
# plot_best_by_last_value('SPHERE', 'r')

# plot_best_by_last_value('SPHERE', 'b')
# plot_best_by_last_value('SPHERE_R', 'r')

# plot_best_by_last_value('SPHERE_R', 'b')
# plot_best_by_last_value('SPHERE_R_G', 'r')

# plot('SPHERE_R', 'g', 'RB-BNAF(Δt=0.5)')
# plot('SPHERE_R_DT', 'm', 'RB-BNAF(Δt=0.5, Δt=0.1)')
# plt.axvline(x=125, linestyle="--", color='gray')
# plt.text(x=62, y=-3.5, s='Δt=0.5')
# plt.text(x=187, y=-3.5, s='Δt=0.1')

plt.xlabel('episodes')
plt.ylabel('rewards')
# plt.title(task)
plt.legend(loc=4)
ax = plt.gca()
ax.set_facecolor('#eaeaf2')
plt.grid(color='white')
plt.show()
