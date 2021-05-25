import matplotlib.pyplot as plt
import numpy as np

# task = 'SimpleControl'
# task = 'DubinsCar'
# task = 'VanDerPol'
task = 'TerminalPendulum'


def plot_best_by_last_value(model, color):
    plt.ylim(-7.5, 0)
    data = np.array([np.load('./Tests/' + task + '/' + model + '/' + str(i) + '.npy') for i in range(20)])
    integral = data[:, -1]
    idx = np.argpartition(integral, 5)
    x = np.arange(data.shape[1])
    best = data[idx[:3]]
    min = -best.min(axis=0)
    mean = -best.mean(axis=0)
    max = -best.max(axis=0)
    plt.fill_between(x, min, max, alpha=0.1, color=color)
    plt.plot(x, mean, color=color, label=model)


def plot(model, color):
    data = np.array([np.load('./Tests/' + task + '/' + model + '/' + str(i) + '.npy') for i in range(10)])
    integral = np.trapz(data, axis=1)
    idx = np.argpartition(integral, 3)
    x = np.arange(500)
    best = data[idx[:3]]
    min = best.min(axis=0)
    mean = best.mean(axis=0)
    max = best.max(axis=0)
    plt.fill_between(x, min, max, color=color)
    plt.plot(x, mean, color=color, label=model)


# plot_best_by_last_value('NAF', 'b')
# plot_best_by_last_value('SPHERE', 'b')
plot_best_by_last_value('SPHERE_R', 'b')
# plot_best_by_last_value('SPHERE_R_DT', 'r')
# plt.axvline(x=250, linestyle="--", color='gray')
# plt.text(x=125, y=-5, s='dt=0.5')
# plt.text(x=375, y=-5, s='dt=0.1')
plot_best_by_last_value('SPHERE_R_G', 'r')
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.title(task)
plt.legend(loc=4)
ax = plt.gca()
ax.set_facecolor('#eaeaf2')
plt.grid(color='white')
plt.show()
