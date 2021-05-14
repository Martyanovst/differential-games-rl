import matplotlib.pyplot as plt
import numpy as np

# task = 'DubinsCar'
task = 'VanDerPol'


def plot_best_by_last_value(model, color):
    data = np.array([np.load('./Tests/' + task + '/' + model + '/' + str(i) + '.npy') for i in range(20)])
    integral = data[:, -1]
    idx = np.argpartition(integral, 5)
    x = np.arange(data.shape[1])
    best = data[idx[:3]]
    min = best.min(axis=0)
    mean = best.mean(axis=0)
    max = best.max(axis=0)
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
    plt.fill_between(x, min, max, alpha=0.1, color=color)
    plt.plot(x, mean, color=color, label=model)


plot_best_by_last_value('NAF', 'b')
# plot_best_by_last_value('NAF_R', 'y')
# plot_best_by_last_value('BOUNDED', 'g')
# plot_best_by_last_value('BOUNDED_R', 'y')
# plot_best_by_last_value('SPHERE', 'r')
# plot_best_by_last_value('SPHERE_R', 'g')
plot_best_by_last_value('BOUNDED_R_G', 'g')
plot_best_by_last_value('SPHERE_R_G', 'r')
plt.xlabel('episode')
plt.ylabel('reward')
plt.title(task)
plt.legend()
plt.show()
