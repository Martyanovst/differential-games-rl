import matplotlib.pyplot as plt
import numpy as np

def plot(model, color, label):
    data = np.array([np.load('./test/' + model + '_test/' + str(i) + '.npy') for i in range(10)])
    integral = np.trapz(data, axis=1)
    idx = np.argpartition(integral, 3)
    x = np.arange(100)
    best = data[idx[:3]]
    min = best.min(axis=0)
    mean = best.mean(axis=0)
    max = best.max(axis=0)
    plt.fill_between(x, min, max, alpha=0.1, color=color)
    plt.plot(x, mean, color=color, label=label)

plot('naf', 'b', 'NAF')
# plot('bounded', 'r', 'BOUNDED NAF')
# plot('bounded_r', 'g', 'BOUNDED R NAF')
# plot('bounded_r_g', 'y', 'BOUNDED R G NAF')
# plot('sphere', 'm', 'SPHERE NAF')
# plot('sphere_r', 'c', 'SPHERE R NAF')
# plot('sphere_r_g', 'r', 'SPHERE R G NAF')
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()
