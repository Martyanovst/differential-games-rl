import matplotlib.pyplot as plt
import numpy as np

def plot(model, color):
    data = np.array([np.load('./Tests/DubinsCar/' + model + '/' + str(i) + '.npy') for i in range(10)])
    integral = np.trapz(data, axis=1)
    idx = np.argpartition(integral, 3)
    x = np.arange(1000)
    best = data[idx[:3]]
    min = best.min(axis=0)
    mean = best.mean(axis=0)
    max = best.max(axis=0)
    plt.fill_between(x, min, max, alpha=0.1, color=color)
    plt.plot(x, mean, color=color, label=model)

plot('NAF', 'b')
plot('BOUNDED', 'g')
plot('SPHERE', 'r')
# plot('bounded_r', 'y', 'BOUNDED R NAF')
# plot('bounded_r_g', 'y', 'BOUNDED R G NAF')
# plot('sphere', 'm', 'SPHERE NAF')
# plot('sphere_r', 'c', 'SPHERE R NAF')
# plot('sphere_r_g', 'r', 'SPHERE R G NAF')
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()
