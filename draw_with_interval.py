import matplotlib.pyplot as plt
import numpy as np

def plot(model, color):
    data = np.array([np.load('./Tests/DubinsCar/' + model + '/' + str(i) + '.npy') for i in range(10)])
    integral = np.trapz(data, axis=1)
    idx = np.argpartition(integral, 3)
    x = np.arange(500)
    best = data[idx[:3]]
    min = best.min(axis=0)
    mean = best.mean(axis=0)
    max = best.max(axis=0)
    plt.fill_between(x, min, max, alpha=0.1, color=color)
    plt.plot(x, mean, color=color, label=model)

plot('NAF', 'b')
plot('NAF_R', 'y')
plot('BOUNDED', 'g')
plot('SPHERE', 'r')
plot('BOUNDED_R', 'y')
plot('SPHERE_R', 'm')
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()
