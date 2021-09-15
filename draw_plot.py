import matplotlib.pyplot as plt
import numpy as np


def plot(model, color, label):
    data = np.load('./rewards/earthOrbit/' + model + '.npy')
    plt.plot(np.arange(1000), data, color=color, label=label)

plot('naf', 'b', 'naf')
plot('bnaf', 'r', 'bnaf')
plot('rb-bnaf', 'g', 'rb-bnaf')
plot('gb-bnaf', 'y', 'gb-bnaf')

# plot_best_by_last_value('NAF', 'b')
# plot_best_by_last_value('SPHERE', 'r')

# plot_best_by_last_value('SPHERE', 'b')
# plot_best_by_last_value('SPHERE_R', 'r')

# plot_best_by_last_value('SPHERE_R', 'b')
# plot_best_by_last_value('SPHERE_R_G', 'r')

plt.ylim(-10, 0)
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