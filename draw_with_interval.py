import matplotlib.pyplot as plt
import numpy as np

task = 'SimpleControl'
# task = 'DubinsCar'
# task = 'VanDerPol'
# task = 'TerminalPendulum'

labels = {"NAF": "NAF", "SPHERE": "BNAF", "SPHERE_R": "RB-BNAF", "SPHERE_R_G": "GB-BNAF"}


def plot_best_by_last_value(model, color):
    data = np.array([np.load('./Tests/' + task + '/' + model + '/' + str(i) + '.npy') for i in range(20)])
    integral = data[:, -1]
    idx = np.argpartition(integral, 5)
    x = np.arange(data.shape[1])
    best = data[idx[:3]]
    print(best[:, :1])
    print()


plot_best_by_last_value('NAF', 'b')
plot_best_by_last_value('SPHERE', 'r')
plot_best_by_last_value('SPHERE_R', 'g')
plot_best_by_last_value('SPHERE_R_G', 'y')

# plot_best_by_last_value('NAF', 'b')
# plot_best_by_last_value('SPHERE', 'r')

# plot_best_by_last_value('SPHERE', 'b')
# plot_best_by_last_value('SPHERE_R', 'r')

# plot_best_by_last_value('SPHERE_R', 'b')
# plot_best_by_last_value('SPHERE_R_G', 'r')

# plt.ylim(-4.5, 0)
# plot_sphere_r('SPHERE_R', 'b', 'RB-BNAF(Δt=0.5)')
# plot_sphere_r('SPHERE_R_DT', 'r', 'RB-BNAF(Δt=1, Δt=0.1)')
# plt.axvline(x=125, linestyle="--", color='gray')
# plt.text(x=62, y=-3.5, s='Δt=1')
# plt.text(x=187, y=-3.5, s='Δt=0.1')
#
# plt.xlabel('episodes')
# plt.ylabel('rewards')
# # plt.title(task)
# plt.legend(loc=4)
# ax = plt.gca()
# ax.set_facecolor('#eaeaf2')
# plt.grid(color='white')
# plt.show()
