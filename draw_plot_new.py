import matplotlib.pyplot as plt
import numpy as np

path = './rewards/BNAF_res/env_name=Pendulum_agent_name={}_dt=0.1_lr=0.001_tau=0.01_bs=64_lrpf=1_en=1000_'
env_names = ['NAF', 'BNAF', 'RB-BNAF', 'GB-BNAF', 'DDPG']

X = np.arange(991)

def plot(path, model, color, label):
    data = np.array([np.load(path.format(model) + '/attempt_' + str(attempt) + '/mean_total_rewards.npy') for attempt in range(10)])
    smoothed_data = np.zeros(data.shape)
    for epoch in range(991):
        mean_reward = np.mean(data[:, max(0, epoch - 25):epoch + 1], axis=1)
        smoothed_data[:, epoch] = mean_reward
    maximum = smoothed_data.max(axis=0)
    min = smoothed_data.min(axis=0)
    mean = smoothed_data.mean(axis=0)
    plt.fill_between(X, min, maximum, color=color, alpha=0.1)
    plt.plot(X, mean, color=color, label=label)

plot(path, 'NAF', 'b', 'NAF')
plot(path, 'BNAF', 'r', 'BNAF')
plot(path, 'RB-BNAF', 'g', 'RB-BNAF')
plot(path, 'GB-BNAF', 'm', 'GB-BNAF')
plot(path, 'DDPG', 'y', 'DDPG')

# plt.ylim(-6)

plt.xlabel('episodes')
plt.ylabel('rewards')
plt.title('Pendulum')
plt.legend(loc=4)
ax = plt.gca()
ax.set_facecolor('#eaeaf2')
plt.grid(color='white')
plt.show()