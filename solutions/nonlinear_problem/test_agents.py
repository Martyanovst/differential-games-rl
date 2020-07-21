import matplotlib as mpl
import torch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from models.unlimited_naf import UnlimitedNAFAgent
from problems.nonlinear_problem.nonlinear_problem_env import NonlinearProblem
from problems.nonlinear_problem.optimal_agent import OptimalAgent
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network


def agent_play(env, agent, title):
    state = env.reset()
    total_reward = 0
    x1s = []
    x2s = []
    us = []
    terminal_time = 20
    done = False
    step = 0
    while not done:
        action = agent.get_action(state)
        x1s.append(state[0])
        x2s.append(state[1])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        us.append(action[0])
        state = next_state
        done = env.t >= terminal_time
        step += 1

    mpl.rcParams['legend.fontsize'] = 10

    return x1s, x2s, us


state_shape = 2
action_shape = 1
episodes_n = 1000

mu_model = Seq_Network([state_shape, 100, 100, 100, action_shape], nn.Sigmoid())
p_model = Seq_Network([state_shape, 100, 100, 100, action_shape ** 2], nn.Sigmoid())
v_model = Seq_Network([state_shape, 100, 100, 100, 1], nn.Sigmoid())
noise = OUNoise(action_shape, threshold=1, threshold_min=0.001, threshold_decrease=0.001)
batch_size = 200
agent = UnlimitedNAFAgent(mu_model, p_model, v_model, noise, state_shape, action_shape, batch_size, 1)
agent.noise.threshold = 0
agent.Q.load_state_dict(torch.load('./result'))
env = NonlinearProblem()

optx1, optx2, optu = agent_play(env, OptimalAgent(), 'optimal agent')
x1, x2, u = agent_play(env, agent, 'naf agent')

X1 = np.arange(-1, 1, 0.1)
X2 = np.arange(-1, 1, 0.1)
X1, X2 = np.meshgrid(X1, X2)
Z = -X1 * X2
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
