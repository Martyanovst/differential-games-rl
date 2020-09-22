import torch
from torch import nn

from models.unlimited_naf import UnlimitedNAFAgent
from problems.boundary_problem.boundary_problem_env import BoundaryProblem
from problems.boundary_problem.optimal_agent import OptimalAgent
import matplotlib.pyplot as plt

from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network


def test_agent(env, agent):
    state = env.reset()
    total_reward = 0
    max_action = 0
    x1 = []
    x2 = []
    while not env.done:
        action = agent.get_action(state)
        if abs(action[0]) > max_action:
            max_action = abs(action[0])
        next_state, reward, _, _, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        x1.append(state[1])
        x2.append(state[2])
    plt.plot(x1, x2)

env = BoundaryProblem(-1, 3)
state_shape = 3
action_max = 1.5
action_shape = 1
episodes_n = 500
mu_model = Seq_Network([state_shape, 150, 150, action_shape], nn.Sigmoid())
p_model = Seq_Network([state_shape, 150, 150, action_shape ** 2], nn.Sigmoid())
v_model = Seq_Network([state_shape, 150, 150, 1], nn.Sigmoid())
noise = OUNoise(action_shape, threshold=1, threshold_min=0.001, threshold_decrease=0.002)
batch_size = 200
naf_agent = UnlimitedNAFAgent(mu_model, v_model, noise, state_shape, action_shape, batch_size, 0.999, env.dt)
naf_agent.noise.threshold = 0
naf_agent.Q.load_state_dict(torch.load('./result13' + str(env.initial_x1) + str(env.initial_x2)))
test_agent(env, OptimalAgent(env))
test_agent(env, naf_agent)
plt.legend(['optimal', 'naf'])
plt.show()
