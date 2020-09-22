import torch
from torch import nn

from models.simple_naf import SimpleNaf
import matplotlib.pyplot as plt
from models.unlimited_naf import UnlimitedNAFAgent
from problems.regulator_problem.optimal_agent import OptimalAgent
from problems.regulator_problem.regulator_problem_env import RegulatorProblem
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network


def agent_play(env, agent, title):
    state = env.reset()
    total_reward = 0
    xs = []
    us = []
    terminal_time = 500
    done = False
    step = 0
    while not done:
        action = agent.get_action(state)
        xs.append(env.t)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        us.append(action[0])
        state = next_state
        done = step >= terminal_time
        step += 1

    print(total_reward)
    return xs, us


state_shape = 5
action_shape = 1

mu_model = Seq_Network([state_shape, 150, 150, 150, action_shape], nn.ReLU())
p_model = Seq_Network([state_shape, 150, 150, 150, action_shape ** 2], nn.ReLU())
v_model = Seq_Network([state_shape, 150, 150, 150, 1], nn.ReLU())
noise = OUNoise(action_shape, threshold=1, threshold_min=0.001, threshold_decrease=0.001)
batch_size = 200
agent = SimpleNaf(mu_model, v_model, noise, state_shape, action_shape, batch_size, 1)
agent.noise.threshold = 0
agent.Q.load_state_dict(torch.load('./result13'))
env = RegulatorProblem()
xs, optu = agent_play(env, OptimalAgent(), 'optimal agent')
xs, u = agent_play(env, agent, 'naf agent')
plt.plot(xs, optu)
plt.plot(xs, u)
plt.legend(['optimal', 'naf'])
plt.show()
