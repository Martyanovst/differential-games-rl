import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.bounded.bounded_naf import Bounded_NAF
from problems.dubins_car.dubins_car_env import DubinsCar
from problems.simple_control_problem.simple_control_problem_env import SimpleControlProblem
from utilities.noises import OUNoise
from utilities.sequentialNetwork import Seq_Network

env = DubinsCar(dt=2)
state_shape = env.state_dim
action_shape = env.action_dim
action_max = env.action_max[0]
action_min = env.action_min[0]
episodes_n = 1000
epsilon_min = 0.00000000001
epsilon = 1

mu_model = Seq_Network([state_shape, 128, 128, action_shape], nn.ReLU())
phi_model = Seq_Network([state_shape, 128, 128, action_shape], nn.ReLU())
p_model = Seq_Network([state_shape, 128, 128, action_shape ** 2], nn.ReLU())
v_model = Seq_Network([state_shape, 128, 128, 1], nn.ReLU())
noise = OUNoise(action_shape, threshold=epsilon, threshold_min=epsilon_min,
                threshold_decrease=(epsilon_min / epsilon) ** (1 / episodes_n))
batch_size = 128
agent = Bounded_NAF(mu_model, p_model, v_model, phi_model, noise, state_shape, action_shape, action_max, action_min=action_min,
                    batch_size=batch_size,
                    gamma=1)


def play_and_learn(env):
    total_reward = 0
    state = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        agent.fit(state, action, -reward, done, next_state)
        state = next_state
        step += 1
    agent.noise.decrease()
    return total_reward


def agent_play(env, agent):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward


rewards = np.zeros(episodes_n)
mean_rewards = np.zeros(episodes_n)
for episode in range(episodes_n):
    reward = play_and_learn(env)
    rewards[episode] = reward
    mean_reward = np.mean(rewards[max(0, episode - 25):episode + 1])
    mean_rewards[episode] = mean_reward
    print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f" % (
        episode, agent.noise.threshold, rewards[episode], mean_reward))

agent.noise.threshold = 0
reward = agent_play(env, agent)
plt.plot(range(episodes_n), mean_rewards)
plt.title('Динамика показателя качества в процессе обучения')
plt.legend(['Bounded NAF'])
plt.xlabel('Эпизод')
plt.ylabel('Показатель качества')
plt.show()
torch.save(agent.Q.state_dict(), './test/bounded')
np.save('./test/bounded', mean_rewards)
