import numpy as np
import matplotlib.pyplot as plt
from unequal_game_env import UnequalGame
from optimal_agents import OptimalUAgent, OptimalVAgent

env = UnequalGame()
u_agent = OptimalUAgent(env)

state = env.reset()
done = False
total_reward = 0
u_actions = []
v_actions = []
states = []

while not done:
    states.append(state)
    u_action = u_agent.get_action(state)
    v_action = np.sin(500 * state[0])
    u_actions.append(u_action)
    v_actions.append(v_action)
    state, reward, done, _ = env.step(u_action, v_action)
    total_reward += reward

print(total_reward)

plt.plot([state[1] for state in states])
plt.show()
plt.plot(u_actions)
plt.show()
plt.plot(v_actions)
plt.show()