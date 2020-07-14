from problems.boundary_problem.boundary_problem_env import BoundaryProblem
from problems.boundary_problem.optimal_agent import OptimalAgent
import matplotlib.pyplot as plt

env = BoundaryProblem(-1, 3)
agent = OptimalAgent(env)
state = env.reset()
total_reward = 0
max_action = 0
ts = []
us = []
while not env.done:
    action = agent.get_action(state)
    if abs(action[0]) > max_action:
        max_action = abs(action[0])
    next_state, reward, done, _ = env.step(action)
    total_reward += reward

    state = next_state
print(total_reward, max_action, state[1], state[2])
plt.plot(ts, us)
plt.show()
