from collections import deque

import numpy as np

from Resolvers import OneAgentSolver


class AgentTestingModule:
    def __init__(self, env):
        self.env = env
        self.rewards = np.zeros(0)
        self.mean_rewards = np.zeros(0)

    def __callback__(self, env, agent, episode, sessions):
        total_reward = np.sum(sessions['rewards'])
        self.rewards[episode] = total_reward
        mean_reward = np.mean(self.rewards[max(0, episode - 25):episode + 1])
        self.mean_rewards[episode] = mean_reward
        print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f" % (
            episode, agent.noise.threshold, -total_reward, -mean_reward))

    def __reset__(self, episode_len):
        self.rewards = np.zeros(episode_len)
        self.mean_rewards = np.zeros(episode_len)

    def test_agent(self, agent_gen, episode_n, session_len, epoch_n, dt_array, path=None):
        for epoch in range(epoch_n):
            self.__reset__(episode_n * len(dt_array))
            print('\nEPOCH ' + str(epoch) + '\n')
            agent = agent_gen()
            epsilon = agent.noise.threshold
            agent.memory = deque(maxlen=100000)
            idx = 0
            for dt in dt_array:
                agent.noise.threshold = epsilon
                agent.memory = deque(maxlen=100000)
                self.env.dt = dt
                OneAgentSolver.go(self.env, agent, self.__callback__,
                                  start_episode=idx * episode_n,
                                  episode_n=episode_n,
                                  session_n=1,
                                  session_len=session_len,
                                  agent_learning=True
                                  )
                idx += 1
            if path:
                np.save(path + str(epoch), -self.mean_rewards)
