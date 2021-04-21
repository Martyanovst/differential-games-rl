import numpy as np

from Resolvers import OneAgentSolver


class AgentTestingModule:
    def __init__(self, env):
        self.env = env
        self.rewards = np.zeros(0)
        self.mean_rewards = np.zeros(0)

    def __callback__(self, env, agent, episode, sessions):
        total_reward = np.sum(sessions[0]['rewards'])
        self.rewards[episode] = total_reward
        mean_reward = np.mean(self.rewards[max(0, episode - 25):episode + 1])
        self.mean_rewards[episode] = mean_reward
        print("episode=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f" % (
            episode, agent.noise.threshold, -total_reward, -mean_reward))

    def __reset__(self, episode_n):
        self.rewards = np.zeros(episode_n)
        self.mean_rewards = np.zeros(episode_n)

    def test_agent(self, agent_gen, episode_n, session_len, epoch_n, path=None):
        for epoch in range(epoch_n):
            self.__reset__(episode_n)
            print('\nEPOCH ' + str(epoch) + '\n')
            OneAgentSolver.go(self.env, agent_gen(), self.__callback__,
                              episode_n=episode_n,
                              session_n=1,
                              session_len=session_len,
                              agent_learning=True
                              )
            if path:
                np.save(path + str(epoch), -self.mean_rewards)
