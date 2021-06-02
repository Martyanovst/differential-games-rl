import numpy as np


class AgentEvaluationModule:
    def __init__(self, env):
        self.env = env
        self.rewards = np.zeros(0)
        self.mean_rewards = np.zeros(0)

    def __callback__(self, agent, epoch, total_reward):
        self.rewards[epoch] = total_reward
        mean_reward = np.mean(self.rewards[max(0, epoch - 25):epoch + 1])
        self.mean_rewards[epoch] = mean_reward
        print("epoch=%.0f, noise_threshold=%.3f, total reward=%.3f, mean reward=%.3f" % (
            epoch, agent.noise.threshold, total_reward, mean_reward))

    def __reset__(self, epoch_num):
        self.rewards = np.zeros(epoch_num)
        self.mean_rewards = np.zeros(epoch_num)

    def _evaluate_(self, agent, agent_learning=False, render=False):
        total_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            if agent_learning:
                agent.fit([state, action, reward, done, next_state])
            state = next_state
            total_reward += reward

        agent.noise.decrease()
        return total_reward

    def train_agent(self, agent, epoch_num):
        self.__reset__(epoch_num)
        agent.train()
        for epoch in range(epoch_num):
            rewards = self._evaluate_(agent, agent_learning=True, render=False)
            total_reward = np.sum(rewards)
            self.__callback__(agent, epoch, total_reward)

        return self.mean_rewards

    def eval_agent(self, agent):
        agent.eval()
        total_reward = self._evaluate_(agent, agent_learning=False, render=True)
        print('Evaluation finished: ')
        print('Final state: ')
        print(self.env.state)
        print('Final Score: ')
        print(total_reward)
