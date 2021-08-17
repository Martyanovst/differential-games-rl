import numpy as np


class SingleAgentEvaluationModule:
    def __init__(self, env):
        self.env = env
        self.rewards = np.zeros(0)
        self.mean_rewards = np.zeros(0)

    def __callback__(self, agent, epoch, total_reward):
        self.rewards[epoch] = total_reward
        mean_reward = np.mean(self.rewards[max(0, epoch - 25):epoch + 1])
        self.mean_rewards[epoch] = mean_reward
        print("epoch=%.0f, noise threshold=%.3f, total reward=%.3f, mean reward=%.3f" % (
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
        if agent_learning:
            agent.noise.decrease()
        return total_reward

    def train_agent(self, agent, train_cfg):
        epoch_num = train_cfg['epoch_num']
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


class TwoAgentsEvaluationModule:
    def __init__(self, env):
        self.env = env
        self.rewards = np.zeros(0)
        self.mean_rewards = np.zeros(0)
        self.step_number = 0
        self.learning_delay = 1

    def __callback__(self, agent_u, agent_v, epoch, total_reward):
        self.rewards[epoch] = total_reward
        mean_reward = np.mean(self.rewards[max(0, epoch - 25):epoch + 1])
        self.mean_rewards[epoch] = mean_reward
        print("epoch=%.0f, u-noise=%.3f, v-noise=%.3f, total reward=%.3f, mean reward=%.3f" % (
            epoch, agent_u.noise.threshold, agent_v.noise.threshold, total_reward, mean_reward))

    def __reset__(self, epoch_num):
        self.rewards = np.zeros(epoch_num)
        self.mean_rewards = np.zeros(epoch_num)
        self.step_number = 0

    def u_learning_step(self):
        return (self.step_number // self.learning_delay) % 2 == 0

    def v_learning_step(self):
        return (self.step_number // self.learning_delay) % 2 == 1

    def _evaluate_(self, u_agent, v_agent, u_learning=False, v_learning=False, render=False):
        total_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()
            action_u = u_agent.get_action(state)
            action_v = v_agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action_u, action_v)
            if u_learning and self.u_learning_step():
                u_agent.fit([state, action_u, reward, done, next_state])
            if v_learning and self.v_learning_step():
                v_agent.fit([state, action_v, reward, done, next_state])
            state = next_state
            total_reward += reward
        if u_learning:
            u_agent.noise.decrease()
        if v_learning:
            u_agent.noise.decrease()
        return total_reward

    def train_agent(self, u_agent, v_agent, train_cfg):
        epoch_num = train_cfg['epoch_num']
        self.learning_delay = train_cfg['learning_delay']
        self.__reset__(epoch_num)
        u_agent.train()
        v_agent.train()
        for epoch in range(epoch_num):
            rewards = self._evaluate_(u_agent, v_agent, u_learning=True, v_learning=True, render=False)
            total_reward = np.sum(rewards)
            self.__callback__(u_agent, v_agent, epoch, total_reward)

        return self.mean_rewards

    def eval_agent(self, u_agent, v_agent):
        u_agent.eval()
        v_agent.eval()
        total_reward = self._evaluate_(u_agent, v_agent, u_learning=True, v_learning=True, render=True)
        print('Evaluation finished: ')
        print('Final state: ')
        print(self.env.state)
        print('Final Score: ')
        print(total_reward)
