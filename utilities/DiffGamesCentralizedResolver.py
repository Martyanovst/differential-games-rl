import numpy as np
import matplotlib.pyplot as plt


class DiffGamesCentralizedResolver:
    def fit_agent(self, env, episode_n, agent, title='fit'):
        rewards = np.zeros(episode_n)
        mean_rewards = np.zeros(episode_n)
        for episode in range(episode_n):
            state = env.reset()
            total_reward = 0
            while not env.done:
                u_action = agent.get_u_action(state)
                v_action = agent.get_v_action(state)
                next_state, reward, done, _ = env.step(u_action, v_action)
                reward = float(reward)
                total_reward += reward
                agent.fit(state, u_action, v_action, reward, done, next_state)
                state = next_state
            agent.noise.decrease()
            rewards[episode] = total_reward
            mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
            mean_rewards[episode] = mean_reward
            print("episode=%.0f, total reward=%.3f, threshold=%0.3f" % (
                episode, total_reward, agent.noise.threshold))
        plt.plot(range(episode_n), mean_rewards)
        plt.title(title)
        plt.show()
        return agent

    def play(self, env, agent, v_agent):
        state = env.reset()
        total_reward = 0
        while not env.done:
            u_action = agent.get_action(state)
            v_action = v_agent.get_action(state)
            next_state, reward, done, _ = env.step(u_action, v_action)
            next_state = next_state
            reward = float(reward)
            total_reward += reward
            state = next_state
        return total_reward

    def u_play(self, env, agent, v_agent):
        state = env.reset()
        total_reward = 0
        while not env.done:
            u_action = agent.get_u_action(state)
            v_action = v_agent.get_action(state)
            next_state, reward, done, _ = env.step(u_action, v_action)
            next_state = next_state
            reward = float(reward)
            total_reward += reward
            state = next_state
        return total_reward

    def test_agents(self, env, u_agent, v_agent, title='test'):
        reward = self.play(env, u_agent, v_agent)
        print(reward, title)

    def test_u_agents(self, env, u_agent, v_agent, title='test'):
        reward = self.u_play(env, u_agent, v_agent)
        print(reward, title)

    def fit_u_agent(self, env, episode_n, u_agent, v_agent, title='fit u-agent'):
        rewards = np.zeros(episode_n)
        mean_rewards = np.zeros(episode_n)
        for episode in range(episode_n):
            state = env.reset()
            total_reward = 0
            while not env.done:
                u_action = u_agent.get_action(state)
                v_action = v_agent.get_v_action(state)
                next_state, reward, done, _ = env.step(u_action, v_action)
                reward = float(reward)
                total_reward += reward
                u_agent.fit(state, v_action, -reward, done, next_state)
                state = next_state
            u_agent.noise.decrease()
            rewards[episode] = total_reward
            mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
            mean_rewards[episode] = mean_reward
            print(
                "episode=%.0f, total reward=%.3f, v-threshold=%0.3f" % (episode, total_reward, u_agent.noise.threshold))
        plt.plot(range(episode_n), mean_rewards)
        plt.title(title)
        plt.show()
        return u_agent

    def fit_v_agent(self, env, episode_n, u_agent, v_agent, title='fit v-agent'):
        rewards = np.zeros(episode_n)
        mean_rewards = np.zeros(episode_n)
        for episode in range(episode_n):
            state = env.reset()
            total_reward = 0
            while not env.done:
                u_action = u_agent.get_u_action(state)
                v_action = v_agent.get_action(state)
                next_state, reward, done, _ = env.step(u_action, v_action)
                reward = float(reward)
                total_reward += reward
                v_agent.fit(state, v_action, reward, done, next_state)
                state = next_state
            v_agent.noise.decrease()
            rewards[episode] = total_reward
            mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
            mean_rewards[episode] = mean_reward
            print(
                "episode=%.0f, total reward=%.3f, v-threshold=%0.3f" % (episode, total_reward, v_agent.noise.threshold))
        plt.plot(range(episode_n), mean_rewards)
        plt.title(title)
        plt.show()
        return v_agent
