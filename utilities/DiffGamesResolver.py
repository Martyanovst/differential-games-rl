import numpy as np
import matplotlib.pyplot as plt


class DiffGamesResolver:
    def fit_agents(self, env, episode_n, u_agent, v_agent, title='fit', fit_step=10):
        rewards = np.zeros(episode_n)
        mean_rewards = np.zeros(episode_n)
        is_u_agent_fit = True
        for episode in range(episode_n):
            state = env.reset()
            total_reward = 0
            while not env.done:
                u_action = u_agent.get_action(state, is_u_agent_fit)
                v_action = v_agent.get_action(state, not is_u_agent_fit)
                next_state, reward, done, _ = env.step(u_action, v_action)
                reward = float(reward)
                total_reward += reward
                if is_u_agent_fit:
                    u_agent.fit(state, u_action, -reward, done, next_state)
                    v_agent.memory.append([state, v_action, reward, done, next_state])
                else:
                    v_agent.fit(state, v_action, reward, done, next_state)
                    u_agent.memory.append([state, u_action, -reward, done, next_state])
                state = next_state
            if is_u_agent_fit:
                u_agent.noise.decrease()
            else:
                v_agent.noise.decrease()
            is_u_agent_fit = episode // fit_step % 2 == 0
            rewards[episode] = total_reward
            mean_reward = np.mean(rewards[max(0, episode - 50):episode + 1])
            mean_rewards[episode] = mean_reward
            print(
                "episode=%.0f, mean reward=%.3f, u-threshold=%0.3f, v-threshold=%0.3f" % (
                    episode, mean_reward,
                    u_agent.noise.threshold,
                    v_agent.noise.threshold))
        plt.plot(range(episode_n), mean_rewards)
        plt.title(title)
        plt.show()
        return u_agent, v_agent

    def fit_v_agent(self, env, episode_n, u_agent, v_agent, title='fit v-agent'):
        rewards = np.zeros(episode_n)
        mean_rewards = np.zeros(episode_n)
        for episode in range(episode_n):
            state = env.reset()
            total_reward = 0
            while not env.done:
                u_action = u_agent.get_action(state)
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

    def fit_u_agent(self, env, episode_n, u_agent, v_agent, title='fit u-agent'):
        rewards = np.zeros(episode_n)
        mean_rewards = np.zeros(episode_n)
        for episode in range(episode_n):
            state = env.reset()
            total_reward = 0
            while not env.done:
                u_action = u_agent.get_action(state)
                v_action = v_agent.get_action(state)
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

    def play(self, env, u_agent, v_agent):
        state = env.reset()
        total_reward = 0
        while not env.done:
            u_action = u_agent.get_action(state)
            v_action = v_agent.get_action(state)
            next_state, reward, done, _ = env.step(u_action, v_action)
            reward = float(reward)
            total_reward += reward
            state = next_state
        return total_reward

    def test_agents(self, env, u_agent, v_agent, title):
        reward = self.play(env, u_agent, v_agent)
        print(reward, title)
