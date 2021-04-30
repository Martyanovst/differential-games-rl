import numpy as np


def get_session(env, agent, session_len, agent_learning):
    session = {}
    session['states'], session['actions'], session['rewards'], session['dones'] = [], [], [], []

    state = env.reset()
    session['states'].append(state)

    if agent_learning:
        agent.noise.reset()

    done = False

    for _ in range(session_len):
        action = agent.get_action(state)
        session['actions'].append(action)

        state, reward, done, _ = env.step(action)
        session['states'].append(state)
        session['rewards'].append(reward)
        session['dones'].append(done)

        if done:
            break

    return session


def go(env, agent, callback, start_episode=0, episode_n=100, session_n=1, session_len=10000, agent_learning=True):
    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len, agent_learning) for _ in range(session_n)]

        callback(env, agent, start_episode + episode, sessions)

        if agent_learning:
            agent.fit(sessions)
            agent.noise.decrease()

    return agent
