def go(env, agent, callback, start_episode=0, episode_n=100, session_n=1, session_len=10000, agent_learning=True):
    for episode in range(episode_n):
        session = {'states': [], 'actions': [], 'rewards': [], 'dones': []}
        state = env.reset()
        session['states'].append(state)
        for _ in range(session_len):
            action = agent.get_action(state)
            session['actions'].append(action)

            next_state, reward, done, _ = env.step(action)

            if agent_learning:
                agent.fit([state, action, reward, done, next_state])
                agent.noise.decrease()
            state = next_state
            session['states'].append(state)
            session['rewards'].append(reward)
            session['dones'].append(done)

            if done:
                break

        callback(env, agent, start_episode + episode, session)

    return agent
