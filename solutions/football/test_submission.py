from kaggle_environments import make

env = make("football", configuration={"save_video": True,
                                      "scenario_name": "11_vs_11_kaggle"})

# Define players
left_player = "main.py"  # A custom agent, eg. random_agent.py or example_agent.py
right_player = "main.py"  # eg. A built in 'AI' agent or the agent again


output = env.run([left_player, right_player])

print(output)
# print(f"Final score: {sum([r['reward'] for r in output[0]])} : {sum([r['reward'] for r in output[1]])}")

env.render(mode="human", width=800, height=600)