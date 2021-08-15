import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from environments.enviroment_generator import generate_env
from models.agent_evaluation_module import AgentEvaluationModule
from models.agent_generator import AgentGenerator


def plot_reward(epoch_num, rewards_array, save_plot_path=None):
    plt.plot(range(epoch_num), rewards_array)
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    ax = plt.gca()
    ax.set_facecolor('#eaeaf2')
    plt.grid(color='white')
    if save_plot_path:
        plt.savefig(save_plot_path)
    plt.show()


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=file_path, required=True)
args = parser.parse_args()

with open('args.config') as json_config_file:
    config = json.load(json_config_file)

env = generate_env(config.enviroment)

agent_generator = AgentGenerator(env,
                                 batch_size=config.batch,
                                 epoch_num=config.epoch_num,
                                 gamma=config.gamma,
                                 lr=config.lr)

agent = agent_generator.generate(config.model)
training_module = AgentEvaluationModule(env)
rewards = training_module.train_agent(agent, config.epoch_num)
plot_reward(config.epoch_num, rewards, config.get('save_plot_path'))

if config.save_model_path:
    agent.save(config.save_model_path)

if config.save_rewards_path:
    np.save(config.save_rewards_path, rewards)
