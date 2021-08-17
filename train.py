import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import random
import numpy as np

from environments.enviroment_generator import generate_env
from models.agent_evaluation_module import AgentEvaluationModule
from models.agent_generator import AgentGenerator


def configure_random_seed(seed):
    if seed:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)


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

with open(args.config) as json_config_file:
    config = json.load(json_config_file)
train_settings = config['train_cfg']
configure_random_seed(train_settings.get('random_seed'))

env = generate_env(config['environment'])

agent_generator = AgentGenerator(env, train_cfg=train_settings, model_cfg=config['model'])

agent = agent_generator.generate()
training_module = AgentEvaluationModule(env)
rewards = training_module.train_agent(agent, train_settings['epoch_num'])
plot_reward(train_settings['epoch_num'], rewards, train_settings.get('save_plot_path'))

save_model_path = train_settings.get('save_model_path')
if save_model_path:
    agent.save(save_model_path)

save_rewards_path = train_settings.get('save_rewards_path')
if save_rewards_path:
    np.save(save_rewards_path, rewards)
