import argparse
import matplotlib.pyplot as plt

import numpy as np

from environments.dubinsCar.dubins_car_env import DubinsCar
from environments.simpleMotions.simple_motions_env import SimpleMotions
from environments.vanDerPol.van_der_pol_env import VanDerPol
from environments.pendulum.pendulum_env import Pendulum
from models.agent_evaluation_module import AgentEvaluationModule
from models.agent_generator import AgentGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, choices=('simple-motions', 'van-der-pol', 'pendulum', 'dubins-car'),
                    required=True)
parser.add_argument('--model', type=str, choices=('naf', 'bnaf', 'rb-bnaf', 'gb-bnaf'), required=True)
parser.add_argument('--epoch_num', type=int, default=500)
parser.add_argument('--dt', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--save_model_path', type=str, default=None)
parser.add_argument('--save_rewards_path', type=str, default=None)
parser.add_argument('--save_plot_path', type=str, default=None)
args = parser.parse_args()

if args.env == 'simple-motions':
    env = SimpleMotions(dt=args.dt)
elif args.env == 'van-der-pol':
    env = VanDerPol(dt=args.dt)
elif args.env == 'pendulum':
    env = Pendulum(dt=args.dt)
else:
    env = DubinsCar(dt=args.dt)

agent_generator = AgentGenerator(env, batch_size=args.batch, epoch_num=args.epoch_num, gamma=args.gamma, lr=args.lr)
if args.model == 'naf':
    agent = agent_generator.generate_naf()
elif args.model == 'bnaf':
    agent = agent_generator.generate_b_naf()
elif args.model == 'rb-bnaf':
    agent = agent_generator.generate__b_naf_reward_based()
else:
    agent = agent_generator.generate_naf_bounded_gradient_based()

training_module = AgentEvaluationModule(env)
rewards = training_module.train_agent(agent, args.epoch_num)

if args.save_model_path:
    agent.save(args.save_model_path)

if args.save_rewards_path:
    np.save(args.save_rewards_path, rewards)

plt.plot(range(args.epoch_num), rewards)
plt.xlabel('episodes')
plt.ylabel('rewards')
ax = plt.gca()
ax.set_facecolor('#eaeaf2')
plt.grid(color='white')

if args.save_plot_path:
    plt.savefig(args.save_plot_path)

plt.show()
