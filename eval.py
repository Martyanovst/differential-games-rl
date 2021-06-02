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
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

if args.env == 'simple-motions':
    env = SimpleMotions(dt=args.dt)
elif args.env == 'van-der-pol':
    env = VanDerPol(dt=args.dt)
elif args.env == 'pendulum':
    env = Pendulum(dt=args.dt)
else:
    env = DubinsCar(dt=args.dt)