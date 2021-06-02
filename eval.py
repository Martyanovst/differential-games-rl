import argparse

from environments.dubinsCar.dubins_car_env import DubinsCar
from environments.pendulum.pendulum_env import Pendulum
from environments.simpleMotions.simple_motions_env import SimpleMotions
from environments.vanDerPol.van_der_pol_env import VanDerPol
from models.agent_evaluation_module import AgentEvaluationModule
from models.agent_generator import AgentGenerator
from models.naf import NAF

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, choices=('simple-motions', 'van-der-pol', 'pendulum', 'dubins-car'),
                    required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

if args.env == 'simple-motions':
    env = SimpleMotions()
elif args.env == 'van-der-pol':
    env = VanDerPol()
elif args.env == 'pendulum':
    env = Pendulum()
else:
    env = DubinsCar()

agent = AgentGenerator(env).load(args.model)
dt = agent.q_model.dt

evaluation_module = AgentEvaluationModule(env)

evaluation_module.eval_agent(agent)
