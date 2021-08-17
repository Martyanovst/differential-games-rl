from environments.dubinsCar.dubins_car_env import DubinsCar
from environments.pendulum.pendulum_env import Pendulum
from environments.simpleMotions.simple_motions_env import SimpleMotions

from environments.vanDerPol.van_der_pol_env import VanDerPol


def generate_env(config):
    env_name = config['env_name']
    if env_name == 'simple-motions':
        return SimpleMotions(**config['params'])
    elif env_name == 'van-der-pol':
        return VanDerPol(**config['params'])
    elif env_name == 'pendulum':
        return Pendulum(**config['params'])
    elif env_name == 'dubins-car':
        return DubinsCar(**config['params'])

