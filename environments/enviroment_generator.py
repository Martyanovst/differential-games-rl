from environments.dubinsCar.dubins_car_env import DubinsCar
from environments.pendulum.pendulum_env import Pendulum
from environments.simpleMotions.simple_motions_env import SimpleMotions

from environments.vanDerPol.van_der_pol_env import VanDerPol


def generate_env(config):
    if config.env_name == 'simple-motions':
        return SimpleMotions(**config.env_params)
    elif config.env_name == 'van-der-pol':
        return VanDerPol(**config.env_params)
    elif config.env_name == 'pendulum':
        return Pendulum(**config.env_params)
    elif config.env_name == 'dubins-car':
        return DubinsCar(**config.env_params)

