from Agents.AgentGenerator import AgentGenerator
from Environments.DubinsCar.DubinsCar import DubinsCar_SymmetricActionInterval as DubinsCar
from Resolvers.AgentTestingModule import AgentTestingModule

env = DubinsCar(dt=1, inner_step_n=100)
epoch_n = 20
episode_n = 500
noise_min = 1e-3
batch_size = 128

tester = AgentTestingModule(env)
path = './../../Tests/DubinsCar/SPHERE_R_G/'

agent_generator = AgentGenerator(env, batch_size, episode_n, noise_min)
tester.test_agent(agent_generator.generate_naf_sphere_case_rg_based, episode_n, session_len=1000, epoch_n=epoch_n,
                  path=path, dt_array=[0.5])
