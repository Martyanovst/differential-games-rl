from Agents.AgentGenerator import AgentGenerator
from Environments.SimpleControl.simple_control_problem_env import SimpleControlProblem
from Resolvers.AgentTestingModule import AgentTestingModule

env = SimpleControlProblem(dt=0.5)
epoch_n = 20
episode_n = 250
noise_min = 1e-3
batch_size = 128

tester = AgentTestingModule(env)
path = './../../Tests/SimpleControl/SPHERE_R_G/'

agent_generator = AgentGenerator(env, batch_size, episode_n, noise_min)
tester.test_agent(agent_generator.generate_naf_sphere_case_rg_based, episode_n, session_len=1000, epoch_n=epoch_n, dt_array=[0.5],
                  path=path)
