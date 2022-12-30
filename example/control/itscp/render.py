from example.control.itscp._env import ItscpEnv
from example.control.itscp.problem import problem_0, problem_1, problem_2

env = ItscpEnv()
env.schedule_callback = problem_0
env.config['num_intersection'] = 1 # 1 for fast sim
env.config['num_lane'] = 1
env.config['mode'] = 'micro'
env.config['max_num_micro_vehicle_per_lane'] = 20
env.reset()

action = [0.4816, 0.4593, 0.4681] # [0.4821, 0.4607, 0.4704] # [0.5 for _ in range((env.config['num_intersection'] ** 2) * 3)]

while True:

    env.reset()

    for _ in range(env.num_timestep):
        
        env._simulate_step_B(action, True)
        env.render(action)