from time import time
from example.control.itscp._env import ItscpEnv
from example.control.trainer import Trainer
from example.control.itscp.problem import problem_0, problem_1, problem_2

env = ItscpEnv()
env.schedule_callback = problem_0
env.config['num_intersection'] = 1
env.config['num_lane'] = 1
env.config['mode'] = 'micro'
env.config['max_num_micro_vehicle_per_lane'] = 7
env.config['render'] = True
env.config['random_seed'] = 1 # np.random.randint(0, 1)
env.reset()

trainer = Trainer(env, lr=1e-3)
trainer.train(1, 1000, 1, 1, "./result/control/itscp/micro_{}".format(time()))
# trainer.train(1, 100, 4, 4, "./result/control/itscp/micro_{}".format(time()))