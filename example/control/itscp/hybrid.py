from time import time
from example.control.itscp._env import ItscpEnv
from example.control.trainer import Trainer
from example.control.itscp.problem import problem_0, problem_1

env = ItscpEnv()
env.schedule_callback = problem_0
env.config['num_intersection'] = 3
env.config['num_lane'] = 1
env.config['cell_length'] = 20
env.config['mode'] = 'hybrid'
env.config['render'] = False
env.config['random_seed'] = 1 # np.random.randint(0, 1)
env.reset()

trainer = Trainer(env)
trainer.train(1, 100, 1, 1, "./result/control/itscp/hybrid_{}".format(time()))