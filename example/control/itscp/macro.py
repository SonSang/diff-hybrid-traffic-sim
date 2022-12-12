from time import time
from example.control.itscp._env import ItscpEnv
from example.control.trainer import Trainer
from example.control.itscp.problem import problem_0, problem_1

env = ItscpEnv()
env.schedule_callback = problem_1
env.config['num_intersection'] = 1
env.config['num_lane'] = 1
env.config['mode'] = 'macro'
env.reset()

trainer = Trainer(env)
trainer.train(1, 100, 1, 1, "./result/control/itscp/macro_{}".format(time()))