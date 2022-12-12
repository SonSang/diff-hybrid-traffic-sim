from example.control.itscp._env import ItscpEnv, LaneID
from typing import List, Dict

def itscp_schedule(lane_id: List[LaneID], num_timestep: int):

    '''
    Set a schedule.
    '''

    schedule: Dict[LaneID, List[float]] = {}

    for id in lane_id:

        curr_schedule = []

        for _ in range(num_timestep):

            if id.loc == 'south' or id.loc == 'north':

                r = 1.0

            else:

                r = 0.0

            curr_schedule.append(r)

        schedule[id] = curr_schedule

    return schedule

env = ItscpEnv()
env.schedule_callback = itscp_schedule
env.config['num_intersection'] = 1 # 1 for fast sim
env.config['num_lane'] = 3
env.config['mode'] = 'micro'
env.reset()

action = [0.5 for _ in range((env.config['num_intersection'] ** 2) * 3)]

while True:

    env.reset()

    for _ in range(env.num_timestep):
        
        env._simulate_step(action)
        env.render(action)