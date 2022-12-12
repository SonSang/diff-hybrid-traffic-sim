from example.control.itscp._env import LaneID
from typing import List, Dict

def problem_0(lane_id: List[LaneID], num_timestep: int):

    '''
    Only NS direction flows.
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

def problem_1(lane_id: List[LaneID], num_timestep: int):

    '''
    NS flows at the first half of time, then WE flows rest of the time.
    '''

    schedule: Dict[LaneID, List[float]] = {}

    for id in lane_id:

        curr_schedule = []

        for step in range(num_timestep):

            if id.loc == 'south' or id.loc == 'north':

                r = float(step < num_timestep * 0.5)

            else:

                r = float(step > num_timestep * 0.5)
                
            curr_schedule.append(r)

        schedule[id] = curr_schedule

    return schedule