import numpy as np
from example.control.itscp._env import LaneID
from typing import List, Dict

PROBLEM_2_SCHEDULE = None

def problem_0(lane_id: List[LaneID], num_timestep: int):

    '''
    Only NS direction flows.
    '''

    schedule: Dict[LaneID, List[float]] = {}

    for id in lane_id:

        curr_schedule = []

        for _ in range(num_timestep):

            if id.loc == 'south' or id.loc == 'north':

                r = 0.0

            else:

                r = 1.0

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


def problem_2(lane_id: List[LaneID], num_timestep: int):

    '''
    Random flow.
    '''

    global PROBLEM_2_SCHEDULE

    if PROBLEM_2_SCHEDULE is None:

        # only select one of directions to have incoming flow in one session;

        num_session = 3

        session_ns_flow = []

        for _ in range(num_session):

            r = np.random.random((1,)).item()

            ns_flow = r < 0.5

            session_ns_flow.append(ns_flow)

        schedule: Dict[LaneID, List[float]] = {}

        for id in lane_id:

            curr_schedule = []

            num_step_per_session = num_timestep // num_session

            for session in range(num_session):

                if id.loc == "north" or id.loc == "south":

                    r = float(session_ns_flow[session])

                else:

                    r = 1.0 - float(session_ns_flow[session])

                for step in range(num_step_per_session):

                    curr_schedule.append(r)

            for _ in range(num_timestep % num_session):

                curr_schedule.append(curr_schedule[-1])

            schedule[id] = curr_schedule

        PROBLEM_2_SCHEDULE = schedule

    return PROBLEM_2_SCHEDULE