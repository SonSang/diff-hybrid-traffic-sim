import numpy as np
from example.control.itscp._env import LaneID
from typing import List, Dict

def problem(lane_id: List[LaneID], num_timestep: int, num_session: int):
    '''
    First divide entire timestep into [num_session] sessions.
    Then for each session, randomly select which direction among
    [North-South] and [West-East] should have more incoming flow
    than the other.
    '''
    
    num_timestep_per_session = num_timestep // num_session

    schedule: Dict[LaneID, List[float]] = {}
    
    session_dir = []
    for i in range(num_session):
        if i == 0:
            green_dir_ns = np.random.random((1)).item() > 0.5
            if green_dir_ns:
                session_dir.append("NS")
            else:
                session_dir.append("WE")
        else:
            if session_dir[-1] == "NS":
                session_dir.append("WE")
            else:
                session_dir.append("NS")

    for id in lane_id:

        curr_schedule = []

        for session in range(num_session):
            
            r = np.random.random((1)).item()
            
            if session_dir[session] == "NS":
                
                if id.loc == "north" or id.loc == "south":
                    
                    r = 0.9 + r * 0.1
                    
                else:
                    
                    r = 0.0 + r * 0.01
                    
            else:
                
                if id.loc == "west" or id.loc == "east":
                    
                    r = 0.9 + r * 0.1
                    
                else:
                    
                    r = 0.0 + r * 0.01

            for _ in range(num_timestep_per_session):

                curr_schedule.append(r)

        curr_schedule = curr_schedule[:num_timestep]

        schedule[id] = curr_schedule

    return schedule

def problem_1(lane_id: List[LaneID], num_timestep: int):
    
    return problem(lane_id, num_timestep, 1)


def problem_2(lane_id: List[LaneID], num_timestep: int):
    
    return problem(lane_id, num_timestep, 2)


def problem_3(lane_id: List[LaneID], num_timestep: int):
    
    return problem(lane_id, num_timestep, 3)