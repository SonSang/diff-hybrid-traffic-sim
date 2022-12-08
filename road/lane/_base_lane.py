'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''
from typing import Dict

class BaseLane:
    '''
    Lane, 0 for macro lane, 1 for micro lane
    '''
    def __init__(self, id: int, length: float, speed_limit: float):

        self.id = id

        self.next_lane: Dict[int, BaseLane] = {}
        self.prev_lane: Dict[int, BaseLane] = {}
        
        self.length = length
        self.speed_limit = speed_limit

    def is_macro(self):
        raise NotImplementedError()

    def is_micro(self):
        raise NotImplementedError()

    def forward(self, delta_time: float):
        raise NotImplementedError()

    def update_state(self):
        raise NotImplementedError()

    def add_prev_lane(self, lane):
        self.prev_lane[lane.id] = lane

    def add_next_lane(self, lane):
        self.next_lane[lane.id] = lane

    def clear(self):
        raise NotImplementedError()