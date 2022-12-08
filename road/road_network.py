import torch as th
from typing import Dict

from road.lane._base_lane import BaseLane
from road.lane._macro_lane import MacroLane
from road.lane._micro_lane import MicroLane
from road.hybrid import Hybrid

class RoadNetwork:

    def __init__(self, speed_limit: float):

        # for now, we only support uniform speed limit across entire network;

        self.lane: Dict[int, BaseLane] = {}
        self.speed_limit = speed_limit


    def add_lane(self, id: int, lane: BaseLane):
        
        assert isinstance(lane, MacroLane) or isinstance(lane, MicroLane), ""
        assert id not in self.lane.keys(), ""
        
        lane.speed_limit = self.speed_limit
        lane.id = id

        self.lane[id] = lane

    def forward(self, delta_time: float):
        
        # take one step for each lane;
        # in this step, updated states are not stored in current states;
        
        for lane in self.lane.values():

            lane.forward(delta_time)

        # apply new states to current states of each lane;

        for lane in self.lane.values():

            lane.update_state()
        
        # deal with hybrid cases;

        for lane in self.lane.values():

            if lane.next_lane is None:

                continue
        
            # hybrid

            if lane.is_macro() and lane.next_lane.is_micro():

                Hybrid.macro_to_micro(lane, lane.next_lane)

            elif lane.is_micro() and lane.next_lane.is_macro():
                
                Hybrid.micro_to_macro(lane, lane.next_lane)

    def connect_lane(self, prev_lane_id: int, next_lane_id: int):

        '''
        Connect two lanes of given id.
        '''

        prev_lane = self.lane[prev_lane_id]
        next_lane = self.lane[next_lane_id]

        prev_lane.add_next_lane(next_lane)
        next_lane.add_prev_lane(prev_lane)