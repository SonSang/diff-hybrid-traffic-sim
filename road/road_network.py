from numpy import isin
import torch as th
from typing import List

from road.lane.base_lane import BaseLane
from road.lane.macro_lane import MacroLane
from road.lane.micro_lane import MicroLane
from road.hybrid import Hybrid

class RoadNetwork:
    def __init__(self, speed_limit: float):
        self.lanes: List[BaseLane] = []
        self.speed_limit = speed_limit

        # boundary values (macro)
        self.macro_left_rho: th.Tensor = th.ones([1, 1], dtype=th.float32) * 0.2
        self.macro_left_u: th.Tensor = th.ones([1, 1], dtype=th.float32) * speed_limit
        self.macro_right_rho: th.Tensor = th.ones([1, 1], dtype=th.float32) * 0.0
        self.macro_right_u: th.Tensor = th.ones([1, 1], dtype=th.float32) * speed_limit

        # boundary values (micro)
        self.micro_left_spawn_rate: float = 0.2
        self.micro_right_pos_delta: th.Tensor = th.ones([1, 1], dtype=th.float32) * 1000.0
        self.micro_right_vel_delta: th.Tensor = th.ones([1, 1], dtype=th.float32) * 0.0

    def num_lanes(self):
        return len(self.lanes)

    def add_lane(self, lane: BaseLane):
        assert isinstance(lane, MacroLane) or isinstance(lane, MicroLane), ""
        lane.speed_limit = self.speed_limit

        if self.num_lanes() > 0:
            lane.prev_lane = self.lanes[-1]
            lane.prev_lane.next_lane = lane
        else:
            lane.prev_lane = None
        lane.next_lane = None

        self.lanes.append(lane)

    def forward_step(self, delta_time: float, parallel: bool = True):
        # take one step for each lane
        for lane in self.lanes:
            if lane.is_macro():
                lane.forward_step(delta_time, 
                                    self.macro_left_rho, 
                                    self.macro_left_u, 
                                    self.macro_right_rho, 
                                    self.macro_right_u,
                                    parallel)
            else:
                lane.forward_step(delta_time,
                                    self.micro_right_pos_delta,
                                    self.micro_right_vel_delta,
                                    parallel)

        # update state
        for lane in self.lanes:
            lane.update_state()
        
        # deal with boundary state
        for lane in self.lanes:
            if lane.next_lane is None:
                if lane.is_micro() and lane.num_vehicle() > 0:
                    # remove exiting vehicle        
                    head_pos = lane.curr_pos[[-1], :]
                    if head_pos >= lane.length:
                        lane.remove_head_vehicle()

                continue

            # hybrid
            if lane.is_macro() and lane.next_lane.is_micro():
                _, _, accel_max, accel_pref, target_vel, min_space, time_pref, vehicle_length = lane.next_lane.rand_vehicle()
                Hybrid.macro_to_micro(lane, 
                                    lane.next_lane, 
                                    accel_max, 
                                    accel_pref, 
                                    target_vel, 
                                    min_space, 
                                    time_pref, 
                                    vehicle_length)

            elif lane.is_micro() and lane.next_lane.is_macro():
                Hybrid.micro_to_macro(lane, lane.next_lane)

    def print(self):
        macro_nv = 0
        micro_nv = 0
        for i, lane in enumerate(self.lanes):
            if lane.is_macro():
                macro_nv += lane.num_vehicle()
            else:
                micro_nv += lane.num_vehicle()
        print("# vehicle: {} (macro = {}, micro = {})".format(macro_nv + micro_nv, macro_nv, micro_nv))
            # print("Lane # {}.".format(i))
            # lane.print()
            # print()