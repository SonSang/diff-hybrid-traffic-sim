from typing import Dict

from model.macro._arz import ARZ

from road.network.road_network import RoadNetwork
from road.lane._macro_lane import MacroLane
from road.lane._micro_lane import MicroLane

from road.vehicle.vehicle import DEFAULT_VEHICLE_LENGTH

from dmath.operation import sigmoid

import numpy as np

MICRO_STOP_DISTANCE = 3.0
MICRO_STOP_GRADIENT = 1e-0

class ItscpRoadNetwork(RoadNetwork):

    '''
    Specialized simulator for ITSCP environment.

    This particular simulator is needed to simulate the intersection environment,
    because there is a signal that blocks traffic flow. Since signaling is a
    non-differentiable, or discrete operation in nature, we have to handle it in
    differentiable way to use gradient-based optimization methods.
    '''

    def __init__(self, speed_limit: float):

        super().__init__(speed_limit)

        self.lane_signal: Dict = {}
        self.lane_incoming: Dict = {}

        # list of micro vehicles waiting to be generated;

        self.lane_waiting_micro_vehicle: Dict = {}
        self.lane_waiting_micro_route: Dict = {}

    def interpolate_signal(self, signal, green_value, red_value):

        '''
        Interpolate between two values using [signal].
        '''

        assert signal >= 0.0 and signal <= 1.0, ""

        return green_value * signal + red_value * (1.0 - signal)

    def setup_macro_boundary(self, id: int, differentiable: bool):

        lane: MacroLane = self.lane[id]

        assert lane.is_macro(), ""

        '''
        Leftmost cell
        '''

        # 1. get green light values;

        if not lane.has_prev_lane():

            green_leftmost_r = self.lane_incoming[id]
            green_leftmost_u = ARZ.compute_u_eq(green_leftmost_r, self.speed_limit)

        else:

            green_leftmost_r, green_leftmost_u = self.get_macro_boundary(id, True, differentiable)

        # 2. get red light values;

        red_leftmost_r = 0
        red_leftmost_u = self.speed_limit

        # 3. interpolate between them using signal;

        if not lane.has_prev_lane():

            # if there is no prev lane at all, it is same as green signal
            # as the lane is boundary lane and has to accept incoming flow;

            prev_signal = 1.0

        else:

            prev_lane_id = self.macro_route.get_prev_lane(lane.id)

            if prev_lane_id == -1:

                # if there is no connected prev lane, it is same as red signal;

                prev_signal = 0.0

            else:

                prev_signal = self.lane_signal[prev_lane_id]

        final_rightmost_r = self.interpolate_signal(prev_signal, green_leftmost_r, red_leftmost_r)
        final_rightmost_u = self.interpolate_signal(prev_signal, green_leftmost_u, red_leftmost_u)

        lane.set_leftmost_cell(final_rightmost_r, final_rightmost_u)

        '''
        Rightmost cell
        '''

        # first, get rightmost cell when green light;

        green_rightmost_r, green_rightmost_u = self.get_macro_boundary(id, False, differentiable)

        # second, get rightmost cell when red light;

        red_rightmost_r, red_rightmost_u = 1.0, 0.0

        # interpolate between them using signal;

        signal = self.lane_signal[id]

        if differentiable:

            signal = sigmoid(signal - 0.5, constant=32)

        else:

            signal = float(signal > 0.5)

        final_rightmost_r = signal * green_rightmost_r + (1.0 - signal) * red_rightmost_r
        final_rightmost_u = signal * green_rightmost_u + (1.0 - signal) * red_rightmost_u

        lane.set_rightmost_cell(final_rightmost_r, final_rightmost_u)

    def setup_micro_boundary(self, id: int, differentiable: bool):

        '''
        Set micro lane's bdry info using signal.
        '''

        SIGNAL_SIGMOID_CONSTANT = 16.0

        lane: MicroLane = self.lane[id]

        assert lane.is_micro(), ""

        # generate new vehicle according to [incoming];

        if not lane.has_prev_lane():

            enough_space = lane.entering_free_space() > DEFAULT_VEHICLE_LENGTH * 0.5

            if enough_space:

                rand = np.random.random((1,)).item()

                incoming = self.lane_incoming[id]

                waiting_vehicle_list = self.lane_waiting_micro_vehicle[id]
                waiting_route_list = self.lane_waiting_micro_route[id]

                if rand < incoming and len(waiting_vehicle_list) and len(waiting_route_list):

                    nv = waiting_vehicle_list[-1]
                    nr = waiting_route_list[-1]

                    self.add_vehicle(nv, nr)

                    self.lane_waiting_micro_vehicle[id] = waiting_vehicle_list[:-1]
                    self.lane_waiting_micro_route[id] = waiting_route_list[:-1]

        # first get delta values assuming green light;
        
        super().setup_micro_boundary(id, differentiable)

        green_head_position_delta = lane.head_position_delta
        green_head_speed_delta = lane.head_speed_delta

        if lane.num_vehicle() == 0:

            return

        # second get delta values assuming red light;

        hv = lane.get_head_vehicle()
        hr = self.micro_route[hv.id]

        red_head_position_delta = lane.length - hv.position - (hv.length * 0.5)
        red_head_position_delta = max(red_head_position_delta, 0.0)
        red_head_speed_delta = 0.0

        # get signal by weighted sum of signal values for adjacent lanes;

        prev_exist = hr.prev_lane_id() != -1
        next_exist = hr.next_lane_id() != -1

        if differentiable and prev_exist:

            prev_score = sigmoid(-hv.position, constant=SIGNAL_SIGMOID_CONSTANT)

        else:

            prev_score = 0

        if differentiable:

            curr_score = sigmoid(hv.position, constant=SIGNAL_SIGMOID_CONSTANT) * \
                            sigmoid(lane.length - hv.position, constant=SIGNAL_SIGMOID_CONSTANT)

        else:

            curr_score = 1

        if differentiable and next_exist:

            next_score = sigmoid(hv.position - lane.length, constant=SIGNAL_SIGMOID_CONSTANT)

        else:

            next_score = 0

        score_sum = prev_score + curr_score + next_score

        prev_score = prev_score / score_sum
        curr_score = curr_score / score_sum
        next_score = next_score / score_sum

        final_signal = 0

        if prev_exist:

            prev_signal = self.lane_signal[hr.prev_lane_id()]

            final_signal = final_signal + prev_score * prev_signal

        curr_signal = self.lane_signal[hr.curr_lane_id()]

        final_signal = final_signal + curr_score * curr_signal

        if next_exist:

            next_signal = self.lane_signal[hr.next_lane_id()]

            final_signal = final_signal + next_score * next_signal

        # get final delta values by weighted sum;

        if differentiable:

            final_signal = sigmoid(final_signal - 0.5, constant=32)

            lane.head_position_delta = green_head_position_delta * final_signal + red_head_position_delta * (1.0 - final_signal)
            lane.head_speed_delta = green_head_speed_delta * final_signal + red_head_speed_delta * (1.0 - final_signal)

        else:

            if ItscpRoadNetwork.is_signal_green(final_signal):

                lane.head_position_delta = green_head_position_delta
                lane.head_speed_delta = green_head_speed_delta

            else:

                lane.head_position_delta = red_head_position_delta
                lane.head_speed_delta = red_head_speed_delta

    @staticmethod
    def is_signal_green(signal: float):

        return signal >= 0.5
