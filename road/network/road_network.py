from typing import Dict

import numpy as np

from road.lane._base_lane import BaseLane
from road.lane._macro_lane import MacroLane
from road.lane._micro_lane import MicroLane, DEFAULT_HEAD_POSITION_DELTA, DEFAULT_HEAD_SPEED_DELTA
from road.vehicle.micro_vehicle import MicroVehicle, DEFAULT_VEHICLE_LENGTH

from road.network.route import MacroRoute, MicroRoute
from road.network.conversion import Conversion

from dmath.operation import sigmoid

MAX_ROUTE_LENGTH = 32

class RoadNetwork:

    def __init__(self, speed_limit: float):

        self.lane: Dict[int, BaseLane] = {}

        # for now, we only support uniform speed limit across entire network;

        self.speed_limit = speed_limit

        # for now, we only support uniform length of vehicle;

        self.vehicle_length = DEFAULT_VEHICLE_LENGTH

        # macro;

        self.macro_route = MacroRoute()

        # micro;

        self.vehicle: Dict[int, MicroVehicle] = {}
        self.micro_route: Dict[int, MicroRoute] = {}

        # num;

        self.num_lane = 0
        self.num_vehicle = 0

    def add_lane(self, lane: BaseLane):
        
        assert isinstance(lane, MacroLane) or isinstance(lane, MicroLane), ""
        
        lane.speed_limit = self.speed_limit
        
        lane.id = self.num_lane
        self.num_lane += 1

        self.lane[lane.id] = lane

        return lane.id

    def add_vehicle(self, nv: MicroVehicle, route: MicroRoute):

        assert nv.length == self.vehicle_length, ""

        nv.id = self.num_vehicle
        self.num_vehicle += 1

        self.vehicle[nv.id] = nv
        self.micro_route[nv.id] = route

        # add vehicle to the lane;

        curr_lane = self.lane[route.curr_lane_id()]

        assert curr_lane.is_micro()
        
        curr_lane: MicroLane = curr_lane
        curr_lane.add_vehicle(nv)

        return nv.id

    def forward(self, delta_time: float, differentiable: bool):

        '''
        Take a single forward step of this simulation.
        
        @ differentiable: If it is true, it gives slightly different
        result from original, discrete rule-based process, since it needs
        to be differentiable.

        '''

        # set boundary values for lanes in the network;

        for lane in self.lane.values():

            self.setup_boundary(lane.id, differentiable)
        
        # take one step for each lane;
        # in this step, updated states are not stored in current states;
        
        for lane in self.lane.values():

            lane.forward(delta_time)

        # apply new states to current states of each lane;

        for lane in self.lane.values():

            lane.update_state()
        
        # deal with hybrid cases;

        self.conversion(delta_time)

    def conversion(self, delta_time: float):

        '''
        Deal with conversion process between lanes.
        '''

        for lane in self.lane.values():

            if lane.is_macro():

                self.conversion_macro(lane, delta_time)

            else:

                self.conversion_micro(lane)

    def conversion_macro(self, lane: MacroLane, delta_time: float):

        next_lane_id = self.macro_route.get_next_lane(lane.id)

        if next_lane_id == -1:

            return

        next_lane = self.lane[next_lane_id]

        if next_lane.is_macro():

            Conversion.macro_to_macro(self, lane, next_lane)

        else:

            Conversion.macro_to_micro(self, lane, next_lane, delta_time)

    def conversion_micro(self, lane: MicroLane):

        if lane.num_vehicle():

            # check if head vehicle goes out of the lane;

            vid = lane.get_head_vehicle().id
            hv = self.vehicle[vid]
            hr = self.micro_route[vid]

            next_lane_id = hr.next_lane_id()

            if next_lane_id == -1:

                Conversion.micro_to_none(self, lane)

            else:    

                next_lane = self.lane[hr.next_lane_id()]

                if next_lane.is_macro():

                    Conversion.micro_to_macro(self, lane)

                else:

                    Conversion.micro_to_micro(self, lane)

    def connect_lane(self, prev_lane_id: int, next_lane_id: int):

        '''
        Connect two lanes of given id.
        '''

        prev_lane = self.lane[prev_lane_id]
        next_lane = self.lane[next_lane_id]

        prev_lane.add_next_lane(next_lane)
        next_lane.add_prev_lane(prev_lane)

    def setup_boundary(self, id: int, differentiable: bool):

        '''
        Set boundary values for the given lane.
        '''

        lane = self.lane[id]

        if lane.is_macro():

            self.setup_macro_boundary(id, differentiable)

        else:

            self.setup_micro_boundary(id, differentiable)

    '''
    Functions for macro simulation.
    '''

    def get_macro_state_of_micro_lane(self, id: int, differentiable: bool):

        '''
        Get macro states (density, speed) of the given micro lane.
        '''

        lane: MicroLane = self.lane[id]

        assert lane.is_micro(), ""

        density = 0
        speed_sum = 0
        num_vehicle = 0

        # vehicles on this lane;

        for v in lane.curr_vehicle:

            on_this_lane = lane.on_this_lane(v.position, differentiable)

            density = density + on_this_lane * (v.length / lane.length)
            speed_sum = speed_sum + on_this_lane * v.speed
            num_vehicle = num_vehicle + on_this_lane

        # vehicles on prev lanes;

        for prev_lane in lane.prev_lane.values():

            if prev_lane.is_macro():

                continue

            mi_prev_lane: MicroLane = prev_lane

            for v in mi_prev_lane.curr_vehicle:

                nr = self.micro_route[v.id]

                # only consider vehicle that will come to this lane;

                if nr.next_lane_id() != id:

                    continue

                position = -(mi_prev_lane.length - v.position)

                on_this_lane = lane.on_this_lane(position, differentiable)

                density = density + on_this_lane * (v.length / lane.length)
                speed_sum = speed_sum + on_this_lane * v.speed
                num_vehicle = num_vehicle + on_this_lane

        # vehicles on next lanes;

        for next_lane in lane.next_lane.values():

            if next_lane.is_macro():

                continue

            mi_next_lane: MicroLane = next_lane

            for v in mi_next_lane.curr_vehicle:

                nr = self.micro_route[v.id]

                # only consider vehicle that went through this lane;

                if nr.prev_lane_id() != id:

                    continue

                position = lane.length + v.position

                on_this_lane = lane.on_this_lane(position, differentiable)

                density = density + on_this_lane * (v.length / lane.length)
                speed_sum = speed_sum + on_this_lane * v.speed
                num_vehicle = num_vehicle + on_this_lane

        density = min(density, 1.0)

        if num_vehicle > 0:
            
            speed = speed_sum / num_vehicle

        else:

            speed = self.speed_limit

        return density, speed

    def get_macro_boundary(self, id: int, left: bool, differentiable: bool):

        '''
        Get boundary values for given macro lane.
        '''

        if left:

            adj_lane_id = self.macro_route.get_prev_lane(id)

        else:

            adj_lane_id = self.macro_route.get_next_lane(id)

        # assume empty cell if there is no adjacent lane;

        if adj_lane_id == -1:

            return 0, self.speed_limit

        else:

            adj_lane = self.lane[adj_lane_id]

            if adj_lane.is_macro():

                ma_adj_lane: MacroLane = adj_lane

                if left:

                    cell = ma_adj_lane.curr_cell[-1]

                else:

                    cell = ma_adj_lane.curr_cell[0]

                return cell.state.q.r, cell.state.u

            else:

                mi_adj_lane: MicroLane = adj_lane

                if left:

                    # for prev micro lane, we do not care about it;

                    r = 0.0
                    u = self.speed_limit

                else:

                    # for next micro lane, we use its average density and speed;

                    r, u = self.get_macro_state_of_micro_lane(mi_adj_lane.id, differentiable)

                return r, u

    def setup_macro_boundary(self, id: int, differentiable: bool):

        '''
        Find bdry values for given macro lane.

        The bdry values are set based on [self.macro_route], which tells us
        connectivity information between macro lanes.
        '''

        lane = self.lane[id]

        assert lane.is_macro(), ""

        m_lane: MacroLane = lane

        # leftmost cell;

        leftmost_r, leftmost_u = self.get_macro_boundary(id, True, differentiable)
        m_lane.set_leftmost_cell(leftmost_r, leftmost_u)

        # rightmost cell;

        rightmost_r, rightmost_u = self.get_macro_boundary(id, False, differentiable)
        m_lane.set_rightmost_cell(rightmost_r, rightmost_u)

    def create_random_macro_route(self):

        '''
        Create a random [MacroRoute] for this road network.
        '''

        route = MacroRoute()

        # randomly permute lane ids to prevent uniform ordering;

        perm_lane_id = np.random.permutation(list(self.lane.keys()))

        for lane_id in perm_lane_id:

            lane = self.lane[lane_id]

            if lane.is_micro():

                continue

            perm_next_lane_id = np.random.permutation(list(lane.next_lane.keys()))

            for next_lane_id in perm_next_lane_id:

                # if the next lane has not been connected to yet;

                if not next_lane_id in route.prev_lane_dict.keys():

                    route.next_lane_dict[lane_id] = next_lane_id

                    route.prev_lane_dict[next_lane_id] = lane_id

                    break

        return route

    '''
    Functions for micro simulation.
    '''

    def setup_micro_boundary(self, id: int, differentiable: bool):

        '''
        Find brdy values for given micro lane.

        By iterating through next lanes in the route of the vehicle,
        this function finds the leading vehicle in (possibly) differentiable
        manner. This function is needed to set bdry conditions of 
        micro lanes in this road network.
        '''

        # to make decision process for leading vehicle differentiable, 
        # we adopt a smooth offset that determines possible leading vehicles;

        MICRO_BDRY_SMOOTH_OFFSET = 5.0
        SIGMOID_CONSTANT = 16.0 / MICRO_BDRY_SMOOTH_OFFSET
        AHEAD_SIGMOID_CONSTANT = 256.0

        lane: MicroLane = self.lane[id]

        assert lane.is_micro(), ""

        # if there is no vehicle, just set bdry as default;

        if lane.num_vehicle() == 0:

            lane.head_position_delta = DEFAULT_HEAD_POSITION_DELTA
            lane.head_speed_delta = DEFAULT_HEAD_SPEED_DELTA

            return

        # start from remaining length of current lane;

        nv: MicroVehicle = self.vehicle[lane.get_head_vehicle().id]
        nr: MicroRoute = self.micro_route[lane.get_head_vehicle().id]

        curr_head_position_delta = lane.length - nv.position - (nv.length) * 0.5

        # iterate through future path and collect vehicles that could be leading vehicle;
        # we give score to each of the possible leading vehicles, because it needs to be differentiable;
        # the score is used to do weighted-average and get the final delta values;

        curr_lane_idx = nr.curr_idx

        possible_lv_score = {}
        possible_lv_position_delta = {}
        possible_lv_speed_delta = {}

        while curr_lane_idx < nr.route_length() - 1:

            curr_lane_id = nr.route[curr_lane_idx]
            next_lane_id = nr.route[curr_lane_idx + 1]

            curr_lane = self.lane[curr_lane_id]
            next_lane = self.lane[next_lane_id]

            # if we found a leading vehicle located exactly on the route, terminate;
            # or, if the next lane is macro lane, terminate;

            found_direct_leading_vehicle = False
            is_next_lane_micro = False

            # iterate through [next_lane]'s prev lanes and find possible leading vehicle;

            for n_prev_lane in next_lane.prev_lane.values():

                # if [n_prev_lane] is same as our lane, just continue;

                if n_prev_lane.id == id:

                    continue

                if isinstance(n_prev_lane, MacroLane):

                    continue

                elif isinstance(n_prev_lane, MicroLane):

                    if n_prev_lane.num_vehicle() == 0:

                        continue

                    if differentiable:

                        possible_lv = n_prev_lane.get_head_vehicle()

                        # if possible leading vehicle does not proceed to same route, ignore it;

                        possible_lv_next_lane_id = self.micro_route[possible_lv.id].next_lane_id()

                        if possible_lv_next_lane_id == -1 or possible_lv_next_lane_id != next_lane_id:

                            continue

                        # score;

                        possible_lv_offset = n_prev_lane.length - possible_lv.position

                        score = sigmoid(MICRO_BDRY_SMOOTH_OFFSET - possible_lv_offset,
                                        SIGMOID_CONSTANT,)

                        # compare remaining distance to the end of the lane;
                        # if possible lv is ahead, then use it as lv;

                        possible_lv_offset = possible_lv_offset - possible_lv.length * 0.5

                        ahead_score = sigmoid(curr_head_position_delta - possible_lv_offset, 
                                            constant=AHEAD_SIGMOID_CONSTANT)

                        score = score * ahead_score

                        possible_lv_score[possible_lv.id] = score

                        # delta;

                        position_delta = curr_head_position_delta + (possible_lv_offset - possible_lv.length * 0.5)
                        position_delta = max(position_delta, 0.0)
                        possible_lv_position_delta[possible_lv.id] = position_delta

                        speed_delta = nv.speed - possible_lv.speed
                        possible_lv_speed_delta[possible_lv.id] = speed_delta

                else:

                    raise ValueError()

            # iterate through [curr_lane]'s next lanes and find possible leading vehicles;

            for n_next_lane in curr_lane.next_lane.values():

                if isinstance(n_next_lane, MacroLane):

                    # if the next lane is macro, we only consider it only when it is on the route;

                    if n_next_lane.id != next_lane.id:

                        continue

                    is_next_lane_micro = True

                    # compute delta values based on the cell states;

                    avg_density = 0
                    avg_speed = 0

                    for cell in n_next_lane.curr_cell:

                        avg_density = avg_density + cell.state.q.r
                        avg_speed = avg_speed + cell.state.u

                    avg_density = avg_density / n_next_lane.num_cell
                    avg_speed = avg_speed / n_next_lane.num_cell

                    virtual_position = n_next_lane.length * (1.0 - avg_density)
                    virtual_speed = avg_speed
                    virtual_vid = -1

                    possible_lv_score[virtual_vid] = 1.0

                    position_delta = curr_head_position_delta + virtual_position
                    position_delta = max(position_delta, 0.0)
                    possible_lv_position_delta[virtual_vid] = position_delta

                    speed_delta = nv.speed - virtual_speed
                    possible_lv_speed_delta[virtual_vid] = speed_delta

                elif isinstance(n_next_lane, MicroLane):

                    if n_next_lane.num_vehicle() == 0:

                        continue

                    if n_next_lane.id == next_lane.id or differentiable:

                        possible_lv = n_next_lane.get_tail_vehicle()

                        # if possible leading vehicle did not come from same route, ignore it;

                        possible_lv_prev_lane_id = self.micro_route[possible_lv.id].prev_lane_id()

                        if possible_lv_prev_lane_id == -1 or possible_lv_prev_lane_id != curr_lane_id:

                            continue

                        possible_lv_offset = possible_lv.position

                        if n_next_lane.id == next_lane.id:

                            # we found direct leading vehicle on the route;

                            found_direct_leading_vehicle = True
                            score = 1.0

                        else:

                            score = sigmoid(MICRO_BDRY_SMOOTH_OFFSET - possible_lv_offset,
                                            SIGMOID_CONSTANT,)

                        possible_lv_score[possible_lv.id] = score

                        position_delta = curr_head_position_delta + (possible_lv_offset - possible_lv.length * 0.5)
                        position_delta = max(position_delta, 0.0)
                        possible_lv_position_delta[possible_lv.id] = position_delta

                        speed_delta = nv.speed - possible_lv.speed 
                        possible_lv_speed_delta[possible_lv.id] = speed_delta

                else:

                    raise ValueError()

            if found_direct_leading_vehicle or is_next_lane_micro:

                break

            curr_head_position_delta = curr_head_position_delta + next_lane.length

            curr_lane_idx += 1

        # if there was no possible leading vehicle at all, just return default values;
        
        if len(possible_lv_score.keys()) == 0:

            lane.head_position_delta = DEFAULT_HEAD_POSITION_DELTA
            lane.head_speed_delta = DEFAULT_HEAD_SPEED_DELTA

            return

        # do weighted average of position deltas and speed deltas;

        score_sum = 0

        for score in possible_lv_score.values():

            score_sum = score_sum + score

        for id in possible_lv_score.keys():

            possible_lv_score[id] = possible_lv_score[id] / score_sum

        w_position_delta = 0
        w_speed_delta = 0

        for id in possible_lv_score.keys():

            score = possible_lv_score[id]

            w_position_delta = w_position_delta + score * possible_lv_position_delta[id]
            w_speed_delta = w_speed_delta + score * possible_lv_speed_delta[id]

        # set final deltas;

        lane.head_position_delta = w_position_delta
        lane.head_speed_delta = w_speed_delta

    def create_default_vehicle_with_random_route(self, lane_id: int):

        '''
        Create default vehicle starting from the given lane.
        '''

        vehicle = MicroVehicle.default_micro_vehicle(self.speed_limit)
        route = self.create_random_route(lane_id)

        return vehicle, route

    def create_random_vehicle_with_random_route(self, lane_id: int):

        '''
        Create random vehicle starting from the given lane.
        '''

        vehicle = MicroVehicle.random_micro_vehicle(self.speed_limit)
        route = self.create_random_route(lane_id)

        return vehicle, route

    def create_random_route(self, lane_id: int):

        '''
        Create random route for micro vehicle starting from the given lane.
        '''

        route = []

        curr_lane_id = lane_id

        for _ in range(MAX_ROUTE_LENGTH):

            route.append(curr_lane_id)

            curr_lane = self.lane[curr_lane_id]

            if not curr_lane.has_next_lane():

                break

            next_lane_idx = np.random.randint(0, curr_lane.num_next_lane())
            next_lane_id = list(curr_lane.next_lane.keys())[next_lane_idx]

            beg_next_lane_idx = next_lane_idx

            # try to select route that has not been selected before;

            while next_lane_id in route:

                next_lane_idx = (next_lane_idx + 1) % curr_lane.num_next_lane()
                next_lane_id = list(curr_lane.next_lane.keys())[next_lane_idx]
                
                # if there is no new path, we just stick to first choice;

                if next_lane_idx == beg_next_lane_idx:

                    break

            curr_lane_id = next_lane_id

        route = MicroRoute(route)

        return route
