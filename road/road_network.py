from typing import Callable, Dict

from road.lane._base_lane import BaseLane
from road.lane._macro_lane import MacroLane
from road.lane._micro_lane import MicroLane
from road.callback import default_bdry_callback
from road.hybrid import Hybrid

class RoadNetwork:

    def __init__(self, speed_limit: float):

        # for now, we only support uniform speed limit across entire network;

        self.lane: Dict[int, BaseLane] = {}
        self.speed_limit = speed_limit

        # set lane connectivity of [network] in current time step;
        # tt is done by setting [curr_next_lane] and [curr_prev_lane] of each lane;

        self.connectivity_callback: Callable = default_connectivity_callback
        self.connectivity_callback_args: Dict = {'network': self}

    def add_lane(self, id: int, lane: BaseLane):
        
        assert isinstance(lane, MacroLane) or isinstance(lane, MicroLane), ""
        assert id not in self.lane.keys(), ""
        
        lane.speed_limit = self.speed_limit
        lane.id = id

        self.lane[id] = lane

    def forward(self, delta_time: float):

        # set connectivity between lanes in the network;

        self.connectivity_callback(self.connectivity_callback_args)

        # set bdry conditions for lanes in the network;

        for lane in self.lane.values():

            callback = lane.bdry_callback if lane.bdry_callback is not None else default_bdry_callback

            callback(lane.bdry_callback_args)
        
        # take one step for each lane;
        # in this step, updated states are not stored in current states;
        
        for lane in self.lane.values():

            lane.forward(delta_time)

        # apply new states to current states of each lane;

        for lane in self.lane.values():

            lane.update_state()
        
        # deal with hybrid cases;

        for lane in self.lane.values():

            has_next_lane = lane.next_lane is not None and lane.curr_next_lane != -1

            # micro;

            if lane.is_micro() and not has_next_lane:

                if lane.num_vehicle():

                    hv = lane.curr_vehicle[-1]

                    if hv.position >= lane.length:

                        lane.curr_vehicle = lane.curr_vehicle[:-1]

            if not has_next_lane:

                continue
        
            # hybrid

            next_lane = lane.next_lane[lane.curr_next_lane]

            if lane.is_macro() and next_lane.is_micro():

                Hybrid.macro_to_micro(lane, next_lane)

            elif lane.is_micro() and next_lane.is_macro():
                
                Hybrid.micro_to_macro(lane, next_lane)

            elif lane.is_micro() and next_lane.is_micro():

                Hybrid.micro_to_micro(lane, next_lane)

    def connect_lane(self, prev_lane_id: int, next_lane_id: int):

        '''
        Connect two lanes of given id.
        '''

        prev_lane = self.lane[prev_lane_id]
        next_lane = self.lane[next_lane_id]

        prev_lane.add_next_lane(next_lane)
        next_lane.add_prev_lane(prev_lane)

def occupied_length(lane: BaseLane):

    '''
    Compute length of occupied region for the given lane.    
    '''

    if isinstance(lane, MicroLane):

        # compute density based on vehicles;

        length = 0

        for v in lane.curr_vehicle:

            length += v.length

    elif isinstance(lane, MacroLane):

        # compute length based on cells;

        length = 0

        for c in lane.curr_cell:

            length += c.state.q.r * lane.cell_length

    else:

        raise ValueError()          

    return length

def default_connectivity_callback(args: Dict):

    '''
    Default connectivity callback for road network.

    In this scheme, we iterate over each lane and select next lane that is most "empty".
    Also, we iterate over each lane and select prev lane that is most "full".
    If preferred next and prev lane does not match, go over to next round.
    '''

    network: RoadNetwork = args['network']

    for lane in network.lane.values():

        lane.curr_next_lane = -1
        lane.curr_prev_lane = -1

    preferred_next_lane: Dict[int, int] = {}
    preferred_prev_lane: Dict[int, int] = {}

    while True:

        did_match = False

        # set preferred next & prev lane;

        for lane in network.lane.values():

            # next lane;

            if len(lane.next_lane) and lane.curr_next_lane == -1:

                min_next_density = float("inf")

                for n_lane in lane.next_lane.values():

                    # if there is already match, ignore;

                    if n_lane.curr_prev_lane != -1:

                        continue

                    density = occupied_length(n_lane)

                    if density < min_next_density:

                        preferred_next_lane[lane.id] = n_lane.id
                        min_next_density = density

            # prev lane;

            if len(lane.prev_lane) and lane.curr_prev_lane == -1:

                max_prev_density = -float("inf")

                for n_lane in lane.prev_lane.values():

                    # if there is already match, ignore;

                    if n_lane.curr_next_lane != -1:

                        continue

                    density = occupied_length(n_lane)

                    if density > max_prev_density:

                        preferred_prev_lane[lane.id] = n_lane.id
                        max_prev_density = density

        # match preferred lanes;

        for lane in preferred_next_lane.keys():

            n_lane = preferred_next_lane[lane]
            p_lane = preferred_prev_lane[n_lane]

            if p_lane == lane:

                # match;

                did_match = True

                network.lane[p_lane].curr_next_lane = n_lane
                network.lane[n_lane].curr_prev_lane = p_lane

        preferred_next_lane.clear()
        preferred_prev_lane.clear()

        if not did_match:

            break