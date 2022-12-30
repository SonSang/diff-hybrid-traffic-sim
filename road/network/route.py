from typing import Dict, List

class MacroRoute:

    '''
    [MacroRoute] defines connectivity relationship between macro lanes,
    or at least, macro and micro lanes. This is needed because we need
    boundary cell information in running simulation for macro lanes.

    Note that this class defines connectivity between all the lanes in
    a road network, not only one of them.
    '''

    def __init__(self):

        self.next_lane_dict: Dict[int, int] = {}
        self.prev_lane_dict: Dict[int, int] = {}

    def get_next_lane(self, lane_id: int):

        if lane_id in self.next_lane_dict.keys():

            return self.next_lane_dict[lane_id]

        else:

            return -1

    def get_prev_lane(self, lane_id: int):

        if lane_id in self.prev_lane_dict.keys():

            return self.prev_lane_dict[lane_id]

        else:

            return -1

class MicroRoute:

    '''
    [MicroRoute] is a sequence of lanes where a micro vehicle runs on.
    It also contains the current index among the sequence, so that we
    can query which lane the vehicle is running on right now.
    '''

    def __init__(self, route: List[int], curr_idx: int = 0):

        self.route: List[int] = route
        self.curr_idx = curr_idx

    def increment_curr_idx(self):

        self.curr_idx += 1

    def route_length(self):

        return len(self.route)

    def curr_lane_id(self):

        return self.route[self.curr_idx]

    def prev_lane_id(self):

        if self.curr_idx > 0:

            return self.route[self.curr_idx - 1]

        else:

            return -1

    def next_lane_id(self):

        if self.curr_idx < self.route_length() - 1:

            return self.route[self.curr_idx + 1]

        else:

            return -1