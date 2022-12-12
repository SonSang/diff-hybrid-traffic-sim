from typing import Dict
from road.lane._macro_lane import MacroLane
from road.lane._micro_lane import MicroLane, DEFAULT_HEAD_POSITION_DELTA, DEFAULT_HEAD_SPEED_DELTA

def default_bdry_callback(args: Dict):

    '''
    Default bdry callback function for a lane.
    '''

    lane = args['lane']

    if isinstance(lane, MacroLane):

        default_macro_bdry_callback(args)

    elif isinstance(lane, MicroLane):

        default_micro_bdry_callback(args)

    else:

        raise ValueError()

def default_macro_bdry_callback(args: Dict):
    
    '''
    Default boundary callback function for macro lane.
    '''

    lane: MacroLane = args['lane']

    # next lane;

    if len(lane.next_lane):

        if lane.curr_next_lane == -1:

            # if there are next lanes but do not have match, it is regarded as blocked;

            lane.set_rightmost_cell(1.0, 0.0)

        else:

            next_lane = lane.next_lane[lane.curr_next_lane]

            if isinstance(next_lane, MicroLane):

                # compute density and speed by averaging vehicles;

                density = 0
                speed = 0

                if len(next_lane.curr_vehicle):

                    for v in next_lane.curr_vehicle:

                        density += v.length
                        speed += v.speed

                    density /= next_lane.length
                    speed /= len(next_lane.curr_vehicle)

                else:

                    density = 0
                    speed = next_lane.speed_limit

                density = min(density, 1.0)

                lane.set_rightmost_cell(density, speed)

            elif isinstance(next_lane, MacroLane):

                density = next_lane.curr_cell[0].state.q.r
                speed = next_lane.curr_cell[0].state.u

                lane.set_rightmost_cell(density, speed)

            else:

                raise ValueError()
    
    else:

        # if there was no next lane at the first place, assume vacant lane;

        lane.set_rightmost_cell(0.0, lane.speed_limit)

    # prev lane;

    if len(lane.prev_lane):

        if lane.curr_prev_lane == -1:

            # if there are prev lanes but do not have match, it is regarded as vacant;

            lane.set_leftmost_cell(0.0, lane.speed_limit)

        else:

            prev_lane = lane.prev_lane[lane.curr_prev_lane]
            
            if isinstance(prev_lane, MicroLane):

                # zero density;

                density = 0
                speed = prev_lane.speed_limit
                lane.set_leftmost_cell(density, speed)

            elif isinstance(prev_lane, MacroLane):

                density = prev_lane.curr_cell[-1].state.q.r
                speed = prev_lane.curr_cell[-1].state.u

                lane.set_leftmost_cell(density, speed)

            else:

                raise ValueError()

    else:

        # if there was no prev lane at the first place, assume vacant lane;

        lane.set_leftmost_cell(0.0, lane.speed_limit)

def default_micro_bdry_callback(args: Dict):
    
    '''
    Default boundary callback function for micro lane.
    '''

    lane: MicroLane = args['lane']

    lane.head_position_delta = DEFAULT_HEAD_POSITION_DELTA
    lane.head_speed_delta = DEFAULT_HEAD_SPEED_DELTA

    if len(lane.curr_vehicle) == 0:

        return

    # next lane;

    v = lane.curr_vehicle[-1]

    if len(lane.next_lane):

        if lane.curr_next_lane == -1:

            # if there was no match, assume it is blocked;

            lane.head_position_delta = lane.length - v.position - (v.length * 0.5)
            lane.head_speed_delta = v.speed

        else:

            next_lane = lane.next_lane[lane.curr_next_lane]

            if isinstance(next_lane, MicroLane):

                # if there is any vehicle in the next lane, use it as leading vehicle;

                if len(next_lane.curr_vehicle):

                    nv = next_lane.curr_vehicle[0]

                    lane.head_position_delta = (lane.length + nv.position - v.position) - (v.length + nv.length) * 0.5
                    lane.head_speed_delta = v.speed - nv.speed

            elif isinstance(next_lane, MacroLane):

                # accumulate density until some threshold;

                THRESHOLD = 5.0

                accum = 0.0

                for i, cell in enumerate(next_lane.curr_cell):

                    next_accum = accum + cell.state.q.r * next_lane.cell_length

                    if next_accum >= THRESHOLD:

                        # simply use mid point of the cell as leading vehicle position;
                        # also use the speed of the cell to compute speed delta;

                        cell_pos = (i + 0.5) * next_lane.cell_length

                        lane.head_position_delta = (lane.length + cell_pos - v.position) - (v.length * 0.5)
                        lane.head_speed_delta = v.speed - cell.state.u

            else:

                raise ValueError()

    lane.head_position_delta = max(lane.head_position_delta, 0.0)