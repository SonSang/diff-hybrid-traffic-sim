import torch as th

from road.lane._macro_lane import MacroLane
from road.lane._micro_lane import MicroLane
from road.vehicle.micro_vehicle import MicroVehicle

from typing import Union

class Hybrid:
    
    '''
    This class provides differentiable scheme to convert between
    macroscopic and microscopic lanes. 

    Here we use ancillary variable "a" here to enhance gradient
    flow across multiple heterogeneous lanes. For instance, if
    there is macro-micro-macro lane, without "a", the gradient
    only flows through speed terms, which restrict the quality
    of the gradients. However, by using "a" term, we can let 
    gradients to flow across density terms, too.
    '''

    @staticmethod
    def macro_to_micro(prev_lane: MacroLane, 
                        next_lane: MicroLane, 
                        nv_accel_max: Union[float, th.Tensor],
                        nv_accel_pref: Union[float, th.Tensor],
                        nv_target_speed: Union[float, th.Tensor],
                        nv_min_space: Union[float, th.Tensor],
                        nv_time_pref: Union[float, th.Tensor],
                        nv_vehicle_length: Union[float, th.Tensor]):

        '''
        If prev lane's last cell accumulates enough flux capacitor,
        it emits a new vehicle in to next lane. 
        
        Since the new vehicle's speed is determined by the speed of
        the last cell's speed, which depends on both density and 
        relative flow of it, the new vehicle's future states would 
        depend partially on the states of the last cell and gradients 
        can flow across these states.  

        However, note that we use discrete operator to compare flux
        capacitor value and new vehicle's length in generating the
        new vehicle. Therefore, the gradient could be discontinuous.
        To amend this issue, we could use soft functions like sigmoid
        to compare the values and use the results in generated vehicle.
        However, we did not include it here.
        '''

        flux = prev_lane.flux_capacitor.item()

        # if flux is accumulated enough and there is free space in the next lane, create a new vehicle;

        if flux >= nv_vehicle_length and \
            next_lane.entering_free_space() > (nv_vehicle_length * 0.5):

            # use ancilliary variable [a] to let gradient flow;
            # [a] gets same value as the length of the new vehicle;

            nv_vehicle_a = prev_lane.flux_capacitor - (flux - nv_vehicle_length)

            nv_curr_position = th.zeros((1,), dtype=th.float32)
            nv_curr_speed = prev_lane.curr_cell[-1].state.u

            # update lane;

            prev_lane.flux_capacitor -= nv_vehicle_length

            nv = next_lane.random_vehicle()
            
            nv.position = nv_curr_position
            nv.speed = nv_curr_speed
            nv.a = nv_vehicle_a

            next_lane.add_tail_vehicle(nv)
            

    @staticmethod
    def micro_to_macro(prev_lane: MicroLane, next_lane: MacroLane):
        
        '''
        If prev lane's head vehicle traverses end of the lane, 
        remove it and apply its states to next lane's first cell.

        Since the first cell's speed is determined by the speed of
        the entering vehicle's speed, the continuing cell's future 
        states would depend partially on the states of the vehicle 
        and gradients can flow across these states.

        However, it also suffers from discontinuous gradient across 
        position and density, as we use discrete operator.
        '''

        if prev_lane.num_vehicle() == 0:
            return

        head_position = prev_lane.curr_vehicle[-1].position
        head_speed = prev_lane.curr_vehicle[-1].speed

        if head_position >= prev_lane.length:
            
            # use ancilliary variable [a] to let gradient flow;

            head_a = prev_lane.curr_vehicle[-1].a
            head_length = prev_lane.curr_vehicle[-1].length
            d_head_length = head_length

            if isinstance(d_head_length, th.Tensor):
                d_head_length = d_head_length.item()

            # cell variables;

            p_r = next_lane.curr_cell[0].state.q.r
            p_u = next_lane.curr_cell[0].state.u
            
            next_cell_length = next_lane.cell_length

            overlap_size = th.clamp(head_position - prev_lane.length + head_length * 0.5, max=next_cell_length)
            add_r = (head_a / d_head_length) * (overlap_size / next_cell_length)
            n_r = p_r + add_r

            # differentiable clamp;

            d_n_r = n_r
            if isinstance(d_n_r, th.Tensor):
                d_n_r = d_n_r.item()

            if n_r > 1.0 - 1e-5:
                n_r = n_r - (d_n_r - (1.0 - 1e-5))
            elif n_r < 1e-5:
                n_rho = n_r - (d_n_r - 1e-5)

            # update lane;

            prev_lane.curr_vehicle = prev_lane.curr_vehicle[:-1]

            next_lane.curr_cell[0].state.q.r = n_rho
            next_lane.curr_cell[0].state.u = \
                ((next_cell_length * p_r * p_u) + (overlap_size * head_speed)) / \
                (next_cell_length * p_r + overlap_size)

    @staticmethod
    def micro_to_micro(prev_lane: MicroLane, next_lane: MicroLane):
        
        if prev_lane.num_vehicle() == 0:
            
            return

        hv = prev_lane.curr_vehicle[-1]
        head_position = hv.position

        if head_position >= prev_lane.length:

            head_position = head_position - prev_lane.length

            hv.position = head_position
            
            next_lane.add_tail_vehicle(hv)

            prev_lane.curr_vehicle = prev_lane.curr_vehicle[:-1]