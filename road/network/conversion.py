import torch as th

from road.lane._macro_lane import MacroLane
from road.lane._micro_lane import MicroLane, MicroVehicle
from road.network.route import MicroRoute
from model.macro._arz import ARZ

class Conversion:

    @staticmethod
    def macro_to_macro(network, prev_lane: MacroLane, next_lane: MacroLane):

        pass

    @staticmethod
    def macro_to_micro(network, prev_lane: MacroLane, next_lane: MicroLane, delta_time: float):

        '''
        First add flux to [prev_lane]'s flux capacitor that corresponds
        to [next_lane]. If [prev_lane]'s last cell accumulates enough flux 
        capacitor, it emits a new vehicle in to [next_lane]. 
        
        Since the new vehicle's speed is determined by the speed of
        the last cell's speed, which depends on both density and 
        relative flow of it, the new vehicle's future states would 
        depend partially on the states of the last cell and gradients 
        can flow across these states.
        '''

        # add flux to flux capacitor for the next lane;

        flux_inc = prev_lane.curr_cell[-1].state.q.r * prev_lane.curr_cell[-1].state.u * delta_time

        prev_lane.add_flux_capacitor(next_lane.id, flux_inc)

        # check flux and generate new vehicle;

        flux = prev_lane.flux_capacitor[next_lane.id]

        if isinstance(flux, th.Tensor):

            tflux = flux.clone()
            flux = flux.item()
            
        else:
            
            tflux = flux

        # if flux is accumulated enough and there is free space in the next lane, create a new vehicle;

        nv = MicroVehicle.default_micro_vehicle(next_lane.speed_limit)

        # bigger entering free space (from 0.5 to 1.0) to prevent emergence break if possible;
        
        if flux >= nv.length and next_lane.entering_free_space() >= (nv.length * 1.0):

            # use ancilliary variable [a] to let gradient flow;
            # [a] gets same value as the length of the new vehicle;

            nv.position = 0
            nv.speed = prev_lane.curr_cell[-1].state.u
            nv.a = tflux - (flux - nv.length)

            # detach flux;
            if isinstance(tflux, th.Tensor):
                prev_lane.flux_capacitor[next_lane.id] = th.tensor(tflux.detach().cpu().item() - nv.length, dtype=tflux.dtype, device=tflux.device)
            else:
                prev_lane.flux_capacitor[next_lane.id] = tflux - nv.length

            # update network;

            nr = network.create_random_route(next_lane.id)
            network.add_vehicle(nv, nr)

    @staticmethod
    def micro_to_macro(network, prev_lane: MicroLane):

        '''
        If prev lane's head vehicle traverses end of the lane, 
        remove it and apply its states to next lane's first cell.

        Since the first cell's speed is determined by the speed of
        the entering vehicle's speed, the continuing cell's future 
        states would depend partially on the states of the vehicle 
        and gradients can flow across these states.
        '''

        if prev_lane.num_vehicle():

            # check if head vehicle goes out of the lane;

            hv = prev_lane.get_head_vehicle()
            hr: MicroRoute = network.micro_route[hv.id]

            next_lane: MacroLane = network.lane[hr.next_lane_id()]

            assert next_lane.is_macro(), ""

            if hv.position > prev_lane.length + (1.0 * hv.length):

                # remove from current lane;

                prev_lane.curr_vehicle = prev_lane.curr_vehicle[:-1]

                # add to next lane;

                # use ancilliary variable [a] to let gradient flow;

                head_a = hv.a
                head_length = hv.length
                d_head_length = head_length

                if isinstance(d_head_length, th.Tensor):
                    
                    d_head_length = d_head_length.item()
                    
                v_head = hv.position - prev_lane.length # + (0.5 * hv.length)
                v_tail = v_head - hv.length

                # cell variables;

                next_cell_length = next_lane.cell_length
                
                for ci in range(len(next_lane.curr_cell)):
                    
                    p_r = next_lane.curr_cell[ci].state.q.r
                    p_u = next_lane.curr_cell[ci].state.u
                    
                    c_head = next_lane.curr_cell[ci].end
                    c_tail = next_lane.curr_cell[ci].start
                    
                    if c_head > v_tail and c_tail < v_head:

                        max_head = c_head if c_head > v_head else v_head
                        min_tail = c_tail if c_tail < v_tail else v_tail
                        
                        overlap_size = next_cell_length + head_length - (max_head - min_tail)
                        add_r = (head_a / d_head_length) * (overlap_size / next_cell_length)
                        n_r = p_r + add_r

                        # differentiable clamp;

                        d_n_r = n_r
                        
                        if isinstance(d_n_r, th.Tensor):
                        
                            d_n_r = d_n_r.item()

                        if n_r > 1.0 - 1e-5:
                            
                            n_r = n_r - (d_n_r - (1.0 - 1e-5))
                        
                        elif n_r < 1e-5:
                        
                            n_r = n_r - (d_n_r - 1e-5)

                        # update lane;

                        next_lane.curr_cell[ci].state.q.r = n_r
                        next_lane.curr_cell[ci].state.u = hv.speed
                            # \
                            # ((next_cell_length * p_r * p_u) + (overlap_size * hv.speed)) / \
                            # (next_cell_length * p_r + overlap_size)
                        next_lane.curr_cell[ci].state.q.y = \
                            ARZ.compute_y(next_lane.curr_cell[ci].state.q.r, 
                                        next_lane.curr_cell[ci].state.u, 
                                        next_lane.speed_limit)
                            
                    else:
                        
                        break


    @staticmethod
    def micro_to_micro(network, prev_lane: MicroLane):

        if prev_lane.num_vehicle():

            # check if head vehicle goes out of the lane;

            hv = prev_lane.get_head_vehicle()
            hr: MicroRoute = network.micro_route[hv.id]

            next_lane: MicroLane = network.lane[hr.next_lane_id()]

            assert next_lane.is_micro(), ""

            if hv.position >= prev_lane.length:

                # remove from current lane;

                prev_lane.curr_vehicle = prev_lane.curr_vehicle[:-1]

                # add to next lane;

                hv.position = hv.position - prev_lane.length

                next_lane.add_tail_vehicle(hv)

                hr.increment_curr_idx()

    @staticmethod
    def micro_to_none(network, prev_lane: MicroLane):

        if prev_lane.num_vehicle():

            # check if head vehicle goes out of the lane;

            hv = prev_lane.get_head_vehicle()

            if hv.position >= prev_lane.length:

                # remove from current lane;

                prev_lane.curr_vehicle = prev_lane.curr_vehicle[:-1]