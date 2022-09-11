import torch as th

from road.lane.macro_lane import MacroLane
from road.lane.micro_lane import MicroLane

class Hybrid:
    @staticmethod
    def macro_to_micro(prev_lane: MacroLane, 
                        next_lane: MicroLane, 
                        nv_accel_max: th.Tensor,
                        nv_accel_pref: th.Tensor,
                        nv_target_vel: th.Tensor,
                        nv_min_space: th.Tensor,
                        nv_time_pref: th.Tensor,
                        nv_vehicle_length: th.Tensor):
        '''
        if prev lane's last cell accumulates enough flux capacitor, emits a new vehicle in to next lane
        '''
        flux = prev_lane.flux_capacitor.detach().item()

        # if flux is accumulated enough and there is free space in the next lane, create a new vehicle
        if flux >= nv_vehicle_length.detach().item() and \
                (next_lane.num_vehicle() == 0 or \
                next_lane.curr_pos[0, 0] > (next_lane.vehicle_length[0, 0] + nv_vehicle_length[0, 0]) * 0.5):

            # use ancilliary variable [a] to let gradient flow
            nv_vehicle_a = prev_lane.flux_capacitor - (flux - nv_vehicle_length)

            nv_curr_pos = th.zeros((1, 1), dtype=th.float32)
            nv_curr_vel = prev_lane.curr_u[[-1], :] * 1e-3

            # update lane
            prev_lane.flux_capacitor -= nv_vehicle_length
            next_lane.add_vehicle(nv_curr_pos, 
                                    nv_curr_vel, 
                                    nv_accel_max, 
                                    nv_accel_pref, 
                                    nv_target_vel, 
                                    nv_min_space, 
                                    nv_time_pref, 
                                    nv_vehicle_length, 
                                    nv_vehicle_a)

    @staticmethod
    def micro_to_macro(prev_lane: MicroLane, next_lane: MacroLane):
        '''
        if prev lane's head vehicle comes across end of the lane, remove it and apply its state to next lane's first cell
        '''
        if prev_lane.num_vehicle() == 0:
            return

        head_pos = prev_lane.curr_pos[[-1], :]
        head_vel = prev_lane.curr_vel[[-1], :]

        if head_pos >= prev_lane.length:
            # use ancilliary variable [a] to let gradient flow
            vehicle_a = prev_lane.vehicle_a[[-1], :]
            vehicle_length = prev_lane.vehicle_length[[-1], :]

            overlap_size = th.clamp(head_pos - prev_lane.length + vehicle_length * 0.5, max=next_lane.cell_length)
            add_rho = (vehicle_a / vehicle_length.detach()) * (overlap_size / next_lane.cell_length)

            n_rho = next_lane.curr_rho[[0], :] + add_rho

            # differentiable clamp
            if n_rho > 1.0 - 1e-5:
                n_rho = n_rho - (n_rho.detach() - (1.0 - 1e-5))
            elif n_rho < 1e-5:
                n_rho = n_rho - (n_rho.detach() - 1e-5)

            # update lane
            prev_lane.remove_head_vehicle()

            next_lane.curr_rho[[0], :] = n_rho
            next_lane.curr_u[[0], :] += head_vel * 1e-3