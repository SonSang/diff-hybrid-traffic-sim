'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''
import math
import torch as th

from road.lane.base_lane import BaseLane
from model.macro.darz import dARZLayer

class MacroLane(BaseLane):
    '''
    [MacroLane] runs macroscopic traffic simulation for a single lane using ARZ model. 
    '''
    class Cell:
        '''
        A lane is divided into some number of cells. Each cell
        is specified by a starting point and an ending point in 
        the lane.
        '''
        def __init__(self, start, end):
            self.start = start
            self.end = end
            
    def __init__(self, lane_length: float, speed_limit: float, cell_length: float):
        super().__init__(lane_length, speed_limit)
        
        # use (desired) cell length to compute number of cells
        self.num_cell = math.ceil(self.length / cell_length)
        assert self.num_cell > 0, "Number of cells in a road must be larger than 0."
        self.cell_length = self.length / self.num_cell

        # initialize current (or initial) cell state
        self.curr_rho = th.zeros((self.num_cell, 1), dtype=th.float32)
        self.curr_u = th.ones((self.num_cell, 1), dtype=th.float32) * speed_limit
        
        # Initialize next cells for storing next cell values.
        self.next_rho = th.zeros((self.num_cell, 1), dtype=th.float32)
        self.next_u = th.ones((self.num_cell, 1), dtype=th.float32) * speed_limit

        # flux capacitor used for hybrid simulation
        self.flux_capacitor = th.zeros((1, 1), dtype=th.float32)

    def is_macro(self):
        return True

    def is_micro(self):
        return False

    def to(self, device, dtype):
        self.curr_rho = self.curr_rho.to(device=device, dtype=dtype)
        self.curr_u = self.curr_u.to(device=device, dtype=dtype)

        self.next_rho = self.next_rho.to(device=device, dtype=dtype)
        self.next_u = self.next_u.to(device=device, dtype=dtype)

    def num_vehicle(self):
        nv = 0
        for i in range(self.num_cell):
            nv += th.floor(self.curr_rho[i, :] * self.cell_length / 10.0).to(dtype=th.int32).detach().cpu().item()
        return nv

    def which_cell(self, pos):
        '''
        Return index of the cell where the given [pos] is located.
        '''
        return math.floor(pos / self.cell_length)

    def preprocess(self,
                    delta_time: float,
                    leftmost_rho: th.Tensor, 
                    leftmost_u: th.Tensor, 
                    rightmost_rho: th.Tensor, 
                    rightmost_u: th.Tensor,
                    ):
        # For given array of cell information, generate a set of tensors that can be fed into ARZLayer.
        # It assumes that n-th element is located left to the (n+1)-th element, and right to the (n-1)-th element.
        num_cell = self.num_cell
        rho = self.curr_rho
        u = self.curr_u

        l_rho, l_u = th.zeros_like(rho), th.zeros_like(u)
        r_rho, r_u = th.zeros_like(rho), th.zeros_like(u)
        
        l_rho[1:], l_u[1:] = rho[:num_cell - 1], u[:num_cell - 1]
        r_rho[:num_cell - 1], r_u[:num_cell - 1] = rho[1:], u[1:]

        l_rho[0, 0], l_u[0, 0] = leftmost_rho, leftmost_u
        r_rho[num_cell - 1, 0], r_u[num_cell - 1, 0] = rightmost_rho, rightmost_u

        speed_limit = th.ones_like(rho) * self.speed_limit
        cell_length = th.ones_like(rho) * self.cell_length
        delta_time = th.ones_like(rho) * delta_time
        
        return rho, u, l_rho, l_u, r_rho, r_u, speed_limit, cell_length, delta_time
            
    def forward_step(self, 
                    delta_time: float, 
                    leftmost_rho: th.Tensor, 
                    leftmost_u: th.Tensor, 
                    rightmost_rho: th.Tensor, 
                    rightmost_u: th.Tensor, 
                    parallel: bool = True):
        '''
        take a single forward simulation step by computing cell state values of next time step
        '''
        if self.prev_lane is not None:
            if self.prev_lane.is_macro():
                p_lane: MacroLane = self.prev_lane
                leftmost_rho = p_lane.curr_rho[-1, :]
                leftmost_u = p_lane.curr_u[-1, :]
                pass
            else:
                # assume vacant cell if adjacent lane is micro lane
                leftmost_rho = th.zeros((1, 1))
                leftmost_u = th.ones((1, 1)) * self.speed_limit
        
        if self.next_lane is not None:
            if self.next_lane.is_macro():
                n_lane: MacroLane = self.next_lane
                rightmost_rho = n_lane.curr_rho[0, :]
                rightmost_u = n_lane.curr_u[0, :]
                pass
            else:
                # assume vacant cell if adjacent lane is micro lane
                rightmost_rho = th.zeros((1, 1))
                rightmost_u = th.ones((1, 1)) * self.speed_limit
        
        # take a single step
        rho, u, l_rho, l_u, r_rho, r_u, speed_limit, cell_length, delta_time = \
            self.preprocess(delta_time, leftmost_rho, leftmost_u, rightmost_rho, rightmost_u)

        if parallel:
            self.next_rho, self.next_u = dARZLayer.apply(rho, u, l_rho, l_u, r_rho, r_u, cell_length, speed_limit, delta_time)
        else:
            for i in range(self.num_cell):
                self.next_rho[[i], :], self.next_u[[i], :] = dARZLayer.apply(rho[[i], :],
                                                                            u[[i], :], 
                                                                            l_rho[[i], :],
                                                                            l_u[[i], :], 
                                                                            r_rho[[i], :], 
                                                                            r_u[[i], :], 
                                                                            cell_length[[i], :], 
                                                                            speed_limit[[i], :], 
                                                                            delta_time[[i], :])


    def update_state(self):
        '''
        update current state with next state
        '''
        # update flux capacitor
        if self.next_lane is not None and self.next_lane.is_micro():
            self.flux_capacitor += self.curr_rho[-1, 0] * self.curr_u[-1, 0]

        self.curr_rho = self.next_rho
        self.curr_u = self.next_u

    def print(self):
        print("Macro Lane: # cell = {} / cell length = {:.2f} m / lane length = {:.2f} m / speed_limit = {:.2f} m/sec".format(
            self.num_cell,
            self.cell_length,
            self.length,
            self.speed_limit,
        ))

        print("density: {}".format(list(self.curr_rho.detach().cpu().numpy()[:, 0])))
        print("speed: {}".format(list(self.curr_u.detach().cpu().numpy()[:, 0])))
        print("flux: {}".format(self.flux_capacitor.detach().cpu().item()))