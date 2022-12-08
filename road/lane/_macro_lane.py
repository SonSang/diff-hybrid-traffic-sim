'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''
import math
import torch as th
import numpy as np

from road.lane._base_lane import BaseLane
from road.lane._micro_lane import MicroLane
from model.macro._arz import ARZ

from typing import List, Union

class MacroLane(BaseLane):
    
    '''
    Lane that simulates traffic flow using macroscopic ARZ model.

    Use automatic differentiation for differentiation.
    '''

    class Cell:
        
        '''
        A lane is divided into some number of cells. Each cell
        is specified by a starting point and an ending point in 
        the lane. Each cell stores its state, in this case as
        specified in ARZ model.
        '''
        
        def __init__(self, start, end, speed_limit):
            self.start = start
            self.end = end
            self.state: ARZ.FullQ = ARZ.FullQ(speed_limit)
            
    def __init__(self, id: int, lane_length: float, speed_limit: float, cell_length: float):
        
        super().__init__(id, lane_length, speed_limit)
        
        # use (desired) cell length to compute number of cells;

        self.num_cell = math.ceil(self.length / cell_length)
        assert self.num_cell > 0, "Number of cells in a road must be larger than 0."
        self.cell_length = self.length / self.num_cell

        # initialize current (or initial) cells;
        
        self.curr_cell = [MacroLane.Cell(self.cell_length * i, self.cell_length * (i + 1), speed_limit) 
                                for i in range(self.num_cell)]

        # initialize next cells for storing next cell values;

        self.next_cell = [MacroLane.Cell(self.cell_length * i, self.cell_length * (i + 1), speed_limit) 
                                for i in range(self.num_cell)]

        # leftmost and rightmost cell, which are used if there is no connected lane;
        # by default, they do not have any flow;
        
        self.leftmost_cell = MacroLane.Cell(0, 0, self.speed_limit)
        self.rightmost_cell = MacroLane.Cell(self.cell_length, self.cell_length, self.speed_limit)

        # store riemann solutions in advancing single time step;
        
        self.riemann_solution: List[ARZ.Riemann] = None

        # flux capacitor used for hybrid simulation;
        
        self.flux_capacitor = th.zeros((1,), dtype=th.float32)

    def is_macro(self):
        
        return True

    def is_micro(self):
        
        return False

    def forward(self, delta_time: float):

        '''
        Take a single forward simulation step by computing cell state values of next time step.

        Note that the new state values are stored in [next_cell]; call [update_state] to apply them to [curr_cell].
        '''        

        next_cell = self.next_cell
        num_cell = self.num_cell
        cell_length = self.cell_length

        # compute Riemann solutions;

        self._solve_riemann(delta_time)
        
        update_coefficient = delta_time / cell_length

        # update Q state values using Riemann solutions;

        for cell in range(num_cell):

            _cell = next_cell[cell]

            # flux;
        
            _cell.state.q = self.curr_cell[cell].state.q +              \
                (self.riemann_solution[cell].Q_0.flux() -                   \
                self.riemann_solution[cell + 1].Q_0.flux()) *               \
                update_coefficient

            _cell.state.set_r_y(_cell.state.q.r, _cell.state.q.y, self.speed_limit)

    def _solve_riemann(self, delta_time: float):

        num_cell = self.num_cell

        cell_length = self.cell_length
        speed_limit = self.speed_limit

        # collect Riemann solutions;

        self.riemann_solution = []
        
        for interface in range(num_cell + 1):

            # get left and right cell of the given interface;

            left_Q = self.get_left_cell(interface).state
            right_Q = self.get_right_cell(interface - 1).state

            # solve ARZ system and store;
            
            rs = ARZ.riemann_solve(left_Q, right_Q, speed_limit)
            self.riemann_solution.append(rs)

            # check if time step satisfies the CFL condition;
            
            EPS = 1e-5
            speed0 = max(abs(rs.speed0), EPS)
            speed1 = max(abs(rs.speed1), EPS)
            
            assert (delta_time < cell_length / speed0) and (delta_time < cell_length / speed1), \
                    "Time step size does not meet CFL condition. Please try smaller delta_time."


    def which(self, pos):
        
        '''
        Return index of the cell where the given [pos] is located.
        '''
        return math.floor(pos / self.cell_length)

    def set_leftmost_cell(self, r, u):

        self.leftmost_cell.state = ARZ.FullQ.from_r_u(r, u, self.speed_limit)

    def set_rightmost_cell(self, r, u):

        self.rightmost_cell.state = ARZ.FullQ.from_r_u(r, u, self.speed_limit)

    def get_leftmost_cell(self):
        
        '''
        If there is prev lane, use its info to compute left cell; if not, use [self.left_cell].
        '''
        
        if self.prev_lane is not None:

            if self.prev_lane.is_macro():
            
                p_lane: MacroLane = self.prev_lane
                left_cell = p_lane.curr_cell[-1]
            
            else:
            
                # assume vacant cell if prev lane is micro lane;

                left_cell = ARZ.FullQ.from_r_u(0.0, self.speed_limit, self.speed_limit)

        else:

            left_cell = self.leftmost_cell

        return left_cell

    def get_rightmost_cell(self):

        '''
        If there is next lane, use its info to compute right cell; if not, use [self.right_cell].
        '''

        if self.next_lane is not None:

            if self.next_lane.is_macro():

                n_lane: MacroLane = self.next_lane
                
                return n_lane.curr_cell[0]

            else:

                # if next lane is micro lane, accumulate discrete vehicle's
                # states to get aggregated macro state;

                n_lane: MicroLane = self.next_lane

                min_lane_length = self.cell_length
                
                sum_lane_length = 0
                sum_num_vehicle = 0

                avg_density = 0
                avg_speed = 0

                while True:
                    
                    sum_lane_length += n_lane.length
                    sum_num_vehicle += n_lane.num_vehicle()

                    avg_density += n_lane.length * n_lane.avg_density()
                    avg_speed += n_lane.num_vehicle() * n_lane.avg_speed()

                    n_lane = n_lane.next_lane

                    if sum_lane_length > min_lane_length or \
                        n_lane is None or \
                        n_lane.is_macro():

                        break

                avg_density = avg_density / sum_lane_length
                avg_speed = avg_speed / sum_num_vehicle

                virtual_cell = MacroLane.Cell(0, sum_lane_length)
                virtual_cell.state = ARZ.FullQ.from_r_u(avg_density, avg_speed, self.next_lane.speed_limit)

                return virtual_cell

        else:

            right_cell = self.rightmost_cell

        return right_cell
        
            
    def get_left_cell(self, id):
        
        '''
        Get left cell of given cell.
        '''

        if id == 0:
            
            return self.get_leftmost_cell()
        
        else:

            return self.curr_cell[id - 1]

    
    def get_right_cell(self, id):
        
        '''
        Get right cell of given cell.
        '''

        if id == self.num_cell - 1:
            
            return self.get_rightmost_cell()
        
        else:

            return self.curr_cell[id + 1]

    
    def update_state(self):
        
        '''
        Update current state with next state.
        '''

        for i in range(self.num_cell):
            self.curr_cell[i].state.q.r = self.next_cell[i].state.q.r
            self.curr_cell[i].state.q.y = self.next_cell[i].state.q.y
            self.curr_cell[i].state.u = self.next_cell[i].state.u
            self.curr_cell[i].state.u_eq = self.next_cell[i].state.u_eq
        
        # update flux capacitor if next lane is micro lane;

        if self.next_lane is not None and self.next_lane.is_micro():
            self.flux_capacitor += self.curr_cell[-1].state.q.r * self.curr_cell[-1].state.u

    def set_state_vector_y(self, 
                            rv: Union[th.Tensor, np.ndarray], 
                            yv: Union[th.Tensor, np.ndarray]):

        '''
        Set cell state from given vector, of which length equals to number of cells.

        Accept relative flow (y) as input.

        @rv: Density vector
        @yv: Relative flow vector
        '''

        assert len(rv) == self.num_cell, "Cell number mismatch"
        assert len(yv) == self.num_cell, "Cell number mismatch"

        for i in range(self.num_cell):
            self.curr_cell[i].state.set_r_y(rv[i], yv[i], self.speed_limit)

    def set_state_vector_u(self, 
                            rv: Union[th.Tensor, np.ndarray], 
                            uv: Union[th.Tensor, np.ndarray]):

        '''
        Set cell state from given vector, of which length equals to number of cells.

        Accept speed (u) as input.

        @rv: Density vector
        @uv: Speed vector
        '''

        assert len(rv) == self.num_cell, "Cell number mismatch"
        assert len(uv) == self.num_cell, "Cell number mismatch"

        for i in range(self.num_cell):
            self.curr_cell[i].state.set_r_u(rv[i], uv[i], self.speed_limit)

    def get_state_vector(self):

        '''
        Get state vector in the order of density, relative flow, and speed.
        '''

        r = th.zeros((self.num_cell,), dtype=th.float32)
        y = th.zeros((self.num_cell,), dtype=th.float32)
        u = th.zeros((self.num_cell), dtype=th.float32)

        for i in range(self.num_cell):
            r[i] = self.curr_cell[i].state.q.r
            y[i] = self.curr_cell[i].state.q.y
            u[i] = self.curr_cell[i].state.u

        return r, y, u

    def set_next_state_vector_y(self, 
                                rv: Union[th.Tensor, np.ndarray], 
                                yv: Union[th.Tensor, np.ndarray]):

        '''
        Set next cell state from given vector, of which length equals to number of cells.

        Accept relative flow (y) as input.

        @rv: Density vector
        @yv: Relative flow vector
        '''

        assert len(rv) == self.num_cell, "Cell number mismatch"
        assert len(yv) == self.num_cell, "Cell number mismatch"

        for i in range(self.num_cell):
            self.next_cell[i].state.set_r_y(rv[i], yv[i], self.speed_limit)

    def set_next_state_vector_u(self, 
                            rv: Union[th.Tensor, np.ndarray], 
                            uv: Union[th.Tensor, np.ndarray]):

        '''
        Set next cell state from given vector, of which length equals to number of cells.

        Accept speed (u) as input.

        @rv: Density vector
        @uv: Speed vector
        '''

        assert len(rv) == self.num_cell, "Cell number mismatch"
        assert len(uv) == self.num_cell, "Cell number mismatch"

        for i in range(self.num_cell):
            self.next_cell[i].state.set_r_u(rv[i], uv[i], self.speed_limit)


    def get_next_state_vector(self):

        '''
        Get next state vector in the order of density, relative flow, and speed.
        '''

        r = th.zeros((self.num_cell,), dtype=th.float32)
        y = th.zeros((self.num_cell,), dtype=th.float32)
        u = th.zeros((self.num_cell), dtype=th.float32)

        for i in range(self.num_cell):
            r[i] = self.next_cell[i].state.q.r
            y[i] = self.next_cell[i].state.q.y
            u[i] = self.next_cell[i].state.u

        return r, y, u

    def clear(self):

        '''
        Clear current and next cells, so that each cell has zero density and maximum speed.
        '''

        for cell in self.curr_cell:

            cell.state.clear()

        for cell in self.next_cell:

            cell.state.clear()