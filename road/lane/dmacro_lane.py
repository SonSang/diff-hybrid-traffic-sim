'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''

import torch as th
import numpy as np

from road.lane._macro_lane import MacroLane
from model.macro.darz import dARZ
from typing import List

class dMacroLane(MacroLane):
    
    '''
    Lane that simulates traffic flow using macroscopic ARZ model.

    Use analytical gradients for differentiation.
    '''

    class dLane:

        '''
        Store Jacobian matrices of each cell in the next time 
        step w.r.t. each cell in the previous time step.

        In solving the ARZ system using FVM, as far as the CFL 
        condition is met, the state of a cell (i) in time
        step (n + 1) only depends on the states of the cells
        (i - 1), (i), (i + 1) in time step (n). Therefore, 
        we can compute and store the partial derivatives of
        Q(i, n + 1) w.r.t. Q(i - 1, n), Q(i, n), and
        Q(i + 1, n). Each of these derivatives are 2 x 2
        Jacobian matrix, as cell state value is represented
        with 2 variables (rho and y).

        We store these Jacobian matrices in following layout:

            [ next cell id, prev cell id, dQ(2, 2) ]

        Next cell id = 0 ~ Num. Cell
        Prev cell id = 0 (left cell), 1 (that cell), 2 (right cell)

        dQ = 
            [ dr(n + 1) / dr(n), dr(n + 1) / dy(n) ]
            [ dy(n + 1) / dr(n), dy(n + 1) / dy(n) ]

        '''

        def __init__(self, num_cell):

            # dqs[a, 0] : Gradient of [a] w.r.t. [a - 1]
            # dqs[a, 1] : Gradient of [a] w.r.t. [a + 0]
            # dqs[a, 2] : Gradient of [a] w.r.t. [a + 1]

            self.dqs = np.zeros((num_cell, 3, 2, 2), dtype=np.float32)


    def __init__(self, id: int, lane_length: float, speed_limit: float, cell_length: float):
        
        super().__init__(id, lane_length, speed_limit, cell_length)

        self.d_lane: List[dMacroLane.dLane] = []

        # backup cell;
        self.b_curr_cell: List[MacroLane.Cell] = []

    def forward(self, delta_time: float):

        '''
        Take a single forward simulation step by computing cell state values of next time step.

        Note that the new state values are stored in [next_cell]; call [update_state] to apply them to [curr_cell].
        '''        

        # generate tensor input that would be fed to computation layer;

        cr, cy = self.vectorize_input()

        # feed inputs to computation layer;
        # results include next density and relative flow;

        nr, ny = dMacroForwardLayer.apply(self, cr, cy, delta_time)

        self.set_next_state_vector_y(nr, ny)


    def _forward(self, delta_time: float):
        
        '''
        Take forward step by calling [MacroLane]'s forward function.
        '''

        super().forward(delta_time)

    def _backward(self, delta_time: float):

        '''
        Fill in [d_lane] with info from [riemann_solution].
        '''

        update_coefficient = delta_time / self.cell_length

        d_lane = dMacroLane.dLane(self.num_cell)

        for ci in range(self.num_cell):

            left_cell = self.get_left_cell(ci)
            this_cell = self.curr_cell[ci]
            right_cell = self.get_right_cell(ci)

            rs_L = self.riemann_solution[ci]            # left Riemann solution;
            rs_R = self.riemann_solution[ci + 1]        # right Riemann solution;

            rs_L_dL, rs_L_dR = dARZ.compute_dLdR(rs_L, left_cell.state, this_cell.state, self.speed_limit)
            rs_R_dL, rs_R_dR = dARZ.compute_dLdR(rs_R, this_cell.state, right_cell.state, self.speed_limit)

            Q0_L = rs_L.Q_0
            Q0_R = rs_R.Q_0

            fp_L = dARZ.flux_prime(Q0_L)
            fp_R = dARZ.flux_prime(Q0_R)

            # dqs;
            
            d_lane.dqs[ci, 0] = -update_coefficient * (-np.matmul(fp_L, rs_L_dL))
            d_lane.dqs[ci, 2] = -update_coefficient * (np.matmul(fp_R, rs_R_dR))
            d_lane.dqs[ci, 1] = np.eye(2, dtype=np.float32) - update_coefficient * \
                (np.matmul(fp_R, rs_R_dL) - np.matmul(fp_L, rs_L_dR))


        self.d_lane.append(d_lane)

    def vectorize_input(self):
        
        '''
        To use analytical gradients, this class vectorizes states and pass them to computation layer.

        Note that only the variables included in the vectorized states support gradients.

        Currently, we support density and relative flow of every cell, and those at boundary, in the input.
        '''

        cr, cy, _ = self.get_state_vector()

        leftmost_cell = self.get_leftmost_cell()
        rightmost_cell = self.get_rightmost_cell()

        l_r, r_r, l_y, r_y = th.zeros((1,)), th.zeros((1,)), th.zeros((1,)), th.zeros((1,))
        l_r[0] = leftmost_cell.state.q.r
        r_r[0] = rightmost_cell.state.q.r
        l_y[0] = leftmost_cell.state.q.y
        r_y[0] = rightmost_cell.state.q.y

        cr = th.cat([l_r, cr, r_r])
        cy = th.cat([l_y, cy, r_y])

        return cr, cy

    def backup_cell(self):

        '''
        Store current cell states in backup cell states.
        '''

        self.b_curr_cell = []

        self.b_curr_cell.append(self.leftmost_cell)

        for cell in self.curr_cell:

            self.b_curr_cell.append(cell)

        self.b_curr_cell.append(self.rightmost_cell)
        

    def detach_cell(self):

        '''
        Change states in current cells to plain floating point numbers.
        '''

        # backup original states first;

        self.backup_cell()

        self.leftmost_cell = dMacroLane.decell(self.leftmost_cell)

        for i, cell in enumerate(self.curr_cell):

            self.curr_cell[i] = dMacroLane.decell(cell)

        self.rightmost_cell = dMacroLane.decell(self.rightmost_cell)

    @staticmethod
    def decell(cell: MacroLane.Cell):

        '''
        Change cell states in pytorch Tensor to plain floating point number.
        '''

        nc = MacroLane.Cell(cell.start, cell.end, cell.state.u_max)

        nc.state.u_max = cell.state.u_max
        nc.state.u = cell.state.u
        nc.state.u_eq = cell.state.u_eq
        nc.state.q.r = cell.state.q.r
        nc.state.q.y = cell.state.q.y

        if isinstance(nc.state.u, th.Tensor):

            nc.state.u = nc.state.u.item()
        
        if isinstance(nc.state.u_eq, th.Tensor):

            nc.state.u_eq = nc.state.u_eq.item()

        if isinstance(nc.state.q.r, th.Tensor):

            nc.state.q.r = nc.state.q.r.item()
        
        if isinstance(nc.state.q.y, th.Tensor):

            nc.state.q.y = nc.state.q.y.item()

        return nc

        
class dMacroForwardLayer(th.autograd.Function):

    @staticmethod
    def forward(ctx, lane: dMacroLane, r: th.Tensor, y: th.Tensor, delta_time: float):

        '''
        Note that tensor inputs are just placeholders; we assume they are
        already stored in [lane] for acceleration. Therefore, do not
        use this layer elsewhere than [dMacroLane.forward]!
        '''

        '''
        For acceleration, we do not use pytorch Tensor in computation.
        Therefore, before computation, we backup original cell states
        and restore them after every computation is completed.
        '''

        lane.detach_cell()
        
        # take forward step;
        
        lane._forward(delta_time)

        # compute gradient info for differentiation;

        lane._backward(delta_time)

        # recover original states;

        lane.leftmost_cell = lane.b_curr_cell[0]
        lane.curr_cell = lane.b_curr_cell[1:-1]
        lane.rightmost_cell = lane.b_curr_cell[-1]

        # set context;

        ctx.dqs = lane.d_lane[-1].dqs

        # return state vector r and y, and fluxes at boundary;

        nr, ny, _ = lane.get_next_state_vector()
    
        return nr, ny

    @staticmethod
    def backward(ctx, grad_nr: th.Tensor, grad_ny: th.Tensor):

        # compute gradients in our lane;

        dqs: np.ndarray = ctx.dqs
        dqs = dqs.transpose((0, 1, 3, 2))

        num_cell = len(grad_nr)

        # compute gradients to propagate;

        grad_ncell = np.zeros((num_cell, 1, 2, 1), dtype=np.float32)
        grad_ncell[:, 0, 0, 0] = grad_nr.numpy()
        grad_ncell[:, 0, 1, 0] = grad_ny.numpy()

        grad_cell = np.matmul(dqs, grad_ncell)  # transpose needed;
        grad_cell = np.squeeze(grad_cell, axis=-1)

        grad_ry = np.zeros((num_cell + 2, 2), dtype=np.float32)
        grad_ry[1:-1] = grad_cell[:, 1, :]
        grad_ry[2:-1] += grad_cell[:-1, 2, :]
        grad_ry[1:-2] += grad_cell[1:, 0, :]
        
        # boundaries;
        grad_ry[0] = grad_cell[0, 0, :]
        grad_ry[-1] = grad_cell[-1, 2, :]

        grad_r = th.tensor(grad_ry[:, 0])
        grad_y = th.tensor(grad_ry[:, 1])

        assert not th.any(th.isnan(grad_r)) and not th.any(th.isnan(grad_y)), ""
        
        return None, grad_r, grad_y, None