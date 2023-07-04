'''
@ author: Yiling Qiao & SonSang (Sanghyun Son)
@ email: shh1295@gmail.com

Script to solve inverse problem in hybrid traffic simulation.
'''

import argparse
import time
import torch as th

from example.inverse._inverse import InverseProblem
from road.network.road_network import RoadNetwork
from road.lane.dmacro_lane import dMacroLane
from road.lane.dmicro_lane import dMicroLane

class HybridInverseProblem(InverseProblem):

    def __init__(self, 
                num_trial: int, 
                num_timestep: int, 
                num_episode: int,
                delta_time: float, 
                speed_limit: float,
                run_name: str, 
                num_cell: int,
                cell_length: float):

        super().__init__(num_trial, num_timestep, num_episode, delta_time, speed_limit, run_name)

        self.num_cell = num_cell
        self.cell_length = cell_length
        self.lane_length = num_cell * cell_length
        
        self.dtype = th.float32

    def init_network(self):

        '''
        Randomly initialize road network to simulate.
        '''

        num_cell = self.num_cell
        cell_length = self.cell_length
        speed_limit = self.speed_limit
        lane_length = self.lane_length

        # initialize density and speed of boundary cell randomly;
        
        bdry_density = th.rand((4,), dtype=self.dtype)
        bdry_speed = th.rand((4,), dtype=self.dtype) * speed_limit
        
        # create network;
        
        self.network = RoadNetwork(speed_limit)
        
        # create first macro lane;
        
        lane = dMacroLane(0, lane_length, speed_limit, cell_length)
        lane.set_leftmost_cell(bdry_density[0], bdry_speed[0])
        lane.set_rightmost_cell(bdry_density[1], bdry_speed[1])
        self.network.add_lane(lane)

        # initialize density and speed of inner cell randomly;
        
        init_density, init_speed = self.random_initial_state()
        lane.set_state_vector_u(init_density, init_speed)
        
        # create second micro lane;
        
        lane = dMicroLane(1, lane_length, speed_limit)
        self.network.add_lane(lane)
        
        # create third macro lane;
        lane = dMacroLane(2, lane_length, speed_limit, cell_length)
        lane.set_leftmost_cell(bdry_density[2], bdry_speed[2])
        lane.set_rightmost_cell(bdry_density[3], bdry_speed[3])
        self.network.add_lane(lane)

        self.network.connect_lane(0, 1)
        self.network.connect_lane(1, 2)
        self.network.macro_route = self.network.create_random_macro_route()

    def random_initial_state(self):

        '''
        Get random state of the road network.
        
        If [self.beg_state] is not None, just add some noise to it.
        '''
        
        num_cell = self.num_cell
        dtype = self.dtype

        if self.beg_state is None:

            speed_limit = th.tensor([self.speed_limit], dtype=dtype)

            # density and speed of inner cells;

            inner_density = th.rand((num_cell,), dtype=dtype)
            inner_speed = th.rand((num_cell,), dtype=dtype) * speed_limit
            
        else:
            
            inner_density = self.beg_state[0]
            inner_speed = self.beg_state[1]
            
            inner_density = inner_density + th.randn((num_cell), dtype=dtype) * 1e-2
            inner_speed = inner_speed + th.randn((num_cell), dtype=dtype) * 1e-2
            
            inner_density = th.clamp(inner_density, min=0., max=1.)
            inner_speed = th.clamp(inner_speed, min=0., max=self.speed_limit)

        return (inner_density, inner_speed)

    def set_initial_state(self, state):

        '''
        Set state of the road network.
        '''

        lane: dMacroLane = self.network.lane[0]
        lane.set_state_vector_u(state[0], state[1])
    
    def get_state(self):

        '''
        Get state of the road network.
        '''

        lane: dMacroLane = self.network.lane[0]

        state = lane.get_state_vector()

        # only return density and speed;

        return state[0], state[2]      
    
    def get_initial_state(self):
        
        return self.get_state() 
    
    def get_target_state(self):
        
        return self.get_state()

    def tensorize(self, state, requires_grad: bool = True):

        '''
        Turn the given state into a pytorch tensor.
        '''

        init_density: th.Tensor = state[0].detach().clone()
        init_speed: th.Tensor = state[1].detach().clone()

        init_density.requires_grad = requires_grad
        init_speed.requires_grad = requires_grad

        return (init_density, init_speed)

    def init_torch_optimizer(self, state, lr: float) -> th.optim.Adam:

        '''
        Initialize pytorch Adam optimizer for given state.
        '''

        optimizer = th.optim.Adam(state, lr=lr)
        
        return optimizer

    def vectorize(self, state):

        '''
        Turn the given state into a single vector to be 
        used in gradient-free algorithms.
        '''

        density: th.Tensor = state[0]
        speed: th.Tensor = state[1]

        vstate = th.cat([density, speed])

        return vstate.numpy()

    def unvectorize(self, vstate):

        '''
        Inverse function of [vectorize].
        '''

        num_cell = self.num_cell
        dtype = self.dtype

        density = th.tensor(vstate[:num_cell], dtype=dtype)
        speed = th.tensor(vstate[num_cell:], dtype=dtype)

        return density, speed

    
    def bounds(self):

        '''
        Get lower bound and upper bound of the state.
        '''

        speed_limit = self.speed_limit
        num_cell = self.num_cell
        dtype = self.dtype

        density_lb = th.zeros((num_cell,), dtype=dtype)
        density_ub = th.ones((num_cell,), dtype=dtype)

        speed_lb = th.zeros((num_cell,), dtype=dtype)
        speed_ub = th.ones((num_cell,), dtype=dtype) * speed_limit

        lb = (density_lb, speed_lb)
        ub = (density_ub, speed_ub)

        return lb, ub
    
    def compute_beg_error(self, sa, sb):
        
        '''
        Compute error between two initial states (MSE).
        '''

        density_a: th.Tensor = sa[0]
        density_b: th.Tensor = sb[0]

        speed_a: th.Tensor = sa[1]
        speed_b: th.Tensor = sb[1]

        density_error = th.pow(density_a - density_b, 2.0).sum()
        speed_error = th.pow(speed_a - speed_b, 2.0).sum()
        
        return density_error + speed_error
    
    def compute_end_error(self, sa, sb):
        
        '''
        Compute error between two final states (MSE).
        '''

        density_a: th.Tensor = sa[0]
        density_b: th.Tensor = sb[0]

        speed_a: th.Tensor = sa[1]
        speed_b: th.Tensor = sb[1]

        density_error = th.pow(density_a - density_b, 2.0).sum()
        speed_error = th.pow(speed_a - speed_b, 2.0).sum()
        
        return density_error + speed_error

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Script to solve inverse problem in hybrid traffic simulation")
    parser.add_argument("--n_trial", type=int, default=5)
    parser.add_argument("--n_cell", type=int, default=10)
    parser.add_argument("--n_timestep", type=int, default=500)
    parser.add_argument("--cell_length", type=float, default=5.0)
    parser.add_argument("--speed_limit", type=float, default=30.0)
    parser.add_argument("--delta_time", type=float, default=0.01)
    parser.add_argument("--n_episode", type=int, default=100)
    args = parser.parse_args()

    # get environment parameters;
    
    num_trial = args.n_trial
    num_cell = args.n_cell
    num_timestep = args.n_timestep
    num_episode = args.n_episode
    speed_limit = args.speed_limit
    cell_length = args.cell_length
    delta_time = args.delta_time

    # problem solve;

    run_name = "hybrid_{}".format(time.time())
    problem = HybridInverseProblem(num_trial, num_timestep, num_episode, delta_time, speed_limit, run_name, num_cell, cell_length)
    problem.evaluate()