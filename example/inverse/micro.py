'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com

Script to solve inverse problem in microscopic traffic simulation.
'''

import argparse
import time
import torch as th

from example.inverse._inverse import InverseProblem
from road.network.road_network import RoadNetwork
from road.lane.dmicro_lane import dMicroLane

class MicroInverseProblem(InverseProblem):

    def __init__(self, 
                num_trial: int, 
                num_timestep: int,
                num_episode: int, 
                delta_time: float, 
                speed_limit: float,
                run_name: str, 
                num_vehicle: int,
                vehicle_length: float):

        super().__init__(num_trial, num_timestep, num_episode, delta_time, speed_limit, run_name)

        self.num_vehicle = num_vehicle
        self.vehicle_length = vehicle_length
        self.dtype = th.float32

    def init_network(self):

        '''
        Randomly initialize road network to simulate.
        '''

        lane_length = 1e10  # arbitrary large value;

        num_vehicle = self.num_vehicle
        speed_limit = self.speed_limit

        # initialize position and speed of each vehicle randomly;
        
        init_position, init_speed = self.random_state()

        # create micro lane;

        lane = dMicroLane(0, lane_length, speed_limit)

        # initialize lane;

        for _ in range(num_vehicle):

            mv = lane.random_vehicle()
            lane.add_tail_vehicle(mv)

        lane.set_state_vector(init_position, init_speed)

        # create network;

        self.network = RoadNetwork(speed_limit)
        self.network.add_lane(lane.id, lane)


    def random_state(self):

        '''
        Get random state of the road network.
        '''

        num_vehicle = self.num_vehicle
        vehicle_length = self.vehicle_length
        dtype = self.dtype

        speed_limit = th.tensor([self.speed_limit], dtype=dtype)

        position_start = th.arange(0, num_vehicle) * 4.0 * vehicle_length
        init_position = position_start + th.rand((num_vehicle,), dtype=dtype) * 2.0 * vehicle_length
        
        init_speed = th.rand((num_vehicle,), dtype=dtype)
        init_speed = th.lerp(0.3 * speed_limit, 0.7 * speed_limit, init_speed)

        return (init_position, init_speed)

    def set_state(self, state):

        '''
        Set state of the road network.
        '''

        lane: dMicroLane = self.network.lane[0]
        lane.set_state_vector(state[0], state[1])
    
    def get_state(self):

        '''
        Get state of the road network.
        '''

        lane: dMicroLane = self.network.lane[0]

        return lane.get_state_vector()

    def tensorize(self, state, requires_grad: bool = True):

        '''
        Turn the given state into a pytorch tensor.
        '''

        init_position: th.Tensor = state[0].detach().clone()
        init_speed: th.Tensor = state[1].detach().clone()

        init_position.requires_grad = requires_grad
        init_speed.requires_grad = requires_grad

        return (init_position, init_speed)

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

        position: th.Tensor = state[0]
        speed: th.Tensor = state[1]

        vstate = th.cat([position, speed])

        return vstate.numpy()

    def unvectorize(self, vstate):

        '''
        Inverse function of [vectorize].
        '''

        num_vehicle = self.num_vehicle
        dtype = self.dtype

        position = th.tensor(vstate[:num_vehicle], dtype=dtype)
        speed = th.tensor(vstate[num_vehicle:], dtype=dtype)

        return position, speed

    
    def bounds(self):

        '''
        Get lower bound and upper bound of the state.
        '''

        speed_limit = self.speed_limit
        num_vehicle = self.num_vehicle
        vehicle_length = self.vehicle_length
        dtype = self.dtype

        position_lb = th.arange(0, num_vehicle) * 4.0 * vehicle_length
        position_ub = position_lb + th.ones((num_vehicle,), dtype=dtype) * 2.0 * vehicle_length

        speed_lb = th.ones((num_vehicle), dtype=dtype) * 0
        speed_ub = th.ones((num_vehicle), dtype=dtype) * speed_limit

        lb = (position_lb, speed_lb)
        ub = (position_ub, speed_ub)

        return lb, ub

    def compute_error(self, sa, sb):

        '''
        Compute error between two states (MSE).
        '''

        position_a: th.Tensor = sa[0]
        position_b: th.Tensor = sb[0]

        speed_a: th.Tensor = sa[1]
        speed_b: th.Tensor = sb[1]

        position_error = th.pow(position_a - position_b, 2.0).sum()
        speed_error = th.pow(speed_a - speed_b, 2.0).sum()
        
        return position_error + speed_error

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Script to solve inverse problem microscopic traffic simulation")
    parser.add_argument("--n_trial", type=int, default=5)
    parser.add_argument("--n_vehicle", type=int, default=100)
    parser.add_argument("--n_timestep", type=int, default=1000)
    parser.add_argument("--vehicle_length", type=float, default=5.0)
    parser.add_argument("--speed_limit", type=float, default=30.0)
    parser.add_argument("--delta_time", type=float, default=0.03)
    parser.add_argument("--n_episode", type=int, default=100)
    args = parser.parse_args()

    # get environment parameters;
    
    num_trial = args.n_trial
    num_vehicle = args.n_vehicle
    num_timestep = args.n_timestep
    num_episode = args.n_episode
    speed_limit = args.speed_limit
    vehicle_length = args.vehicle_length
    delta_time = args.delta_time

    # problem solve;

    run_name = "micro_{}".format(time.time())
    problem = MicroInverseProblem(num_trial, num_timestep, num_episode, delta_time, speed_limit, run_name, num_vehicle, vehicle_length)
    problem.evaluate()