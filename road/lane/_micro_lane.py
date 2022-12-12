'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''
import numpy as np
import torch as th

from road.lane._base_lane import BaseLane
from road.vehicle.micro_vehicle import MicroVehicle
from model.micro._idm import IDM

from typing import List, Union

DEFAULT_HEAD_POSITION_DELTA = 1e3
DEFAULT_HEAD_SPEED_DELTA = 0

class MicroLane(BaseLane):

    '''
    Lane that simulates traffic flow using microscopic IDM model.

    Use automatic differentiation for differentiation.
    '''
    
    def __init__(self, id: int, lane_length: float, speed_limit: float):
    
        super().__init__(id, lane_length, speed_limit)

        # list of vehicles;
        # here we assume i-th vehicle is right behind (i + 1)-th vehicle;
        
        self.curr_vehicle: List[MicroVehicle] = []

        # position and speed that would be used to update vehicle states;

        self.acc_info = []
        self.next_vehicle_position: List[Union[float, th.Tensor]] = []
        self.next_vehicle_speed: List[Union[float, th.Tensor]] = []

        # value to use for the head vehicle, which does not have leading vehicle;

        self.head_position_delta = DEFAULT_HEAD_POSITION_DELTA
        self.head_speed_delta = DEFAULT_HEAD_SPEED_DELTA

        # brdy callback;

        self.bdry_callback = None
        self.bdry_callback_args = {'lane': self}

    def is_macro(self):
        return False

    def is_micro(self):
        return True

    def add_head_vehicle(self, vehicle: MicroVehicle):

        self.curr_vehicle.append(vehicle)

    def add_tail_vehicle(self, vehicle: MicroVehicle):

        self.curr_vehicle.insert(0, vehicle)

    def num_vehicle(self):

        return len(self.curr_vehicle)
            
    def forward(self, delta_time: float):

        '''
        Take a single forward simulation step by computing vehicle state values of next time step.

        Note that the new state values are stored in [next_vehicle_position] and [next_vehicle_speed]; 
        call [update_state] to apply them to [curr_vehicle].
        '''      

        self.next_vehicle_position.clear()
        self.next_vehicle_speed.clear()

        self.acc_info.clear()

        for vi, mv in enumerate(self.curr_vehicle):

            # compute next position and speed using Eulerian method;

            position_delta, speed_delta = self.compute_state_delta(vi)

            assert position_delta >= 0, "Vehicle collision detected"

            acc_info = IDM.compute_acceleration(mv.accel_max,
                                                    mv.accel_pref,
                                                    mv.speed,
                                                    mv.target_speed,
                                                    position_delta,
                                                    speed_delta,
                                                    mv.min_space,
                                                    mv.time_pref,
                                                    delta_time)

            self.acc_info.append(acc_info)

            acc = acc_info[0]

            next_position = mv.position + delta_time * mv.speed
            next_speed = mv.speed + delta_time * acc

            self.next_vehicle_position.append(next_position)
            self.next_vehicle_speed.append(next_speed)


    def compute_state_delta(self, id):
        
        '''
        Compute position and speed delta to leading vehicle.
        '''

        if id == len(self.curr_vehicle) - 1:

            position_delta = self.head_position_delta
            speed_delta = self.head_speed_delta

        else:

            mv = self.curr_vehicle[id]
            lv = self.curr_vehicle[id + 1]          # leading vehicle;
            
            position_delta = abs(lv.position - mv.position) - ((lv.length + mv.length) * 0.5)
            speed_delta = mv.speed - lv.speed

        return position_delta, speed_delta

    def update_state(self):
        
        '''
        Update current vehicle state with next state.
        '''

        for i, mv in enumerate(self.curr_vehicle):

            mv.position = self.next_vehicle_position[i]
            mv.speed = self.next_vehicle_speed[i]        

    def set_state_vector(self, position: th.Tensor, speed: th.Tensor):

        '''
        Set vehicle state from given vector, of which length equals to number of vehicles.
        '''

        assert len(position) == self.num_vehicle(), "Vehicle number mismatch"
        assert len(speed) == self.num_vehicle(), "Vehicle number mismatch"

        for i in range(self.num_vehicle()):

            self.curr_vehicle[i].position = position[i]
            self.curr_vehicle[i].speed = speed[i]

    def get_state_vector(self):

        '''
        Get state vector in the order of position and speed.
        '''

        position = th.zeros((self.num_vehicle(),))
        speed = th.zeros((self.num_vehicle(),))

        for i in range(self.num_vehicle()):
            position[i] = self.curr_vehicle[i].position
            speed[i] = self.curr_vehicle[i].speed

        return position, speed


    def set_next_state_vector(self, position: th.Tensor, speed: th.Tensor):

        '''
        Set next vehicle state from given vector, of which length equals to number of vehicles.
        '''

        assert len(position) == self.num_vehicle(), "Vehicle number mismatch"
        assert len(speed) == self.num_vehicle(), "Vehicle number mismatch"

        self.next_vehicle_position.clear()
        self.next_vehicle_speed.clear()

        for i in range(self.num_vehicle()):

            self.next_vehicle_position.append(position[i])
            self.next_vehicle_speed.append(speed[i])

    def get_next_state_vector(self):

        '''
        Get next state vector in the order of position and speed.
        '''

        position = th.zeros((self.num_vehicle(),))
        speed = th.zeros((self.num_vehicle(),))

        for i in range(self.num_vehicle()):
            position[i] = self.next_vehicle_position[i]
            speed[i] = self.next_vehicle_speed[i]

        return position, speed

    def entering_free_space(self):

        '''
        Get free space at the beginning of the lane;
        '''

        if self.num_vehicle():

            return self.curr_vehicle[0].position - 0.5 * (self.curr_vehicle[0].length)

        else:

            return self.length

    def random_vehicle(self):
        
        '''
        Generate a vehicle with randomly chosen attributes, which depend on this lane's speed limit.
        '''

        speed_limit = self.speed_limit

        # vehicle length;
        # @TODO: apply dynamic sizes;

        vehicle_length = 5.0

        # maximum acceleration;

        a_max = np.random.rand()
        a_max = np.interp(a_max, [0, 1], [speed_limit * 1.5, speed_limit * 2.0])

        # preferred acceleration;
        
        a_pref = np.random.rand()
        a_pref = np.interp(a_pref, [0, 1], [speed_limit * 1.0, speed_limit * 1.5])

        # target speed;
        
        target_speed = np.random.rand()
        target_speed = np.interp(target_speed, [0, 1], [speed_limit * 0.8, speed_limit * 1.2])
        
        # min space ahead;
        
        min_space = np.random.rand()
        min_space = np.interp(min_space, [0, 1], [vehicle_length * 0.2, vehicle_length * 0.4])

        # preferred time to go;
        
        time_pref = np.random.rand()
        time_pref = np.interp(time_pref, [0, 1], [0.2, 0.6])

        return MicroVehicle(0, 0, a_max, a_pref, target_speed, min_space, time_pref, vehicle_length, vehicle_length)

    def occupied_length(self):

        '''
        Compute length of occupied regions by vehicles.
        '''

        ol = 0

        for mv in self.curr_vehicle:

            ol += mv.length

        # clip length to this lane's length;

        ol = min(ol, self.length)

        return ol


    def avg_density(self):

        '''
        Compute average density of this lane by dividing occupied lane length by entire length.
        '''

        return self.occupied_length() / self.length

    def avg_speed(self):

        '''
        Compute average speed of every vehicle. If there is no vehicle, return speed limit.
        '''

        avg = 0

        for mv in self.curr_vehicle:

            avg += mv.speed

        avg = avg / max(len(self.curr_vehicle), 1)

        return avg

    def clear(self):

        '''
        Clear every vehicle and next state info.
        '''

        self.curr_vehicle.clear()

        self.next_vehicle_position = []
        self.next_vehicle_speed = []