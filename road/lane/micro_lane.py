'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''
import torch as th

from road.lane.base_lane import BaseLane
from road.lane.macro_lane import MacroLane
from model.micro.didm import dIDMLayer

class MicroLane(BaseLane):
    '''
    [MacroLane] runs microscopic traffic simulation for a single lane using IDM model. 
    Here we assume i-th vehicle is right behind (i + 1)-th vehicle.
    '''            
    def __init__(self, lane_length: float, speed_limit: float):
        super().__init__(lane_length, speed_limit)

        # initialize vehicle state
        self.curr_pos: th.Tensor = th.zeros((0, 1), dtype=th.float32)
        self.curr_vel: th.Tensor = th.zeros((0, 1), dtype=th.float32)
        self.accel_max: th.Tensor = th.zeros((0, 1), dtype=th.float32)
        self.accel_pref: th.Tensor = th.zeros((0, 1), dtype=th.float32)
        self.target_vel: th.Tensor = th.zeros((0, 1), dtype=th.float32)
        self.min_space: th.Tensor = th.zeros((0, 1), dtype=th.float32)
        self.time_pref: th.Tensor = th.zeros((0, 1), dtype=th.float32)
        self.vehicle_length: th.Tensor = th.zeros((0, 1), dtype=th.float32)

        # Initialize next cells for storing next cell values.
        self.next_pos: th.Tensor = th.zeros((0, 1), dtype=th.float32)
        self.next_vel: th.Tensor = th.zeros((0, 1), dtype=th.float32)

        # ancilliary variable to allow gradient flow in hybrid simulation
        self.vehicle_a: th.Tensor = th.zeros((0, 1), dtype=th.float32)

    def is_macro(self):
        return False

    def is_micro(self):
        return True

    def to(self, device, dtype):
        self.curr_pos = self.curr_pos.to(device=device, dtype=dtype)
        self.curr_vel = self.curr_vel.to(device=device, dtype=dtype)
        self.accel_max = self.accel_max.to(device=device, dtype=dtype)
        self.accel_pref = self.accel_pref.to(device=device, dtype=dtype)
        self.target_vel = self.target_vel.to(device=device, dtype=dtype)
        self.min_space = self.min_space.to(device=device, dtype=dtype)
        self.time_pref = self.time_pref.to(device=device, dtype=dtype)
        self.vehicle_length = self.vehicle_length.to(device=device, dtype=dtype)
        
        self.next_pos = self.next_pos.to(device=device, dtype=dtype)
        self.next_vel = self.next_vel.to(device=device, dtype=dtype)

    def add_vehicle(self, 
                    pos: th.Tensor, 
                    vel: th.Tensor, 
                    accel_max: th.Tensor,
                    accel_pref: th.Tensor,
                    target_vel: th.Tensor,
                    min_space: th.Tensor,
                    time_pref: th.Tensor,
                    vehicle_length: th.Tensor,
                    vehicle_a: th.Tensor):

        assert pos.ndim == 2, ""

        self.curr_pos = th.cat((pos, self.curr_pos), dim=0)
        self.curr_vel = th.cat((vel, self.curr_vel), dim=0)
        self.next_pos = th.cat((pos, self.next_pos), dim=0)
        self.next_vel = th.cat((vel, self.next_vel), dim=0)
        self.accel_max = th.cat((accel_max, self.accel_max), dim=0)
        self.accel_pref = th.cat((accel_pref, self.accel_pref), dim=0)
        self.target_vel = th.cat((target_vel, self.target_vel), dim=0)
        self.min_space = th.cat((min_space, self.min_space), dim=0)
        self.time_pref = th.cat((time_pref, self.time_pref), dim=0)
        self.vehicle_length = th.cat((vehicle_length, self.vehicle_length), dim=0)
        self.vehicle_a = th.cat((vehicle_a, self.vehicle_a), dim=0)

    def remove_head_vehicle(self):
        self.curr_pos = self.curr_pos[:-1, :]
        self.curr_vel = self.curr_vel[:-1, :]
        self.next_pos = self.next_pos[:-1, :]
        self.next_vel = self.next_vel[:-1, :]
        self.accel_max = self.accel_max[:-1, :]
        self.accel_pref = self.accel_pref[:-1, :]
        self.target_vel = self.target_vel[:-1, :]
        self.min_space = self.min_space[:-1, :]
        self.time_pref = self.time_pref[:-1, :]
        self.vehicle_length = self.vehicle_length[:-1, :]
        self.vehicle_a = self.vehicle_a[:-1, :]

    def rand_vehicle(self):
        dtype = th.float32
        device = th.device("cpu")

        speed_limit = th.ones((1, 1), dtype=dtype, device=device) * self.speed_limit

        # position and velocity: randomly generate
        pos = th.rand((1, 1), dtype=dtype, device=device) * self.length
        vel = th.rand((1, 1), dtype=dtype, device=device) * self.speed_limit

        # max acceleration
        a_max = th.rand((1, 1), dtype=dtype, device=device)
        a_max = th.lerp(speed_limit * 1.5, speed_limit * 2.0, a_max)

        # preferred acceleration
        a_pref = th.rand((1, 1), dtype=dtype, device=device)
        a_pref = th.lerp(speed_limit * 1.0, speed_limit * 1.5, a_pref)

        # target velocity
        v_target = th.rand((1, 1), dtype=dtype, device=device)
        v_target = th.lerp(speed_limit * 0.8, speed_limit * 1.2, v_target)

        # length
        # vehicle_length = th.rand((1, 1), dtype=dtype, device=device)
        # vehicle_length = th.lerp(4.0, 6.0, vehicle_length)
        vehicle_length = th.ones((1, 1), dtype=dtype, device=device) * 5.0

        # min space ahead
        min_space = th.rand((1, 1), dtype=dtype, device=device)
        min_space = th.lerp(vehicle_length * 0.1, vehicle_length * 0.3, min_space)

        # preferred time to go
        time_pref = th.rand((1, 1), dtype=dtype, device=device)
        time_pref = th.lerp(th.ones_like(time_pref) * 0.2, th.ones_like(time_pref) * 0.4, time_pref)

        return pos, vel, a_max, a_pref, v_target, min_space, time_pref, vehicle_length

    def preprocess(self, 
                    last_pos_delta: th.Tensor, 
                    last_vel_delta: th.Tensor, 
                    delta_time: float):
        # For given vehicle information, generate a set of tensors that can be fed into IDMLayer.
        num_vehicle = self.num_vehicle()

        # pos_delta: assume position[i + 1] > position[i]
        pos_delta = th.zeros_like(self.curr_pos)
        pos_delta[:num_vehicle-1] = self.curr_pos[1:] - self.curr_pos[:num_vehicle-1] - \
                                        ((self.vehicle_length[1:] + self.vehicle_length[:num_vehicle-1]) * 0.5)
        pos_delta[[num_vehicle-1], :] = last_pos_delta

        assert not th.any(pos_delta < 0), "Vehicle overlap"

        # vel_delta
        vel_delta = th.zeros_like(self.curr_vel)
        vel_delta[:num_vehicle-1] = self.curr_vel[:num_vehicle - 1] - self.curr_vel[1:]
        vel_delta[[num_vehicle-1], :] = last_vel_delta
        
        # delta time
        delta_time = th.ones_like(self.curr_pos) * delta_time
        
        return pos_delta, vel_delta, delta_time

    def num_vehicle(self):
        return self.curr_pos.shape[0]
            
    def forward_step(self, 
                    delta_time: float,
                    last_pos_delta: th.Tensor = th.tensor([1e+3], dtype=th.float32),
                    last_vel_delta: th.Tensor = th.tensor([0.0], dtype=th.float32),
                    parallel: bool = True):
        '''
        take a single forward simulation step by computing acceleration using IDM model
        '''

        if self.num_vehicle() == 0:
            return

        last_pos_delta += self.length - self.curr_pos[[-1], :] - self.vehicle_length[[-1], :] * 0.5

        # do not care about prev lanes
        if self.next_lane is not None:
            my_last_pos = self.curr_pos[[-1], :]
            my_last_vel = self.curr_vel[[-1], :]
            my_last_len = self.vehicle_length[[-1], :]

            if self.next_lane.is_macro():
                n_lane: MacroLane = self.next_lane
                
                density = th.clamp(n_lane.curr_rho[0, :], 0.0, 1.0)
                speed = n_lane.curr_u[[0], :]

                if self.num_vehicle() > 0:
                    last_pos_delta = (self.length - my_last_pos - (my_last_len * 0.5)) + n_lane.cell_length * (1.0 - density)
                    last_vel_delta = self.curr_vel[[-1], :] - speed
            else:
                n_lane: MicroLane = self.next_lane
                if self.num_vehicle() > 0 and n_lane.num_vehicle() > 0:
                    next_first_pos = n_lane.curr_pos[[0], :]
                    next_first_vel = n_lane.curr_vel[[0], :]
                    next_first_len = n_lane.vehicle_length[[0], :]

                    last_pos_delta = (self.length - my_last_pos - (my_last_len * 0.5)) + \
                                        (next_first_pos - (next_first_len * 0.5))
                    last_vel_delta = my_last_vel - next_first_vel

        pos_delta, vel_delta, delta_time = self.preprocess(last_pos_delta, last_vel_delta, delta_time)
        
        # take a single step
        if parallel:
            acc = dIDMLayer.apply(self.accel_max, 
                                    self.accel_pref, 
                                    self.curr_vel, 
                                    self.target_vel, 
                                    pos_delta, 
                                    vel_delta, 
                                    self.min_space, 
                                    self.time_pref, 
                                    delta_time)

            self.next_pos = self.curr_pos + self.curr_vel * delta_time
            self.next_vel = self.curr_vel + acc * delta_time
        else:
            for i in range(self.num_vehicle()):
                acc = dIDMLayer.apply(self.accel_max[[i], :], 
                                        self.accel_pref[[i], :], 
                                        self.curr_vel[[i], :], 
                                        self.target_vel[[i], :], 
                                        pos_delta[[i], :], 
                                        vel_delta[[i], :], 
                                        self.min_space[[i], :], 
                                        self.time_pref[[i], :], 
                                        delta_time[[i], :])

                self.next_pos[[i], :] = self.curr_pos[[i], :] + self.curr_vel[[i], :] * delta_time[[i], :]
                self.next_vel[[i], :] = self.curr_vel[[i], :] + acc * delta_time[[i], :]

    def update_state(self):
        '''
        update current state with next state
        '''
        self.curr_pos = self.next_pos
        self.curr_vel = self.next_vel

    def print(self):
        print("Micro Lane: # vehicle = {} / lane length = {:.2f} m / speed_limit = {:.2f} m/sec".format(
            self.num_vehicle(),
            self.length,
            self.speed_limit,
        ))

        print("pos: {}".format(list(self.curr_pos.detach().cpu().numpy()[:, 0])))
        print("vel: {}".format(list(self.curr_vel.detach().cpu().numpy()[:, 0])))