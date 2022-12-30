'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''

import torch as th
import numpy as np

from road.lane._micro_lane import MicroLane
from road.vehicle.micro_vehicle import MicroVehicle
from model.micro.didm import dIDM

from typing import List

class dMicroLane(MicroLane):
    
    '''
    Lane that simulates traffic flow using microscopic IDM model.

    Use analytical gradients for differentiation.
    '''

    class dLane:

        '''
        Store Jacobian matrices of each vehicle in the next time 
        step w.r.t. each vehicle in the previous time step.

        In solving the IDM, the state of a vehicle (i) in time
        step (n + 1) only depends on the states of the vehicles
        (i), (i + 1) in time step (n). Therefore, we can compute 
        and store the partial derivatives of Q(i, n + 1) w.r.t. 
        Q(i, n), and Q(i + 1, n). Each of these derivatives is a 
        2 x 2 Jacobian matrix, as vehicle state value is represented
        with 2 variables (position and speed).

        We store these Jacobian matrices in following layout:

            [ next vehicle id, prev vehicle id, dQ(2, 2) ]

        Next vehicle id = 0 ~ Num. Vehicle
        Prev vehicle id = 0 (that vehicle), 1 (leading vehicle)

        dQ = 
            [ dp(n + 1) / dp(n), dp(n + 1) / ds(n) ]
            [ ds(n + 1) / dp(n), ds(n + 1) / ds(n) ]
        '''

        def __init__(self, num_vehicle):

            # dqs[a, 0] : Gradient of [a] w.r.t. [a + 0]
            # dqs[a, 1] : Gradient of [a] w.r.t. [a + 1]

            self.dqs = np.zeros((num_vehicle, 2, 2, 2), dtype=np.float32)

    def __init__(self, id: int, lane_length: float, speed_limit: float):
        
        super().__init__(id, lane_length, speed_limit)

        self.d_lane: List[dMicroLane.dLane] = []
        self.b_curr_vehicle: List[MicroVehicle] = []

    def forward(self, delta_time: float):


        '''
        Take a single forward simulation step by computing vehicle state values of next time step.

        Note that the new state values are stored in [next_vehicle_position] and [next_vehicle_speed]; 
        call [update_state] to apply them to [curr_vehicle].
        '''

        cp, cs = self.vectorize_input()

        np, ns = dMicroForwardLayer.apply(self, cp, cs, delta_time)

        self.set_next_state_vector(np, ns)

    def _forward(self, delta_time: float):
        
        '''
        Take forward step by calling [MacroLane]'s forward function.
        '''

        super().forward(delta_time)

    def _backward(self, delta_time: float):

        '''
        Fill in [d_lane] with info from [acc_info].
        '''

        d_lane = dMicroLane.dLane(self.num_vehicle())

        for vi, mv in enumerate(self.curr_vehicle):

            position_delta, speed_delta = self.compute_state_delta(vi)

            curr_acc_info = self.acc_info[vi]

            d_lane.dqs[vi, 0] = dIDM.compute_dEgo(mv.accel_max,
                                                        mv.accel_pref,
                                                        mv.speed,
                                                        mv.target_speed,
                                                        position_delta,
                                                        speed_delta,
                                                        mv.min_space,
                                                        mv.time_pref,
                                                        curr_acc_info[1],
                                                        delta_time, 
                                                        curr_acc_info[2],
                                                        curr_acc_info[3])

            d_lane.dqs[vi, 1] = dIDM.compute_dLeading(mv.accel_max,
                                                        mv.accel_pref,
                                                        mv.speed,
                                                        mv.target_speed,
                                                        position_delta,
                                                        speed_delta,
                                                        mv.min_space,
                                                        mv.time_pref,
                                                        curr_acc_info[1],
                                                        delta_time, 
                                                        curr_acc_info[2],
                                                        curr_acc_info[3])

        self.d_lane.append(d_lane)

    
    def vectorize_input(self):
        
        '''
        To use analytical gradients, this class vectorizes states and pass them to computation layer.

        Note that only the variables included in the vectorized states support gradients.

        Currently, we support position and speed of every vehicle, and head pos, speed delta, in the input.
        '''

        cp, cs = self.get_state_vector()

        if len(cp) and len(cs):

            head_position = th.zeros((1,), dtype=th.float32)
            head_position[0] = cp[-1] + self.head_position_delta

            head_speed = th.zeros((1,), dtype=th.float32)
            head_speed[0] = cs[-1] - self.head_speed_delta

            cp = th.cat([cp, head_position])
            cs = th.cat([cs, head_speed])

        return cp, cs

    def backup_vehicle(self):

        '''
        Store current vehicle states in backup vehicle states.
        '''

        self.b_curr_vehicle = []

        for vehicle in self.curr_vehicle:

            self.b_curr_vehicle.append(vehicle)

        self.b_curr_vehicle.append(self.head_position_delta)
        self.b_curr_vehicle.append(self.head_speed_delta)
        

    def detach_vehicle(self):

        '''
        Change states in current vehicles to plain floating point numbers.
        '''

        # backup original states first;

        self.backup_vehicle()

        for i, vehicle in enumerate(self.curr_vehicle):

            self.curr_vehicle[i] = dMicroLane.devehicle(vehicle)

    def clear_gradient(self):

        '''
        Clear [d_lane], which is needed for gradient computation
        '''

        self.d_lane = []

    @staticmethod
    def devehicle(vehicle: MicroVehicle):

        '''
        Change vehicle states in pytorch Tensor to plain floating point number.
        '''

        nv = MicroVehicle(vehicle.id,
                            vehicle.position, 
                            vehicle.speed, 
                            vehicle.accel_max, 
                            vehicle.accel_pref, 
                            vehicle.target_speed, 
                            vehicle.min_space, 
                            vehicle.time_pref, 
                            vehicle.length,
                            vehicle.a)

        if isinstance(nv.position, th.Tensor):

            nv.position = nv.position.item()
        
        if isinstance(nv.speed, th.Tensor):

            nv.speed = nv.speed.item()

        return nv
            

class dMicroForwardLayer(th.autograd.Function):

    @staticmethod
    def forward(ctx, lane: dMicroLane, p: th.Tensor, s: th.Tensor, delta_time: float):
        
        '''
        Note that tensor inputs are just placeholders; we assume they are
        already stored in [lane] for acceleration. Therefore, do not
        use this layer elsewhere than [dMicroLane.forward]!
        '''

        '''
        For acceleration, we do not use pytorch Tensor in computation.
        Therefore, before computation, we backup original vehicle states
        and restore them after every computation is completed.
        '''

        lane.detach_vehicle()
        
        # take forward step;
        
        lane._forward(delta_time)

        # compute gradient info for differentiation;

        lane._backward(delta_time)

        # recover original states;

        lane.curr_vehicle = lane.b_curr_vehicle[:-2]
        lane.head_position_delta = lane.b_curr_vehicle[-2]
        lane.head_speed_delta = lane.b_curr_vehicle[-1]

        # set context;

        ctx.dqs = lane.d_lane[-1].dqs

        # return state vector r and y, and fluxes at boundary;

        np, ns = lane.get_next_state_vector()
    
        return np, ns

    @staticmethod
    def backward(ctx, grad_np: th.Tensor, grad_ns: th.Tensor):

        dqs: np.ndarray = ctx.dqs
        dqs = dqs.transpose((0, 1, 3, 2))

        num_vehicle = len(grad_np)
        
        # compute gradients to propagate;

        grad_nvehicle = np.zeros((num_vehicle, 1, 2, 1), dtype=np.float32)
        grad_nvehicle[:, 0, 0, 0] = grad_np
        grad_nvehicle[:, 0, 1, 0] = grad_ns

        grad_vehicle = np.matmul(dqs, grad_nvehicle)        # transpose needed;
        grad_vehicle = np.squeeze(grad_vehicle, -1)

        grad_ps = np.zeros((num_vehicle + 1, 2), dtype=np.float32)
        grad_ps[:-1] = grad_vehicle[:, 0, :]
        grad_ps[1:] += grad_vehicle[:, 1, :]

        grad_p = grad_ps[:, 0]
        grad_s = grad_ps[:, 1]
        
        grad_p = th.tensor(grad_p)
        grad_s = th.tensor(grad_s)

        return None, grad_p, grad_s, None