'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com

Script to sanity check our analytical gradients by comparison with autodiff (Micro).
'''

import argparse
import torch as th
from tqdm import tqdm

from road.vehicle.micro_vehicle import MicroVehicle
from road.lane._micro_lane import MicroLane
from road.lane.dmicro_lane import dMicroLane


if __name__ == "__main__":

    dtype = th.float32
    device = th.device("cpu")

    parser = argparse.ArgumentParser("Script to sanity check analytical gradients of microscopic traffic simulation")
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--n_vehicle", type=int, default=10)
    parser.add_argument("--n_step", type=int, default=1000)
    parser.add_argument("--vehicle_length", type=float, default=5.0)
    parser.add_argument("--speed_limit", type=float, default=30.0)
    parser.add_argument("--delta_time", type=float, default=0.03)
    args = parser.parse_args()

    # get environment parameters;
    
    num_test = args.n_test
    num_vehicle = args.n_vehicle
    num_step = args.n_step

    lane_length = 1e10      # arbitrary large number

    speed_limit = th.tensor([args.speed_limit], dtype=dtype)
    vehicle_length = th.tensor([args.vehicle_length], dtype=dtype)
    delta_time = args.delta_time


    max_p_error = -1
    max_s_error = -1
    avg_p_error = 0
    avg_s_error = 0

    pbar = tqdm(range(args.n_test))
    for test in pbar: 

        # initialize vehicle parameters randomly;

        # max acceleration

        a_max = th.rand((num_vehicle), dtype=dtype, device=device)
        a_max = th.lerp(speed_limit * 1.5, speed_limit * 2.0, a_max)

        # preferred acceleration
        
        a_pref = th.rand((num_vehicle), dtype=dtype, device=device)
        a_pref = th.lerp(speed_limit * 1.0, speed_limit * 1.5, a_pref)

        # target speed
        
        target_speed = th.rand((num_vehicle), dtype=dtype, device=device)
        target_speed = th.lerp(speed_limit * 0.8, speed_limit * 1.2, target_speed)

        # min space ahead
        
        min_space = th.rand((num_vehicle), dtype=dtype, device=device)
        min_space = th.lerp(vehicle_length * 0.5, vehicle_length * 1.0, min_space)

        # preferred time to go
        
        time_pref = th.rand((num_vehicle), dtype=dtype, device=device)
        time_pref = th.lerp(th.ones_like(time_pref) * 1.0, th.ones_like(time_pref) * 3.0, time_pref)

        # initialize position and speed of each vehicle randomly;
        
        position_start = th.arange(0, num_vehicle + 1) * 3.0 * vehicle_length
        autodiff_init_position = position_start + th.rand((num_vehicle + 1,), dtype=dtype, device=device) * vehicle_length
        autodiff_init_position[-1] = th.rand((1,)) * 1e3        # position delta
        
        autodiff_init_speed = th.rand((num_vehicle + 1,), dtype=dtype, device=device)
        autodiff_init_speed = th.lerp(0.05 * speed_limit, 0.15 * speed_limit, autodiff_init_speed)
        autodiff_init_speed[-1] = th.rand((1,)) * speed_limit   # speed delta

        autodiff_init_position.requires_grad = True
        autodiff_init_speed.requires_grad = True

        analytic_init_position = autodiff_init_position.detach().clone()
        analytic_init_speed = autodiff_init_speed.detach().clone()

        analytic_init_position.requires_grad = True
        analytic_init_speed.requires_grad = True

        # create micro lane;

        autodiff_lane = MicroLane(lane_length, speed_limit)
        analytic_lane = dMicroLane(lane_length, speed_limit)

        # initialize lane;

        for i in range(num_vehicle):
            
            mv = MicroVehicle(0, 0, a_max[i], a_pref[i], target_speed[i], min_space[i], time_pref[i], vehicle_length)
            autodiff_lane.add_vehicle(mv)

        for i in range(num_vehicle):

            mv = MicroVehicle(0, 0, a_max[i], a_pref[i], target_speed[i], min_space[i], time_pref[i], vehicle_length)
            analytic_lane.add_vehicle(mv)

        autodiff_lane.set_state_vector(autodiff_init_position[:-1], autodiff_init_speed[:-1])
        analytic_lane.set_state_vector(analytic_init_position[:-1], analytic_init_speed[:-1])

        autodiff_lane.head_position_delta = autodiff_init_position[-1]
        autodiff_lane.head_speed_delta = autodiff_init_speed[-1]

        analytic_lane.head_position_delta = analytic_init_position[-1]
        analytic_lane.head_speed_delta = analytic_init_speed[-1]

        # simulate [num_step] steps;

        autodiff_loss = 0
        analytic_loss = 0

        for step in range(num_step):

            autodiff_lane.forward(delta_time)
            autodiff_lane.update_state()

            autodiff_position, autodiff_speed = autodiff_lane.get_state_vector()
            autodiff_loss += autodiff_position.sum() + autodiff_speed.sum()

        for step in range(num_step):

            analytic_lane.forward(delta_time)
            analytic_lane.update_state()

            analytic_position, analytic_speed = analytic_lane.get_state_vector()
            analytic_loss += analytic_position.sum() + analytic_speed.sum()

        # compute gradient from loss;

        autodiff_loss.backward()
        analytic_loss.backward()

        # only consider gradients that are big enough, as there could be numerical instability;

        NUMERIC_EPS = 1e-5
        valid_position_ind = th.where(th.abs(autodiff_init_position.grad) > NUMERIC_EPS)
        valid_speed_ind = th.where(th.abs(autodiff_init_speed.grad) > NUMERIC_EPS)

        p_error = th.max((th.abs(autodiff_init_position.grad - analytic_init_position.grad) \
                            / th.clamp(th.abs(autodiff_init_position.grad), min=NUMERIC_EPS))[valid_position_ind])
        s_error = th.max((th.abs(autodiff_init_speed.grad - analytic_init_speed.grad) \
                            / th.clamp(th.abs(autodiff_init_speed.grad), min=NUMERIC_EPS))[valid_speed_ind])

        max_p_error = max(p_error, max_p_error)
        max_s_error = max(s_error, max_s_error)
        avg_p_error += p_error
        avg_s_error += s_error
            
        pbar.set_description("Backward Diff : Position   = {:.2f} %  / Speed = {:.2f} % ".format(p_error * 1e2, s_error * 1e2))

    
    avg_p_error = avg_p_error / num_test
    avg_s_error = avg_s_error / num_test

    print("Max Grad. Diff. : Position = {:.2f} % / Speed = {:.2f} %".format(max_p_error * 1e2, max_s_error * 1e2))
    print("Avg Grad. Diff. : Position = {:.2f} % / Speed = {:.2f} %".format(avg_p_error * 1e2, avg_s_error * 1e2))