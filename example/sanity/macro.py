'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com

Script to sanity check our analytical gradients by comparison with autodiff (Macro).
'''

import argparse
import torch as th
from tqdm import tqdm

from road.lane._macro_lane import MacroLane
from road.lane.dmacro_lane import dMacroLane

if __name__ == "__main__":

    dtype = th.float32
    device = th.device("cpu")

    parser = argparse.ArgumentParser("Script to sanity check analytical gradients of macroscopic traffic simulation")
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--n_cell", type=int, default=100)
    parser.add_argument("--n_step", type=int, default=10)
    parser.add_argument("--cell_length", type=float, default=100.0)
    parser.add_argument("--speed_limit", type=float, default=30.0)
    parser.add_argument("--delta_time", type=float, default=0.03)
    args = parser.parse_args()

    # get environment parameters;
    
    num_test = args.n_test
    num_cell = args.n_cell
    num_step = args.n_step

    cell_length = args.cell_length
    lane_length = num_cell * cell_length
    speed_limit = args.speed_limit
    delta_time = args.delta_time

    max_r_error = -1
    max_u_error = -1
    avg_r_error = 0
    avg_u_error = 0

    pbar = tqdm(range(args.n_test))
    for test in pbar: 

        # initialize density (rho) and speed (u) of each cell randomly;
        # includes 2 boundary conditions at left and right;
        
        autodiff_init_density = th.rand((num_cell + 2,), dtype=dtype, device=device)
        autodiff_init_speed = th.rand((num_cell + 2,), dtype=dtype, device=device)
        autodiff_init_speed = th.lerp(0.4 * th.tensor([speed_limit], dtype=dtype), 
                                        0.7 * th.tensor([speed_limit], dtype=dtype), 
                                        autodiff_init_speed)

        autodiff_init_density.requires_grad = True
        autodiff_init_speed.requires_grad = True

        analytic_init_density = autodiff_init_density.detach().clone()
        analytic_init_speed = autodiff_init_speed.detach().clone()

        analytic_init_density.requires_grad = True
        analytic_init_speed.requires_grad = True

        # create macro lane;

        autodiff_lane = MacroLane(lane_length, speed_limit, cell_length)
        analytic_lane = dMacroLane(lane_length, speed_limit, cell_length)

        # initialize lane;

        autodiff_lane.set_state_vector_u(autodiff_init_density[1:-1], autodiff_init_speed[1:-1])
        autodiff_lane.set_leftmost_cell(autodiff_init_density[0], autodiff_init_speed[0])
        autodiff_lane.set_rightmost_cell(autodiff_init_density[-1], autodiff_init_speed[-1])

        analytic_lane.set_state_vector_u(analytic_init_density[1:-1], analytic_init_speed[1:-1])
        analytic_lane.set_leftmost_cell(analytic_init_density[0], analytic_init_speed[0])
        analytic_lane.set_rightmost_cell(analytic_init_density[-1], analytic_init_speed[-1])

        # simulate [num_step] steps;

        autodiff_loss = 0
        analytic_loss = 0

        for step in range(num_step):

            autodiff_lane.forward(delta_time)
            autodiff_lane.update_state()

            autodiff_density, autodiff_relflow, autodiff_speed = autodiff_lane.get_state_vector()
            autodiff_loss += autodiff_density.sum() + autodiff_relflow.sum() + autodiff_speed.sum()

        for step in range(num_step):

            analytic_lane.forward(delta_time)
            analytic_lane.update_state()

            analytic_density, analytic_relflow, analytic_speed = analytic_lane.get_state_vector()
            analytic_loss += analytic_density.sum() + analytic_relflow.sum() + analytic_speed.sum()

        # compute gradient from loss;

        autodiff_loss.backward()
        analytic_loss.backward()

        # only consider gradients that are big enough, as there could be numerical instability;

        NUMERIC_EPS = 1e-5
        valid_density_ind = th.where(th.abs(autodiff_init_density.grad) > NUMERIC_EPS)
        valid_speed_ind = th.where(th.abs(autodiff_init_speed.grad) > NUMERIC_EPS)

        r_error = th.max((th.abs(autodiff_init_density.grad - analytic_init_density.grad) \
                            / th.clamp(th.abs(autodiff_init_density.grad), min=NUMERIC_EPS))[valid_density_ind])
        u_error = th.max((th.abs(autodiff_init_speed.grad - analytic_init_speed.grad) \
                            / th.clamp(th.abs(autodiff_init_speed.grad), min=NUMERIC_EPS))[valid_speed_ind])

        max_r_error = max(r_error, max_r_error)
        max_u_error = max(u_error, max_u_error)
        avg_r_error += r_error
        avg_u_error += u_error
            
        pbar.set_description("Backward Diff : Density = {:.2f} %  / Speed = {:.2f} % ".format(r_error * 1e2, u_error * 1e2))

    avg_r_error = avg_r_error / num_test
    avg_u_error = avg_u_error / num_test

    print("Max Grad. Diff. : Density = {:.2f} % / Speed = {:.2f} %".format(max_r_error * 1e2, max_u_error * 1e2))
    print("Avg Grad. Diff. : Density = {:.2f} % / Speed = {:.2f} %".format(avg_r_error * 1e2, avg_u_error * 1e2))