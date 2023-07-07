from time import time
from example.control.itscp._env import ItscpEnv
from example.control.trainer import Trainer
from example.control.itscp.problem import problem_1, problem_2, problem_3

import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Script to solve intersection signal control problem")
    parser.add_argument("--mode", type=str, choices=['macro', 'micro', 'hybrid'], default='macro')
    parser.add_argument("--problem", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--n_trial", type=int, default=5)
    parser.add_argument("--n_intersection", type=int, default=1)
    parser.add_argument("--n_lane", type=int, default=3)
    parser.add_argument("--lane_length", type=float, default=20.)
    parser.add_argument("--speed_limit", type=float, default=60.)
    parser.add_argument("--simulation_length", type=int, default=10)
    parser.add_argument("--signal_length", type=int, default=2)
    parser.add_argument("--n_episode", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # get environment parameters;
    
    if args.problem == 1:
        problem = problem_1
    elif args.problem == 2:
        problem = problem_2
    else:
        problem = problem_3
    mode = args.mode
    num_trial = args.n_trial
    num_intersection = args.n_intersection
    num_lane = args.n_lane
    lane_length = args.lane_length
    speed_limit = args.speed_limit
    simulation_length = args.simulation_length
    signal_length = args.signal_length
    render = False
    num_episode = args.n_episode
    lr = args.lr
    
    # problem solve;

    run_name = "./result/control/itscp/{}_{}".format(mode, int(time()))
    if not os.path.exists(run_name):
        os.makedirs(run_name)
    
    env = ItscpEnv()
    env.schedule_callback = problem
    env.config['num_intersection'] = num_intersection
    env.config['lane_length'] = lane_length
    env.config['num_lane'] = num_lane
    env.config['render'] = render
    env.config['policy_length'] = simulation_length
    env.config['signal_length'] = signal_length
    env.config['mode'] = mode
    env.config['speed_limit'] = speed_limit
    env.reset()
    
    for iter in range(num_trial):
        
        log_path = run_name + "/trial_{}".format(iter)
        
        # ours;
        env.render_eval = True
        trainer = Trainer(env, lr=lr)
        trainer.train(1, num_episode + 1, num_episode // 10, 1, log_path)