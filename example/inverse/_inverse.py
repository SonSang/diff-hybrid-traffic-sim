from math import ceil
from typing import List

import cma
import os
import numpy as np
from tqdm import tqdm
import torch as th
import scipy.optimize
import matplotlib.pyplot as plt
from copy import deepcopy

from road.network.road_network import RoadNetwork

class InverseProblem:
    
    '''
    Define and solve inverse problem posed in traffic simulation.
    
    In this problem, it has to estimate the initial state of the
    traffic that ends up in the given final state. 
    
    We can compare our estimation with the real initial state, but 
    even if the estimation does not correspond to it, if it ends up
    in the given final state, it is still valid estimation.
    '''
    
    def __init__(self, 
                num_trial: int, 
                num_timestep: int, 
                num_episode: int,
                delta_time: float, 
                speed_limit: float, 
                run_name: str,):

        self.network: RoadNetwork = None

        self.num_trial = num_trial
        self.num_timestep = num_timestep
        self.delta_time = delta_time
        self.speed_limit = speed_limit

        self.beg_state = None
        self.end_state = None

        # hyperparams for optimization methods;

        # number of episode evaluation that will be run for optimization;

        self.num_episode = num_episode

        # Gradient Descent;

        self.gd_lr = 1e-1

        # CMA-ES

        self.cma_sigma = 1        

        self.log_dir = "result/inverse/{}".format(run_name)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def initialize(self):

        '''
        Initialize this problem. 

        First randomly initialize the network (since there are other
        hyperparameters for the simulation other than traffic states),
        and then set the beginning state, which corresponds to correct
        answer. Then run simulation for [num_step] to get the [end_state],
        which we will use for optimization.
        '''

        self.init_network()

        self.set_state(self.random_state())

        self.beg_state = self.get_state()

        self.simulate()

        self.end_state = self.get_state()

    
    def simulate(self):

        '''
        Run simulation with road network for [num_step].
        '''

        for _ in range(self.num_timestep):

            self.network.forward(self.delta_time)

    def evaluate(self):

        '''
        For the same problem, use different methods to solve it,
        log the results, and render optimization graph to compare.
        '''

        gd_beg_errors, gd_end_errors = [], []
        cma_beg_errors, cma_end_errors = [], []
        nm_beg_errors, nm_end_errors = [], []
        ss_beg_errors, ss_end_errors = [], []

        for trial in range(self.num_trial):

            print("Trial # {}".format(trial))

            # initialize problem;

            self.initialize()

            # set random beginning state that optimization starts;

            est = self.random_state()

            for i in range(4):

                if i == 0:

                    # Gradient Descent (Ours);
                    
                    dir = self.log_dir + "/gd"
                    beg_errors, end_errors = self.solve_gd(est, self.gd_lr)
                    gd_beg_errors.append(beg_errors)
                    gd_end_errors.append(end_errors)

                elif i == 1:

                    # CMA-ES;

                    dir = self.log_dir + "/cma-es"
                    beg_errors, end_errors = self.solve_cma(est, self.cma_sigma)
                    cma_beg_errors.append(beg_errors)
                    cma_end_errors.append(end_errors)

                elif i == 2:

                    # Nelder-Mead;

                    dir = self.log_dir + "/nelder-mead"
                    beg_errors, end_errors = self.solve_scipy(est, "Nelder-Mead")
                    nm_beg_errors.append(beg_errors)
                    nm_end_errors.append(end_errors)

                elif i == 3:

                    # SLSQP;

                    dir = self.log_dir + "/slsqp"
                    beg_errors, end_errors = self.solve_scipy(est, "SLSQP")
                    ss_beg_errors.append(beg_errors)
                    ss_end_errors.append(end_errors)

                else:

                    raise ValueError()

                if not os.path.exists(dir):
                    os.makedirs(dir)

                self.log_error(dir + "/trial_{}.txt".format(trial), beg_errors, end_errors)

        # render graphs;
        self.render_graph(self.log_dir + "/beg_optimization_graph.png", 
                            gd_beg_errors,
                            cma_beg_errors,
                            nm_beg_errors,
                            ss_beg_errors)

        self.render_graph(self.log_dir + "/end_optimization_graph.png", 
                            gd_end_errors,
                            cma_end_errors,
                            nm_end_errors,
                            ss_end_errors)

    def solve_gd(self, est, lr: float):
        
        '''
        Solve problem using gradient descent. This method uses gradients
        from our differentiable traffic simulator.

        @ lr: Step size for gradient descent.
        '''

        est_beg_state = self.tensorize(est, requires_grad=True)

        beg_errors = []
        end_errors = []

        optimizer: th.optim.SGD = self.init_torch_optimizer(est_beg_state, lr)

        pbar = tqdm(range(self.num_episode))

        for _ in pbar:

            self.set_state(est_beg_state)

            self.simulate()

            est_end_state = self.get_state()

            # compute error based on beginning state;

            beg_error: th.Tensor = self.compute_error(self.beg_state, est_beg_state)
            end_error: th.Tensor = self.compute_error(self.end_state, est_end_state)

            beg_errors.append(beg_error.item())
            end_errors.append(end_error.item())

            # optimize;

            optimizer.zero_grad()
            end_error.backward()
            optimizer.step()

            # apply bound;

            with th.no_grad():
                lb, ub = self.bounds()
                est_beg_state[0][:] = th.max(est_beg_state[0], lb[0])
                est_beg_state[1][:] = th.max(est_beg_state[1], lb[1])
                est_beg_state[0][:] = th.min(est_beg_state[0], ub[0])
                est_beg_state[1][:] = th.min(est_beg_state[1], ub[1])

            pbar.set_description("GD : Error = {:.4f}".format(end_error.item()))

        return beg_errors, end_errors

    
    def solve_cma(self, est, sigma: float):
        
        '''
        Solve problem using CMA-ES method.
        '''

        est_beg_state = self.vectorize(est)

        # cma options: set bounds
        lb, ub = self.bounds()
        lb = th.cat([lb[0], lb[1]]).numpy().tolist()
        ub = th.cat([ub[0], ub[1]]).numpy().tolist()

        options = cma.CMAOptions()
        options.set('bounds', [lb, ub])
        options.set('verbose', -1)

        cma_optimizer = cma.CMAEvolutionStrategy(est_beg_state, sigma, options)
        num_iteration = ceil(self.num_episode / cma_optimizer.popsize)

        beg_errors = []
        end_errors = []

        pbar = tqdm(range(num_iteration))

        for _ in pbar:

            # optimize;

            solutions = cma_optimizer.ask()

            function_values = []

            # evaluation;

            for x in solutions:

                beg_error = self.compute_error(self.beg_state, self.unvectorize(x)).item()
                end_error = self.evaluate_vector_state(x).item()

                if len(end_errors) < self.num_episode:
                    
                    beg_errors.append(beg_error)
                    end_errors.append(end_error)

                function_values.append(end_error)

                pbar.set_description("CE : Error = {:.4f}".format(end_error))

            cma_optimizer.tell(solutions, function_values)
            # cma_optimizer.logger.add()
            # cma_optimizer.disp()


        return beg_errors, end_errors

    
    def solve_scipy(self, est, method: str):

        '''
        Solve problem using Nelder-Mead / SLSQP method.
        '''

        # set bounds
        lb, ub = self.bounds()
        lb = th.cat([lb[0], lb[1]]).numpy().tolist()
        ub = th.cat([ub[0], ub[1]]).numpy().tolist()
        bounds = scipy.optimize.Bounds(lb, ub)

        est_beg_state = self.vectorize(est)

        beg_errors = []
        end_errors = []

        pbar = tqdm(range(1))
        pbar_name = "NM" if method == "Nelder-Mead" else "SL"

        for _ in pbar:

            pbar.set_description("{} : Please check log file later".format(pbar_name))

            scipy.optimize.minimize(fun = InverseProblem.scipy_evaluate_function, 
                                    x0 = est_beg_state,
                                    bounds=bounds,
                                    args = (self, beg_errors, end_errors),
                                    options = {'maxiter': self.num_episode + 1},
                                    method=method,)

        # @BUGFIX: If length of [error] list is shorter than [self.num_optim_episode],
        # append last value for padding. This happens when we particularly use SLSQP,
        # because it uses finite difference method to estimate gradient and it can be
        # zero vector that makes the optimization to terminate early.

        shortage = self.num_episode - len(beg_errors)

        for _ in range(shortage):

            beg_errors.append(beg_errors[-1])
            end_errors.append(end_errors[-1])

        beg_errors = beg_errors[:self.num_episode]
        end_errors = end_errors[:self.num_episode]

        return beg_errors, end_errors

    def init_network(self):

        '''
        Randomly initialize road network to simulate.
        '''

        raise NotImplementedError()

    def random_state(self):

        '''
        Get random state of the road network.
        '''

        raise NotImplementedError()

    def set_state(self, state):

        '''
        Set state of the road network.
        '''

        raise NotImplementedError()

    
    def get_state(self):

        '''
        Get state of the road network.
        '''

        raise NotImplementedError()

    def tensorize(self, state, requires_grad: bool = True):

        '''
        Turn the given state into a pytorch tensor.
        '''

        raise NotImplementedError()

    def init_torch_optimizer(self, state, lr: float) -> th.optim.Adam:

        '''
        Initialize pytorch Adam optimizer for given state.
        '''

        raise NotImplementedError()

    def vectorize(self, state):

        '''
        Turn the given state into a single vector to be 
        used in gradient-free algorithms.
        '''

        raise NotImplementedError()

    def unvectorize(self, vstate):

        '''
        Inverse function of [vectorize].
        '''

        raise NotImplementedError()

    def bounds(self):

        '''
        Get lower bound and upper bound of the state.
        '''

        raise NotImplementedError()

    def evaluate_vector_state(self, vstate):

        '''
        Evaluate the given vectorized beginning state.
        This function will be used in gradient-free
        optimization algorithms.
        '''

        est_beg_state = self.unvectorize(vstate)

        self.set_state(est_beg_state)

        self.simulate()

        est_end_state = self.get_state()

        return self.compute_error(self.end_state, est_end_state)

    @staticmethod
    def scipy_evaluate_function(vstate, problem, beg_errors, end_errors):

        '''
        Evaluation function for scipy-based gradient-free 
        optimization algorithms (Nelder-Mead, SLSQP). 

        Store the errors in given error lists.
        '''

        problem: InverseProblem = problem

        end_error = problem.evaluate_vector_state(vstate)

        est_beg_state = problem.unvectorize(vstate)

        beg_error = problem.compute_error(problem.beg_state, est_beg_state)

        beg_errors.append(beg_error.item())
        end_errors.append(end_error.item())

        return end_error.item()

    def compute_error(self, sa, sb):

        '''
        Compute error between two states.
        '''

        raise NotImplementedError()

    def log_error(self, path: str, beg_errors: List[float], end_errors: List[float]):

        '''
        Log the errors arise from optimization.
        '''

        with open(path, 'w') as f:

            for i in range(len(beg_errors)):

                f.write("{} {}\n".format(beg_errors[i], end_errors[i]))

    def render_graph(self, 
                    path: str,
                    gd_errors: List[List[float]], 
                    cma_errors: List[List[float]], 
                    nm_errors: List[List[float]], 
                    ss_errors: List[List[float]]):

        '''
        Render optimization graph for all of the optimization methods.
        '''

        gd_errors: np.ndarray = np.array(gd_errors)
        cma_errors: np.ndarray = np.array(cma_errors)
        nm_errors: np.ndarray = np.array(nm_errors)
        ss_errors: np.ndarray = np.array(ss_errors)

        num_step = gd_errors.shape[1]

        colors = [
            [0, 0.45, 0.74],            # Ours (GD)
            [0.85, 0.33, 0.1],          # CMAES
            [0.4940, 0.1840, 0.5560],   # Nelder-Mead
            [0.9290, 0.6940, 0.1250],   # SLSQP
        ]

        params = {'legend.fontsize': 25,
                'figure.figsize': (12, 9),
                'axes.labelsize': 30,
                'axes.titlesize': 30,
                'xtick.labelsize':30,
                'ytick.labelsize':30}

        plt.rcParams.update(params)
        plt.grid(alpha=0.3)

        plt.clf()

        xx = range(1, num_step + 1)

        for i in range(4):

            if i == 0:
            
                errors = gd_errors
                method = "Ours"

            elif i == 1:

                errors = cma_errors
                method = "CMAES"

            elif i == 2:

                errors = nm_errors
                method = "Nelder-Mead"

            elif i == 3:

                errors = ss_errors
                method = "SLSQP"

            else:

                raise ValueError()

            mean_ = errors.mean(0)
            std_ = errors.std(0)
            color = colors[i]

            plt.plot(xx, mean_, color=color, label=method, linewidth=4)
            plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)

        plt.legend()
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.grid()

        plt.savefig(path)