from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import RoadNetwork
from highway_env.road.lane import StraightLane, AbstractLane, LineType
from highway_env.road.regulation import RegulatedRoad
from typing import Optional, Dict, List
import numpy as np
import pygame
import torch as th
from gym import spaces

from example.common._comp_lane import CompLane
from example.control.itscp._env_config import default_config
from example.control.itscp._viewer import IntersectionViewer
from example.control.itscp._simulator import ItscpRoadNetwork as SimRoadNetwork

from road.lane._base_lane import BaseLane
from road.network.route import MacroRoute
from road.lane.dmacro_lane import dMacroLane, MacroLane
from road.lane.dmicro_lane import dMicroLane, MicroLane

from dmath.operation import sigmoid

class LaneID:

    '''
    Lane ID for lanes in this scene.

    @ row, col: Identifier for intersection in the grid.
    @ loc: North, East, West, South, Mid.
    @ ploc: North, East, West, South, None, prev direction for 'mid'.
    @ approaching: Whether or not vehicles on this lane approach to this intersection.
    @ lane_id: ID of the lane, 0 starts from left (from the view of approaching side).
    '''

    def __init__(self, row: int, col: int, loc: str, ploc: str, approaching: bool, lane_id: int):

        self.row = row
        self.col = col
        self.loc = loc
        self.ploc = ploc
        self.approaching = approaching
        self.lane_id = lane_id

    def __str__(self):

        app = "approaching" if self.approaching else "leaving"

        return "{}_{}_{}_{}_{}_{}".format(self.row, self.col, self.loc, self.ploc, app, self.lane_id)

    def __eq__(self, another: object) -> bool:
        
        return self.row == another.row and \
                self.col == another.col and \
                self.loc == another.loc and \
                self.ploc == another.ploc and \
                self.approaching == another.approaching and \
                self.lane_id == another.lane_id

    def __hash__(self) -> int:
        
        return hash(str(self))

def itscp_random_schedule(lane_id: List[LaneID], num_timestep: int):

    '''
    Set a randomized schedule. 

    To keep certain level of reality, maintain certain state for multiple time steps
    '''

    num_session = 5
    num_timestep_per_session = num_timestep // num_session

    schedule: Dict[LaneID, List[float]] = {}

    for id in lane_id:

        curr_schedule = []

        for session in range(num_session):

            r = np.random.random((1)).item()

            for _ in range(num_timestep_per_session):

                curr_schedule.append(r)

            curr_schedule = curr_schedule[:num_timestep]

            schedule[id] = curr_schedule

    return schedule

class ItscpEnv(AbstractEnv):
    
    '''
    Intersection Signal Control Problem (ITSCP).

    In this problem, we have to estimate the optimal time allocations of signals
    for given schedule of incoming traffic flow.
    '''

    def __init__(self, schedule_callback=itscp_random_schedule):

        self.schedule_callback = schedule_callback

        self.simulator: SimRoadNetwork = None
        self.lane: Dict[LaneID, CompLane] = {}
        self.schedule: Dict[LaneID, List[float]] = {}

        # schedule of routes;

        self.macro_route_schedule: List[MacroRoute] = []

        # inner structure needed for computing rewards;

        self.queue_length: Dict[LaneID, List] = {}
        self.flux: Dict[LaneID, List] = {}
        self.avg_speed: List = []

        super().__init__()

    @classmethod
    def default_config(cls) -> dict:
        
        config = super().default_config()

        config.update(default_config)
        
        return config

    def action_size(self):
        
        # entire simulation length in time (sec);

        simulation_length = self.config["policy_length"] * self.config["duration"]

        # each signal phase needs one floating point value as an action;
        
        return (int)(simulation_length / self.config["signal_length"]) * (self.num_intersection ** 2)
    
    def reset(self):

        return self._reset()

    def _reset(self) -> None:

        # set np random seed;

        if self.config["random_seed"] > 0:

            np.random.seed(self.config["random_seed"])

        self.num_intersection = self.config["num_intersection"]
        self.num_lane = self.config['num_lane']
        self.num_timestep = self.config["policy_length"] * self.config["duration"] * self.config["simulation_frequency"]

        self._make_road()

        # initialize schedule by callback;

        self.schedule = self.schedule_callback(list(self.lane.keys()), self.num_timestep)

        # define spaces;

        self.observation_space = spaces.Box(shape=(self.config['num_schedule_obs'] * len(self.lane),), low=0, high=1, dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([self.config["action_min"] for _ in range(self.action_size())], dtype=np.float32), \
            np.array([self.config["action_max"] for _ in range(self.action_size())], dtype=np.float32))

        self.update_metadata()
        self.time = self.steps = 0
        self.done = False

        # constants for reward;

        self.reward_queue_c = -1.0          # queue length constant;
        self.reward_flux_c = 1.0            # flux constant;

        # initialize macro route schedule randomly;

        self._make_macro_route()

        # initialize micro vehicles that would be generated;

        self._make_micro_route()

        return self.observe()

    def _make_macro_route(self):

        self.macro_route_schedule.clear()

        for _ in range(self.num_timestep):

            self.macro_route_schedule.append(self.simulator.create_random_macro_route())

    def _make_micro_route(self):

        self.simulator.lane_waiting_micro_vehicle.clear()
        self.simulator.lane_waiting_micro_route.clear()

        max_num_micro_vehicle_per_lane = self.config['max_num_micro_vehicle_per_lane']

        for lane_id in self.simulator.lane.keys():

            self.simulator.lane_waiting_micro_vehicle[lane_id] = []
            self.simulator.lane_waiting_micro_route[lane_id] = []

            for _ in range(max_num_micro_vehicle_per_lane):

                nv, nr = self.simulator.create_default_vehicle_with_random_route(lane_id)

                self.simulator.lane_waiting_micro_vehicle[lane_id].append(nv)
                self.simulator.lane_waiting_micro_route[lane_id].append(nr)
    
    def _make_road(self) -> None:

        """
        Make a 4-way intersection.
        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east) + (lane_id)
        :return: the intersection road
        """
        
        num_intersection = self.config["num_intersection"]
        num_lane = self.config['num_lane']

        lane_width = AbstractLane.DEFAULT_WIDTH                 # 4
        
        right_turn_radius = lane_width + 10                     # 4 + 5 = 9
        
        outer_distance = right_turn_radius + (lane_width * (num_lane - 3 + 0.5))    # 9 + (4 * (3 - 3 + 0.5)) = 11
        access_length = self.config["lane_length"]              # 100
        
        # initialize road networks;
        
        net = RoadNetwork()
        self.road = RegulatedRoad(network=net, np_random=self.np_random)
        
        self.simulator = SimRoadNetwork(self.config["speed_limit"])
        self.lane.clear()

        for row in range(num_intersection):
            
            for col in range(num_intersection):

                center = np.array([col * (outer_distance + access_length), 
                                    row * (outer_distance + access_length)]) * 2.0

                # generate branch lanes;

                approaching_lane_id: List[LaneID] = []

                for corner in range(4):

                    angle = np.radians(90 * corner)

                    # rotation matrix;

                    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

                    # note: since y coordinates are flipped, positive y coordinates located below;

                    for approaching in [True, False]:

                        if corner == 0:
                            loc = "south" if approaching else "east"
                        elif corner == 1:
                            loc = "west" if approaching else "south"
                        elif corner == 2:
                            loc = "north" if approaching else "west"
                        else:
                            loc = "east" if approaching else "north"

                        for lane_id in range(num_lane):

                            c_lane_id = LaneID(row, col, loc, None, approaching, lane_id)
                            
                            # env lane

                            if approaching:

                                start = np.array([lane_width * (lane_id + 0.5), access_length + outer_distance])
                                end = np.array([lane_width * (lane_id + 0.5), outer_distance])

                            else:

                                start = np.flip([lane_width * (lane_id + 0.5), access_length + outer_distance], axis=0)
                                end = np.flip([lane_width * (lane_id + 0.5), outer_distance], axis=0)

                            start = center + rotation @ start
                            end = center + rotation @ end

                            start_node = str(c_lane_id) + "_outer"
                            end_node = str(c_lane_id) + "_inner"

                            self._make_lane(c_lane_id, start, end, start_node, end_node)

                            if approaching:

                                approaching_lane_id.append(c_lane_id)

                # generate connecting lanes in the middle of intersection;

                idx = 0

                for lane_id in approaching_lane_id:

                    curr_lane = self.lane[lane_id]

                    curr_lane_end_pos = curr_lane.env_lane.position(curr_lane.env_lane.length, 0)

                    for turn in ['straight', 'left', 'right']:

                        if turn == 'left':

                            # @TODO: do not use left turn now because of signal;

                            continue

                            if lane_id.lane_id != 0:

                                continue

                            if lane_id.loc == 'north':
                                n_loc = 'east'
                            elif lane_id.loc == 'west':
                                n_loc = 'north'
                            elif lane_id.loc == 'east':
                                n_loc = 'south'
                            else:
                                n_loc = 'west'
                        
                        elif turn == 'right':

                            if lane_id.lane_id != num_lane - 1:

                                continue

                            if lane_id.loc == 'north':
                                n_loc = 'west'
                            elif lane_id.loc == 'west':
                                n_loc = 'south'
                            elif lane_id.loc == 'east':
                                n_loc = 'north'
                            else:
                                n_loc = 'east'

                        else:                                

                            # straight;

                            if lane_id.loc == 'north':
                                n_loc = 'south'
                            elif lane_id.loc == 'west':
                                n_loc = 'east'
                            elif lane_id.loc == 'east':
                                n_loc = 'west'
                            else:
                                n_loc = 'north'

                        n_lane_id = LaneID(lane_id.row, 
                                                    lane_id.col, 
                                                    n_loc, 
                                                    None,
                                                    False, 
                                                    lane_id.lane_id)

                        next_lane = self.lane[n_lane_id]

                        next_lane_start_pos = next_lane.env_lane.position(next_lane.env_lane.length, 0)

                        c_lane_id = LaneID(lane_id.row, 
                                            lane_id.col, 
                                            'mid',
                                            lane_id.loc, 
                                            True, 
                                            idx)

                        idx += 1

                        start_node = str(lane_id) + "_inner"
                        end_node = str(n_lane_id) + "_inner"

                        self._make_lane(c_lane_id, curr_lane_end_pos, next_lane_start_pos, start_node, end_node)

                        self._connect_sim_lane(lane_id, c_lane_id)
                        self._connect_sim_lane(c_lane_id, n_lane_id)

        # connect different intersections with each other;

        for row in range(num_intersection):
            
            for col in range(num_intersection):

                if row > 0:

                    # connect north

                    for approaching in [True, False]:

                        for lane_id in range(num_lane):

                            curr_lane_id = LaneID(row, col, 'north', None, approaching, lane_id)

                            conn_lane_id = LaneID(row - 1, col, 'south', None, not approaching, lane_id)

                            if approaching:

                                self._connect_sim_lane(conn_lane_id, curr_lane_id)

                            else:

                                self._connect_sim_lane(curr_lane_id, conn_lane_id)

                if col > 0:

                    # connect west

                    for approaching in [True, False]:

                        for lane_id in range(num_lane):

                            curr_lane_id = LaneID(row, col, 'west', None, approaching, lane_id)

                            conn_lane_id = LaneID(row, col - 1, 'east', None, not approaching, lane_id)

                            if approaching:

                                self._connect_sim_lane(conn_lane_id, curr_lane_id)

                            else:

                                self._connect_sim_lane(curr_lane_id, conn_lane_id)
    
    def _make_lane(self, lane_id: LaneID, start_pos: np.ndarray, end_pos: np.ndarray, start_node: str, end_node: str):
        
        '''
        Make a lane, which includes a lane for rendering (env_lane) and simulation (sim_lane).
        '''

        num_intersection = self.config['num_intersection']
        num_lane = self.config['num_lane']
        speed_limit = self.config['speed_limit']
        cell_length = self.config['cell_length']

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        approaching = lane_id.approaching

        if lane_id.loc == 'mid':

            line_types = [n, n]

        else:

            if approaching:

                line_types = [c if lane_id.lane_id == 0 else n, c if lane_id.lane_id == num_lane - 1 else s]

            else:

                line_types = [c if lane_id.lane_id == num_lane - 1 else s, n if lane_id.lane_id == 0 else n]

        env_lane = StraightLane(start_pos, end_pos, line_types=line_types, speed_limit=speed_limit)

        self.road.network.add_lane(start_node, end_node, env_lane)

        # sim lane;

        row = lane_id.row
        col = lane_id.col
        lane_length = env_lane.length

        if self.config['mode'] == 'macro':

            sim_lane = dMacroLane(len(self.simulator.lane), lane_length, speed_limit, cell_length)

        elif self.config['mode'] == 'micro':

            # sim_lane = dMicroLane(len(self.simulator.lane), lane_length, speed_limit)
            sim_lane = MicroLane(len(self.simulator.lane), lane_length, speed_limit)

        elif self.config['mode'] == 'hybrid':

            if row == 0 or row == num_intersection - 1 or \
                col == 0 or col == num_intersection - 1:

                sim_lane = dMacroLane(len(self.simulator.lane), lane_length, speed_limit, cell_length)

            else:

                sim_lane = dMicroLane(len(self.simulator.lane), lane_length, speed_limit)

        self.simulator.add_lane(sim_lane)

        # comp lane;

        c_lane = CompLane(self.road, env_lane, sim_lane, not approaching)

        self.lane[lane_id] = c_lane

    def _connect_sim_lane(self, prev_lane_id: LaneID, next_lane_id: LaneID):

        '''
        Connect two lanes in simulation.
        '''

        prev_lane = self.lane[prev_lane_id].sim_lane
        next_lane = self.lane[next_lane_id].sim_lane

        self.simulator.connect_lane(prev_lane.id, next_lane.id)

    def observe(self):
        
        '''
        Observe schedule and return.
        '''
        
        obs = []

        num_schedule_obs = self.config["num_schedule_obs"]

        for lane_id in self.lane.keys():

            sim_lane = self.lane[lane_id].sim_lane

            sc = self.schedule[lane_id]
            t = len(sc) // num_schedule_obs

            for k in range(num_schedule_obs):

                if len(sim_lane.prev_lane) == 0:

                    t0 = int(t * k)
                    t1 = int(t * k + t)
                    
                    if t1 > len(sc):
                        t1 = len(sc)
                    
                    rho_sum = 0

                    for p in range(t0, t1):
                        rho_sum += sc[p]
                    
                    rho_sum /= (t1 - t0)
                    obs.append(rho_sum)
                
                else:

                    obs.append(0)

        return np.array(obs).astype(self.observation_space.dtype)

    def step(self, action, differentiable: bool):

        """
        Perform an action and step the environment dynamics.
        :return: a tuple (observation, reward, terminal, info)
        """

        self.steps += 1

        self.queue_length.clear()
        self.flux.clear()
        self.avg_speed.clear()

        self._simulate(action, differentiable)

        obs = self.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        self.queue_length.clear()
        self.flux.clear()
        
        return obs, reward, terminal, info

    def _is_static_speed(self, speed: float, differentiable: bool):

        '''
        Return True (1.0) if the given speed is a static speed, else False (0.0).
        '''

        static_speed = self.config['static_speed']

        if not isinstance(speed, th.Tensor):

            speed = th.tensor(speed)

        if differentiable:

            # apply differentiable operation;

            is_static = sigmoid(static_speed - speed, constant=16)

        else:

            if speed < static_speed:

                is_static = 1.0

            else:

                is_static = 0.0

        return is_static

    def _simulate_step(self, action, differentiable: bool):

        '''
        Take single simulation step.
        '''

        curr_frame = self.time

        # set borders according to schedule and signal, which is determined by [action];

        for lane_id in self.lane.keys():

            sim_lane: BaseLane = self.lane[lane_id].sim_lane

            # schedule;

            incoming = -1

            if len(sim_lane.prev_lane) == 0:

                schedule = self.schedule[lane_id]
                incoming = schedule[curr_frame]

            # signal;

            signal_info = self.lane_signal_info(lane_id, action, curr_frame, differentiable)

            self.simulator.lane_signal[sim_lane.id] = signal_info[1]
            self.simulator.lane_incoming[sim_lane.id] = incoming

        # set macro route;

        self.simulator.macro_route = self.macro_route_schedule[curr_frame]

        # take simulation step;

        dt = 1.0 / self.config["simulation_frequency"]
        self.simulator.forward(dt, differentiable)
        self.time += 1

        # append obs;

        static_speed = self.config['static_speed']

        for lane_id in self.lane.keys():

            if lane_id not in self.queue_length.keys():

                self.queue_length[lane_id] = []

            if lane_id not in self.flux.keys():

                self.flux[lane_id] = []

            sim_lane = self.lane[lane_id].sim_lane

            # queue length;

            lane_queue_length = 0

            if isinstance(sim_lane, MacroLane):

                for c in sim_lane.curr_cell:

                    density = c.state.q.r

                    num_vehicle = density * sim_lane.cell_length / self.simulator.vehicle_length

                    speed = c.state.u

                    is_static = self._is_static_speed(speed, differentiable)

                    lane_queue_length = lane_queue_length + (is_static * num_vehicle)

            elif isinstance(sim_lane, MicroLane):

                for v in sim_lane.curr_vehicle:

                    speed = v.speed

                    if not isinstance(speed, th.Tensor):

                        speed = th.tensor(speed)

                    if differentiable:

                        # apply differentiable operation;

                        is_static = sigmoid(static_speed - speed, constant=32.0)

                    else:

                        if speed < static_speed:

                            is_static = 1.0

                        else:

                            is_static = 0.0

                    lane_queue_length = lane_queue_length + is_static # * v.length

                # also consider waiting vehicles out of scene;

                # if len(sim_lane.prev_lane) == 0:

                #     lane_queue_length += self.simulator.lane_micro_max_counter[sim_lane.id] - \
                #                             self.simulator.lane_micro_counter[sim_lane.id]

            else:

                raise ValueError()

            # pow to punish longer queue lengths more
            
            lane_queue_length = (lane_queue_length ** 2.0) * dt

            self.queue_length[lane_id].append(lane_queue_length)

        if self.config["render"]:

            self.render(action)

    def _simulate(self, action, differentiable: bool):

        self.time = 0

        for _ in range(self.num_timestep):

            self._simulate_step(action, differentiable)

    def _reward(self, action):
        """
        The reward is weighted sum of following factors.
        1. Queue length: Queue is defined as cells that have speed less than threshold.
        2. Flux: Flux that passed the inner part of the intersection in a single phase.
        """
        reward = 0

        for lane_id in self.lane.keys():

            queue_length = self.queue_length[lane_id]
            flux = self.flux[lane_id]

            for x in queue_length:

                reward = reward + self.reward_queue_c * x

            # for x in flux:

            #     reward = reward + self.reward_flux_c * x

        # for x in self.avg_speed:

        #     reward = reward + x

        # reward = reward / len(self.avg_speed)

        return reward

    def _is_terminal(self) -> bool:
        
        """
        The episode is over if the time is out.
        """
        
        return self.steps >= self.config["duration"] or self.time >= self.num_timestep

    def _info(self, obs, action):
        info = {
        }
        return info

    def render(self, action, mode: str = 'human') -> Optional[np.ndarray]:
        
        """
        Render the environment.
        Create a viewer if none exists, and use it to render an image.

        @ action: needed to render traffic signal
        :param mode: the rendering mode
        """

        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = IntersectionViewer(self, self.config)

        self.enable_auto_render = True

        self.viewer.display(action)
        
        if not self.viewer.offscreen:
        
            self.viewer.handle_events()
        
        if mode == 'rgb_array':
        
            image = self.viewer.get_image()
            return image

    def render_callback(self, action):

        # erase micro vehicles;

        self.road.vehicles.clear()

        self.render_signal_callback(action)

        self.render_traffic_callback()

    def render_signal_callback(self, action):

        '''
        Render signal for rightmost lanes of approaching roads.
        '''

        for lane_id, lane in self.lane.items():

            if lane_id.lane_id == self.num_lane - 1 and \
                lane_id.approaching and \
                lane_id.loc != 'mid':

                _, green = self.lane_signal_info(lane_id, action, self.time, False)

                position = lane.env_lane.position(lane.env_lane.length, 3)
                position = self.viewer.sim_surface.vec2pix(position)

                color = [0, 255, 0] if green else [255, 0, 0]

                pygame.draw.circle(surface=self.viewer.sim_surface, color=color, center=position, radius=5)

    def render_traffic_callback(self):

        '''
        Render traffic on lane.
        '''

        for c_lane in self.lane.values():

            c_lane.render(self.viewer.sim_surface)

    '''
    Signal functions.
    '''

    def lane_signal_info(self, lane_id: LaneID, action, curr_frame, differentiable: bool):

        '''
        Return signal information for the given lane.

        The signal consists of two float values, which is [prev_signal] and [next_signal].
        [prev_signal] is close to 1.0 if the signal between the prev lane and this lane is green, else close to 0.0.
        [next_signal] is close to 1.0 if the signal between the next lane and this lane is green, else close to 0.0.
        '''

        num_signal_frame = self.config['simulation_frequency'] * self.config['signal_length']

        sq_num_int = self.num_intersection ** 2

        curr_phase = min(curr_frame // num_signal_frame, len(action) // sq_num_int - 1)

        curr_phase_action = action[curr_phase * sq_num_int: (curr_phase + 1) * sq_num_int]

        curr_action = curr_phase_action[lane_id.row * self.num_intersection + lane_id.col]

        # turn [curr_action] into Tensor to use sigmoid function;

        if not isinstance(curr_action, th.Tensor):

            curr_action = th.tensor(curr_action)

        # current progress in this phase;
        # if it is smaller than [curr_action], WE light is on;
        # else, NS light is on;

        curr_phase_progress = min((curr_frame % num_signal_frame) / num_signal_frame, 1.0)

        if lane_id.loc == 'mid':
            
            # if the lane is connecting lane in the intersection, [next_signal] is always 1.0;
            
            next_signal = 1.0
            
            # [prev_signal] is determined by signal;
            # @TODO: the signal has to be that of [prev_lane], not itself;

            if lane_id.ploc == "west" or lane_id.ploc == "east":

                prev_signal = sigmoid(curr_action - curr_phase_progress, constant=16) if differentiable \
                                else float(curr_action > curr_phase_progress)

            else:

                prev_signal = sigmoid(curr_phase_progress - curr_action, constant=16) if differentiable \
                                else float(curr_phase_progress > curr_action)
                
        else:

            # if the lane is leaving lane, [prev_signal] and [next_signal] are always 1.0;

            if not lane_id.approaching:
                    
                next_signal = 1.0
                prev_signal = 1.0

            # if the lane is approaching lane, [prev_signal] is always True;
            # [next_signal] is determined by signal;

            else:

                prev_signal = 1.0

                if lane_id.loc == "west" or lane_id.loc == "east":

                    next_signal = sigmoid(curr_action - curr_phase_progress, constant=16) if differentiable \
                                else float(curr_action > curr_phase_progress)

                else:

                    next_signal = sigmoid(curr_phase_progress - curr_action, constant=16) if differentiable \
                                else float(curr_phase_progress > curr_action)

        return prev_signal, next_signal
