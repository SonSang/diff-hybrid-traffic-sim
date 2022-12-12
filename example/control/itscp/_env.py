from math import exp
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
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

from road.road_network import RoadNetwork as SimRoadNetwork
from road.lane._base_lane import BaseLane
from road.callback import default_bdry_callback
from road.lane.dmacro_lane import dMacroLane
from road.lane.dmicro_lane import dMicroLane
from model.macro._arz import ARZ

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

        # inner structure needed for computing rewards;

        self.queue_length: Dict[LaneID, List] = {}
        self.flux: Dict[LaneID, List] = {}

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

        self.num_intersection = self.config["num_intersection"]
        self.num_lane = self.config['num_lane']
        self.num_timestep = self.config["policy_length"] * self.config["duration"] * self.config["simulation_frequency"]

        self._make_road()

        # initialize schedule by callback;

        self.schedule = self.schedule_callback(list(self.lane.keys()), self.num_timestep)

        # Define spaces

        self.observation_space = spaces.Box(shape=(self.config['num_schedule_obs'] * len(self.lane),), low=0, high=1, dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([self.config["action_min"] for _ in range(self.action_size())], dtype=np.float32), \
            np.array([self.config["action_max"] for _ in range(self.action_size())], dtype=np.float32))

        self.update_metadata()
        self.time = self.steps = 0
        self.done = False

        # Constants for reward

        self.reward_queue_c = -1.0          # Queue length constant
        self.reward_flux_c = 1.0            # Flux constant

        return self.observe()
    
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

            sim_lane = dMicroLane(len(self.simulator.lane), lane_length, speed_limit)

        elif self.config['mode'] == 'hybrid':

            if row == 0 or row == num_intersection - 1 or \
                col == 0 or col == num_intersection - 1:

                sim_lane = dMacroLane(len(self.simulator.lane), lane_length, speed_limit, cell_length)

            else:

                sim_lane = dMicroLane(len(self.simulator.lane), lane_length, speed_limit)

        self.simulator.add_lane(sim_lane.id, sim_lane)

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

            sc = self.schedule[lane_id]
            t = len(sc) // num_schedule_obs

            for k in range(num_schedule_obs):

                t0 = int(t * k)
                t1 = int(t * k + t)
                
                if t1 > len(sc):
                    t1 = len(sc)
                
                rho_sum = 0

                for p in range(t0, t1):
                    rho_sum += sc[p]
                
                rho_sum /= (t1 - t0)
                obs.append(rho_sum)

        return np.array(obs).astype(self.observation_space.dtype)

    def step(self, action):

        """
        Perform an action and step the environment dynamics.
        :return: a tuple (observation, reward, terminal, info)
        """

        self.steps += 1

        self.queue_length.clear()
        self.flux.clear()

        self._simulate(action)

        obs = self.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        self.queue_length.clear()
        self.flux.clear()
        
        return obs, reward, terminal, info

    def _simulate_step(self, action):

        '''
        Take single simulation step.
        '''

        curr_frame = self.time

        # set borders according to schedule and signal, which is determined by [action];

        for lane_id in self.lane.keys():

            # schedule;

            incoming = -1

            if lane_id.row == 0 or lane_id.row == self.num_intersection - 1 or \
                lane_id.col == 0 or lane_id.col == self.num_intersection - 1:

                schedule = self.schedule[lane_id]
                incoming = schedule[curr_frame]

            # signal;

            signal_info = self.is_green_light(lane_id, action, curr_frame)

            sim_lane: BaseLane = self.lane[lane_id].sim_lane

            if isinstance(sim_lane, dMacroLane):

                sim_lane.bdry_callback = ItscpEnv.macro_bdry_callback

            elif isinstance(sim_lane, dMicroLane):

                sim_lane.bdry_callback = ItscpEnv.micro_bdry_callback

            else:

                raise ValueError()

            sim_lane.bdry_callback_args = {'lane': sim_lane,
                                            'signal_info': signal_info,
                                            'incoming': incoming, 
                                            'id': lane_id}

        # take simulation step;

        dt = 1.0 / self.config["simulation_frequency"]
        self.simulator.forward(dt)
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

            if isinstance(sim_lane, dMacroLane):

                for cell in sim_lane.curr_cell:

                    rho = cell.state.q.r
                    qlen = sim_lane.cell_length * rho
                    speed = cell.state.u

                    if speed < static_speed:

                        is_static = 1.0

                    else:

                        is_static = 0.0

                    if isinstance(speed, th.Tensor):

                        is_static = is_static + speed.detach() - speed

                    lane_queue_length = lane_queue_length + is_static * qlen

            elif isinstance(sim_lane, dMicroLane):

                for v in sim_lane.curr_vehicle:

                    speed = v.speed

                    if speed < static_speed:

                        is_static = 1.0

                    else:

                        is_static = 0.0

                    if isinstance(speed, th.Tensor):

                        is_static = is_static + speed.detach() - speed

                    lane_queue_length = lane_queue_length + is_static * v.length

            else:

                raise ValueError()

            # pow to punish longer queue lengths more
            
            lane_queue_length = (lane_queue_length ** 2.0) * dt

            self.queue_length[lane_id].append(lane_queue_length)

            # flux;

            lane_flux = 0
            
            if lane_id.approaching:

                if isinstance(sim_lane, dMacroLane):

                    flux = sim_lane.curr_cell[-1].state.flux_r() 

                    lane_flux = lane_flux + flux * sim_lane.cell_length * dt

                elif isinstance(sim_lane, dMicroLane):

                    if len(sim_lane.curr_vehicle):

                        hv = sim_lane.curr_vehicle[-1]

                        tip = hv.position + hv.length * 0.5

                        if tip >= sim_lane.length:

                            going_out = 1.0

                        else:

                            going_out = 0.0
                        
                        if isinstance(hv.position, th.Tensor):

                            going_out = going_out + hv.position - hv.position.detach()

                        lane_flux = lane_flux + going_out * hv.length * hv.speed * dt

                else:

                    raise ValueError()
                
            self.flux[lane_id].append(lane_flux)

    def _simulate(self, action):

        self.time = 0

        for _ in range(self.num_timestep):

            self._simulate_step(action)

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

            for x in flux:

                reward = reward + self.reward_flux_c * x

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

                _, _, green, _ = self.is_green_light(lane_id, action, self.time)

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

    def is_green_light(self, lane_id: LaneID, action, curr_frame):

        '''
        Return signal information for the given lane.

        The signal consists of two boolean values, which is [prev_green] and [next_green].
        [prev_green] indicates if the signal between the prev lane and this lane is green.
        [next_green] indicates if the signal between this lane and the next lane is green.

        It also returns the corresponding element in [action] that made such signal decisions.
        '''

        num_signal_frame = self.config['simulation_frequency'] * self.config['signal_length']

        sq_num_int = self.num_intersection ** 2

        curr_phase = min(curr_frame // num_signal_frame, len(action) // sq_num_int - 1)

        curr_phase_action = action[curr_phase * sq_num_int: (curr_phase + 1) * sq_num_int]

        curr_action = curr_phase_action[lane_id.row * self.num_intersection + lane_id.col]

        # current progress in this phase;
        # if it is smaller than [curr_action], WE light is on;
        # else, NS light is on;

        curr_phase_progress = min((curr_frame % num_signal_frame) / num_signal_frame, 1.0)


        if lane_id.loc == 'mid':
            
            # if the lane is connecting lane in the intersection, [next_green] is always True;
            
            next_green = True
            next_green_action = None

            # [prev_green] is determined by signal;

            if curr_phase_progress < curr_action:

                prev_green = lane_id.ploc == "west" or lane_id.ploc == "east"

            else:

                prev_green = lane_id.ploc == "north" or lane_id.ploc == "south"

            prev_green_action = curr_action

        else:

            # if the lane is leaving lane, [prev_green] and [next_green] are always True;

            if not lane_id.approaching:
                    
                next_green = True
                next_green_action = None

                prev_green = True
                prev_green_action = None

            # if the lane is approaching lane, [prev_green] is always True;
            # [next_green] is determined by signal;

            else:

                prev_green = True
                prev_green_action = None

                if curr_phase_progress < curr_action:

                    next_green = lane_id.loc == "west" or lane_id.loc == "east"

                else:

                    next_green = lane_id.loc == "north" or lane_id.loc == "south"

                next_green_action = curr_action

        return prev_green, prev_green_action, next_green, next_green_action

    @staticmethod
    def macro_bdry_callback(args: Dict):

        '''
        Bdry function for macro lane when using traffic signals.

        If it is green signal, it works same as default bdry function.
        Else, we assume that the next lane is blocked for approaching lanes.
        We assume the prev lane is blocked for middle lanes.
        '''

        lane: dMacroLane = args['lane']
        signal_info = args['signal_info']
        incoming = args['incoming']
        id: LaneID = args['id']

        default_bdry_callback(args)

        (prev_green, prev_action, next_green, next_action) = signal_info

        if id.loc == 'mid':

            # if this lane is 'mid', left cell is determined by signal;

            green_density = lane.leftmost_cell.state.q.r
            green_speed = lane.leftmost_cell.state.u

            red_density = 0.0
            red_speed = lane.speed_limit

            if prev_green:

                leftmost_density = green_density
                leftmost_speed = green_speed

            else:

                leftmost_density = red_density
                leftmost_speed = red_speed

            # differentiable [action];

            if isinstance(prev_action, th.Tensor):

                if id.ploc == "west" or id.ploc == "east":

                    # if WE-dir, when [action] decreases, it becomes closer to [red];

                    leftmost_density = leftmost_density + (green_density - red_density) * (prev_action - prev_action.detach())
                    leftmost_speed = leftmost_speed + (green_speed - red_speed) * (prev_action - prev_action.detach())

                else:

                    # else, when [action] increases, it becomes closer to [red];

                    leftmost_density = leftmost_density + (green_density - red_density) * (prev_action.detach() - prev_action)
                    leftmost_speed = leftmost_speed + (green_speed - red_speed) * (prev_action.detach() - prev_action)

            lane.set_leftmost_cell(leftmost_density, leftmost_speed)

        else:

            if id.approaching:

                # if there is incoming, set leftmost cell;

                if incoming >= 0:

                    r = incoming
                    u = ARZ.compute_u(r, 0, lane.speed_limit)
                    lane.set_leftmost_cell(r, u)

                # right;

                green_density = lane.rightmost_cell.state.q.r
                green_speed = lane.rightmost_cell.state.u

                red_density = 1.0
                red_speed = 0.0

                if next_green:

                    rightmost_density = green_density
                    rightmost_speed = green_speed

                else:

                    rightmost_density = red_density
                    rightmost_speed = red_speed

                # differentiable [action];

                if isinstance(next_action, th.Tensor):

                    if id.loc == "west" or id.loc == "east":

                        # if WE-dir, when [action] decreases, it becomes closer to [red];

                        rightmost_density = rightmost_density + (green_density - red_density) * (next_action - next_action.detach())
                        rightmost_speed = rightmost_speed + (green_speed - red_speed) * (next_action - next_action.detach())

                    else:

                        # else, when [action] increases, it becomes closer to [red];

                        rightmost_density = rightmost_density + (green_density - red_density) * (next_action.detach() - next_action)
                        rightmost_speed = rightmost_speed + (green_speed - red_speed) * (next_action.detach() - next_action)

                lane.set_rightmost_cell(rightmost_density, rightmost_speed)


    @staticmethod
    def micro_bdry_callback(args: Dict):

        '''
        Bdry function for micro lane when using traffic signals.

        If it is green signal, it works same as default bdry function.
        Else, we assume that the next lane is blocked.

        New vehicle is generated at the chance of [incoming].
        '''

        lane: dMicroLane = args['lane']
        signal_info = args['signal_info']
        incoming = args['incoming']
        id: LaneID = args['id']

        default_bdry_callback(args)

        # left: generate new vehicle if needed;

        if len(lane.prev_lane) == 0:

            nv = lane.random_vehicle()

            enough_space = lane.entering_free_space() > nv.length * 0.5

            if enough_space:

                rand = np.random.random((1,)).item()

                if rand < incoming:

                    nv.position = 0
                    nv.speed = 0

                    lane.add_tail_vehicle(nv)

        # right;

        if len(lane.curr_vehicle) == 0:

            return

        v = lane.curr_vehicle[-1]

        (prev_green, prev_action, next_green, next_action) = signal_info

        green_position_delta = lane.head_position_delta
        green_speed_delta = lane.head_speed_delta

        red_position_delta = max((lane.length - v.position) - (v.length * 0.5), 0)
        red_speed_delta = v.speed

        if next_green:

            lane.head_position_delta = green_position_delta
            lane.head_speed_delta = green_speed_delta

        else:

            lane.head_position_delta = red_position_delta
            lane.head_speed_delta = red_speed_delta

        # differentiable [action];

        if id.loc != 'mid' and id.approaching:

            if isinstance(next_action, th.Tensor):

                if id.loc == "west" or id.loc == "east":

                    # if WE-dir, when [action] decreases, it becomes closer to [red];

                    lane.head_position_delta = lane.head_position_delta + (green_position_delta - red_position_delta) * (next_action - next_action.detach())
                    lane.head_speed_delta = lane.head_speed_delta + (green_speed_delta - red_speed_delta) * (next_action - next_action.detach())

                else:

                    # else, when [action] increases, it becomes closer to [red];

                    lane.head_position_delta = lane.head_position_delta + (green_position_delta - red_position_delta) * (next_action.detach() - next_action)
                    lane.head_speed_delta = lane.head_speed_delta + (green_speed_delta - red_speed_delta) * (next_action.detach() - next_action)
