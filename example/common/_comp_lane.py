from typing import List
from math import floor
import numpy as np
import torch as th

import pygame

from highway_env.road.graphics import WorldSurface
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import AbstractLane
from highway_env.road.road import Road, RoadNetwork

from road.lane._base_lane import BaseLane
from road.lane._macro_lane import MacroLane
from road.lane._micro_lane import MicroLane

class RenderCell:
    
    '''
    Information needed to render macro cells.
    '''

    def __init__(self, start: np.ndarray, end: np.ndarray):

        self.start = start
        self.end = end

class CompLane:

    '''
    As we need two different kinds of lanes, one of which is used for rendering,
    and the other is used for simulation, we need some data structure to define
    their matching relationship. This simple structure does the job.

    @ env_lane: Lane used for rendering, we use [highway_env]'s lane here.
    @ sim_lane: Lane used for simulation, we use our [BaseLane] here.
    @ reverse: When rendering [env_lane], we bring information in [sim_lane].
    In the course of doing that, we sometimes need to render information in
    reverse, which means that we render things from [end] of the lane, not 
    from [start].
    '''

    def __init__(self, env_road: Road, env_lane: AbstractLane, sim_lane: BaseLane, reverse: bool):

        self.env_road = env_road

        self.env_lane = env_lane

        self.sim_lane = sim_lane

        self.reverse = reverse

        self.render_cell: List[RenderCell] = None

        if sim_lane.is_macro():

            self.init_render_cell()

    def render(self, surface: WorldSurface):

        if isinstance(self.sim_lane, MacroLane):

            for i, cell in enumerate(self.render_cell):

                sim_cell = self.sim_lane.curr_cell[i]

                density = sim_cell.state.q.r
                density = min(floor(density * 255), 255)
                color = [255, 255 - density, 255 - density]

                start = surface.vec2pix(cell.start)
                end = surface.vec2pix(cell.end)

                pygame.draw.line(surface, color, start, end, width=5)

        elif isinstance(self.sim_lane, MicroLane):

            for v in self.sim_lane.curr_vehicle:

                position = v.position

                if isinstance(position, th.Tensor):

                    position = position.item()

                if self.reverse:

                    position = self.sim_lane.length - position

                position = self.env_lane.position(position, 0)

                heading = self.env_lane.heading_at(position)

                rv = Vehicle(self.env_road, position, heading, v.speed)

                rv.check_collisions = False
                rv.collidable = False

                self.env_road.vehicles.append(rv)

        else:

            raise ValueError()

    def init_render_cell(self):

        assert isinstance(self.sim_lane, MacroLane), ""

        self.render_cell = []

        env_lane = self.env_lane
        sim_lane = self.sim_lane

        for cell in sim_lane.curr_cell:

            start = env_lane.position(cell.start, 0)
            end = env_lane.position(cell.end, 0)

            self.render_cell.append(RenderCell(start, end))