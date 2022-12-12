import pygame
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv
    from highway_env.envs.common.abstract import Action
from highway_env.envs.common.graphics import EnvViewer, ObservationGraphics, EventHandler
from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics

import numpy as np

class IntersectionViewer(EnvViewer):

    """
    Viewer in which camera does not follow a specific ego driver.
    Instead, it is just pinned at the center of the scene.
    """

    SAVE_IMAGES = False

    SCALING_FACTOR = 1.1
    MOVING_FACTOR = 5

    camera_horizontal_pos = 0
    camera_vertical_pos = 0

    def __init__(self, env: 'AbstractEnv', config: Optional[dict] = None) -> None:
        
        super().__init__(env, config)

        if "screen_scale" in config.keys():
            screen_scale = config["screen_scale"]
        else:
            screen_scale = 0

        while screen_scale < 0:
            
            self.sim_surface.scaling *= 1 / self.SCALING_FACTOR
            screen_scale += 1
        
        while screen_scale > 0:
            
            self.sim_surface.scaling *= self.SCALING_FACTOR
            screen_scale -= 1

    def display(self, action) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)

        # render callback;

        if hasattr(self.env, 'render_callback'):

            self.env.render_callback(action)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1

    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            
        # Move camera
        held_keys = pygame.key.get_pressed()
        if held_keys[pygame.K_s]:
            self.sim_surface.scaling *= 1 / self.SCALING_FACTOR
        if held_keys[pygame.K_w]:
            self.sim_surface.scaling *= self.SCALING_FACTOR

        if held_keys[pygame.K_LEFT]:
            self.camera_horizontal_pos -= self.MOVING_FACTOR
        if held_keys[pygame.K_RIGHT]:
            self.camera_horizontal_pos += self.MOVING_FACTOR
        if held_keys[pygame.K_UP]:
            self.camera_vertical_pos -= self.MOVING_FACTOR
        if held_keys[pygame.K_DOWN]:
            self.camera_vertical_pos += self.MOVING_FACTOR

    def window_position(self) -> np.ndarray:
        """the world position of the center of the displayed window."""
        return (self.camera_horizontal_pos, self.camera_vertical_pos)