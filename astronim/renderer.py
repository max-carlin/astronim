import pygame
from astronim.utils.tools import get_2d, Vec3, distance
from astronim.objects import *
import numpy as np
import math
import os
class Renderer:
    '''Handles the rendering for a scene. 

    -Sorts every object by distance. 
    -Draws each object according to their distance away. 
    -Passes in camera and mouse information to each object.

    Attributes
    ----------
    screen : pygame.Surface
        The screen to draw on, passed in by Universe. 
    width : int
        The width of the screen. 
    height : int
        The height of the screen. 
    camera : Vec3
        The camera position in 3d space. 
    rx : float
        Rotation angle of the camera around the x-axis (radians).
    ry : float
        Rotation angle of the camera around the y-axis (radians).

    Methods
    -------
    draw(simulation): 
        Clears the screen, draws all objects in the simulation according to distance from the camera. 

    
    ''' 
    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width 
        self.height = height 
        self.camera = Vec3(0, 0, 0)
        self.rx = 0
        self.ry = 0
        self.camera_movement_called = False
        self.trail_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        self.csv_buffer = []
        self.csv_flush_interval = 300   # write every 300 frames (5 seconds at 60 FPS)
        self.csv_frame_counter = 0
        self.bg_color = (0, 0, 0)

        # if not os.path.exists("position.csv"):
        #     with open("position.csv", "w") as f:
        #         f.write("x,y,z\n")



    def draw(self, simulation): 
        '''Render all objects in the given simulation to the screen.

        params
        ------
        simulation : Simulation
            The current simulation containing star_objects and static_objects.
        '''
        self.screen.fill(self.bg_color)
        self.trail_surface.fill((0, 0, 0, 0))
        #compute min/max speeds this frame
        min_s = float("inf")
        max_s = 0.0

        for obj in simulation.star_objects:
            vx, vy, vz = obj.velocity
            s = vx*vx + vy*vy + vz*vz  # squared speed
            # self.csv_buffer.append((obj.pos.x, obj.pos.y, obj.pos.z))

            if s < min_s:
                min_s = s
            if s > max_s:
                max_s = s
        # convert squared speeds → actual speeds
        speed_min = math.sqrt(min_s) if min_s < float("inf") else 0.0
        speed_max = math.sqrt(max_s) if max_s > 0 else 1e-9

        self.speed_min = speed_min
        self.speed_max = speed_max
        order = []


        # Unified back-to-front depth sort over both dynamic and static objects.
        # Objects without `pos` (defensive — none in the codebase today) sort
        # to the back via float('inf') so they render first.
        all_objects = simulation.star_objects + simulation.static_objects
        distances = []
        for obj in all_objects:
            pos = getattr(obj, 'pos', None)
            distances.append(
                distance(pos, self.camera) if pos is not None else float('inf')
            )
            if isinstance(obj, BlackHole):
                self.csv_buffer.append((obj.pos.x, obj.pos.y, obj.pos.z))

        # `kind='stable'` preserves insertion order for ties — important so a
        # user-ordered (background, foreground) pair renders predictably.
        order = np.argsort(distances, kind='stable')[::-1]

        # BH used by the existing star-behind-BH occlusion check
        black_holes = [o for o in simulation.star_objects if isinstance(o, BlackHole)]
        bh = black_holes[0] if black_holes else None

        for idx in order:
            obj = all_objects[idx]
            obj.set_camera(self.camera, self.rx, self.ry)

            if bh is not None and isinstance(obj, Star):
                if bh.screen_occludes(obj.pos):
                    continue  # STAR IS HIDDEN

            obj.draw(self.screen)

            if getattr(obj, 'trail', False):
                obj.draw_trail_color(self.trail_surface, self.speed_max, self.speed_min)

        # Trails composited last, on top of every depth-sorted object.
        self.screen.blit(self.trail_surface, (0, 0))

    
        if self.camera_movement_called:

            self.camera_function(self.camera)

        # self.csv_frame_counter += 1

        # if self.csv_frame_counter >= self.csv_flush_interval:
        #     with open("position.csv", "a") as f:
        #         for row in self.csv_buffer:
        #             f.write(f"{row[0]},{row[1]},{row[2]}\n")

        #     self.csv_buffer.clear()
        #     self.csv_frame_counter = 0

    # def draw(self, simulation):

    #     # ---- Clear main screen ----
    #     self.screen.fill((0, 0, 0))

    #     # ---- Clear trail layer (alpha) ----
    #     self.trail_surface.fill((0, 0, 0, 0))

    #     # compute min/max speeds this frame
    #     min_s = float("inf")
    #     max_s = 0.0

    #     for obj in simulation.star_objects:
    #         vx, vy, vz = obj.velocity
    #         s = vx*vx + vy*vy + vz*vz  # squared speed
    #         # self.csv_buffer.append((vx, vy, vz, math.sqrt(s)))

    #         if s < min_s:
    #             min_s = s
    #         if s > max_s:
    #             max_s = s

    #     # convert squared speeds → actual speeds
    #     speed_min = math.sqrt(min_s) if min_s < float("inf") else 0.0
    #     speed_max = math.sqrt(max_s) if max_s > 0 else 1e-9

    #     self.speed_min = speed_min
    #     self.speed_max = speed_max



    #     # ---- Depth-sort stars ----
    #     order = []
    #     for obj in simulation.star_objects:
    #         dist = distance(obj.pos, self.camera)
    #         order.append(dist)
    #     stars_sorted = np.array(simulation.star_objects)[np.argsort(order)[::-1]].tolist()

    #     # ---- Draw stars and trails ----
    #     for obj in stars_sorted:
    #         obj.set_camera(self.camera, self.rx, self.ry)

    #         # draw star normally
    #         obj.draw(self.screen)

    #         # draw trail ONTO trail_surface (NOT screen!)
    #         if obj.trail:
    #             obj.draw_trail(self.trail_surface, self.speed_max, self.speed_min)

    #     # ---- Draw static objects ----
    #     for obj in simulation.static_objects:
    #         obj.set_camera(self.camera, self.rx, self.ry)
    #         obj.draw(self.screen)

    #     # ---- Blit trail layer on top ----
    #     self.screen.blit(self.trail_surface, (0, 0))

    #     # ---- Camera animation ----
    #     if self.camera_movement_called:
    #         self.camera_function(self.camera)

    #     pygame.display.flip()

    #     # self.csv_frame_counter += 1

    #     # if self.csv_frame_counter >= self.csv_flush_interval:
    #     #     with open("velocities.csv", "a") as f:
    #     #         for row in self.csv_buffer:
    #     #             f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

    #     #     self.csv_buffer.clear()
    #     #     self.csv_frame_counter = 0

    def camera_animation(self, camera_function):
        self.camera_movement_called = True
        self.camera_function = camera_function

    def clear_camera_animation(self):
        '''Deregister the per-frame camera callback. Called by run_scenes
        between scenes so a camera animation registered by one scene can't
        leak into the next.'''
        self.camera_movement_called = False
        self.camera_function = None


        