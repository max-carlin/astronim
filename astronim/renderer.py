import pygame
from astronim.utils.tools import get_2d, Vec3, distance
import numpy as np
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

    def draw(self, simulation): 
        '''Render all objects in the given simulation to the screen.

        params
        ------
        simulation : Simulation
            The current simulation containing star_objects and static_objects.
        '''
        self.screen.fill((0, 0, 0))

        order = []
        for obj in simulation.star_objects: 
            dist = distance(obj.pos, self.camera)
            order.append(dist)

        stars_sorted = np.array(simulation.star_objects)[np.argsort(order)[::-1]].tolist()

        for obj in stars_sorted: 
            obj.set_camera(self.camera, self.rx, self.ry)
            obj.draw(self.screen)

            if obj.trail: 
                obj.draw_trail(self.screen)

        for obj in simulation.static_objects: 
            obj.set_camera(self.camera, self.rx, self.ry)
            obj.draw(self.screen)

        if self.camera_movement_called: 
            self.camera_function(self.camera)


        pygame.display.flip()

    def camera_animation(self, camera_function):
        self.camera_movement_called = True 
        self.camera_function = camera_function


        