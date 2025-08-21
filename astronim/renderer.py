import pygame
from astronim.utils.tools import get_2d, Vec3, distance
import numpy as np
class Renderer: 
    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width 
        self.height = height 
        self.camera = Vec3(0, 0, 0)
        self.rx = 0
        self.ry = 0

    def draw(self, simulation): 
        self.screen.fill((0, 0, 0))

        order = []
        for obj in simulation.star_objects: 
            dist = distance(obj.pos, self.camera)
            order.append(dist)

        stars_sorted = np.array(simulation.star_objects)[np.argsort(order)].tolist()

        for obj in stars_sorted: 
            obj.set_camera(self.camera, self.rx, self.ry)
            obj.draw(self.screen)

            if obj.trail: 
                obj.draw_trail(self.screen)

        for obj in simulation.static_objects: 
            obj.set_camera(self.camera, self.rx, self.ry)
            obj.draw(self.screen)

        pygame.display.flip()

        