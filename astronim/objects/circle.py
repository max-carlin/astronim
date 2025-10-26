import pygame
from astronim.utils.tools import distance, get_2d, Vec3
from astronim.utils.constants import DEPTH, WIDTH, HEIGHT


class Circle: 
    def __init__(self, center, radius: int, color = (255, 255, 255)):

        # Can circle on objects, so check the type here
        if isinstance(center, Vec3):
            self.center = center
        else: 
            self.center = center.pos

        self.base_radius = radius
        self.color = color
        self.static = True

    def draw(self, screen): 
        pos = get_2d(self.center - Circle.camera, Circle.rx, Circle.ry)
        dist = distance(self.center, Circle.camera)
        radius = max(2, min(5000, int(self.base_radius * DEPTH / dist )))
        
        if pos is None: 
            return

        pygame.draw.circle(screen, self.color, pos, radius, width = 1)

    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry