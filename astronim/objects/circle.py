import pygame
from astronim.utils.tools import distance, get_2d, Vec3
from astronim.utils.constants import DEPTH, WIDTH, HEIGHT
from typing import Callable, Optional

class Circle: 
    def __init__(self, center, radius: int, color = (255, 255, 255), width:int = 1, animation : Optional[Callable] = None):

        # Can circle on objects, so check the type here
        if isinstance(center, Vec3):
            self.center = center
            self.pos = center
        else: 
            self.center = center.pos
            self.pos = center.pos

        
        self.base_radius = radius
        self.color = color
        self.static = True
        self.animate_circle = False
        self.animation = animation
        self.width = width

        if animation is not None: 
            self.animate_circle = True


    def draw(self, screen):
        if self.animate_circle: 
            self.animate(self.animation)
            self.pos = self.center
             
        pos = get_2d(self.center - Circle.camera, Circle.rx, Circle.ry)
        dist = distance(self.center, Circle.camera)
        radius = max(2, min(5000, int(self.base_radius * DEPTH / dist )))
        
        if pos is None: 
            return

        pygame.draw.circle(screen, self.color, pos, radius, width = self.width)

    def animate(self, func):
        '''Allows for animations or movements using the center of the circle. 

        Parameters
        ----------
        func : callable
            The function to apply to the center of the circle. It should take
            the current center of the circle as an argument and return the updated center. 
            
        ''' 
        updated = func(self.center)
        if updated is not None: 
            self.center = updated

    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry