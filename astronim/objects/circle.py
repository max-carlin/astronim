import math
import pygame
from astronim.utils.tools import distance, get_2d, Vec3
from astronim.utils import constants
from typing import Callable, Optional

class Circle:
    def __init__(self, center, radius: int, color = (255, 255, 255), width:int = 1, animation : Optional[Callable] = None, animate: bool = False, speed: float = 0.02, delay: float = 0.0):

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

        # Draw-on reveal (LineBetween-style): after `delay` seconds
        # (frame-paced at 60 fps) the ring sweeps closed clockwise from
        # the top, advancing `speed` per frame.
        self.animate_draw = animate
        self.speed = speed
        self.delay = float(delay)
        self.progress = 0.0 if animate else 1.0
        self._frames = 0


    def draw(self, screen):
        if self.animate_draw:
            self._frames += 1
            if self._frames <= self.delay * 60.0:
                return
            if self.progress < 1:
                self.progress = min(1.0, self.progress + self.speed)

        if self.animate_circle:
            self.animate(self.animation)
            self.pos = self.center

        pos = get_2d(self.center - Circle.camera, Circle.rx, Circle.ry)
        dist = distance(self.center, Circle.camera)
        radius = max(2, min(5000, int(self.base_radius * constants.DEPTH / dist )))

        if pos is None:
            return

        if self.progress < 1.0:
            # Partial ring: arc from (top - progress*2pi) up to the top is
            # the set swept clockwise starting at 12 o'clock
            rect = pygame.Rect(pos[0] - radius, pos[1] - radius,
                               radius * 2, radius * 2)
            pygame.draw.arc(screen, self.color, rect,
                            math.pi / 2 - 2.0 * math.pi * self.progress,
                            math.pi / 2, self.width)
        else:
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