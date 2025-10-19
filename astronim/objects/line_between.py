import pygame 
import numpy as np
from astronim.utils.tools import get_2d, Vec3, distance
from astronim.utils.constants import DEPTH

class LineBetween: 
    def __init__(self, obj1, obj2, color = (255, 255, 255), width: int= 2, animate: bool = False, speed: float = 0.01):

        self.obj1 = obj1
        self.obj2 = obj2

        self.color = color
        self.width = width
        self.animate = animate
        self.speed = speed
        self.progress = 0 if self.animate else 1

    def draw(self, screen): 

        start, end = self.init_start_end()

        if start and end: 

            if self.progress < 1: 
                self.progress = min(1, self.progress + self.speed)

            start_x, start_y = start
            end_x, end_y = end

            line_x = start_x + (end_x - start_x) * self.progress
            line_y = start_y + (end_y - start_y) * self.progress

            pygame.draw.line(screen, self.color, start, (line_x, line_y), width = self.width )

    def init_start_end(self): 
        '''
        Checks valid types for start amd end variables. Line should accept a tuple, an object, or a vec3
        '''

        if isinstance(self.obj1, tuple): 
            start = get_2d(Vec3(self.obj1[0], self.obj1[1], self.obj1[2]) - LineBetween.camera, LineBetween.rx, LineBetween.ry)
        elif isinstance(self.obj1, Vec3): 
            start = get_2d(self.obj1 - LineBetween.camera, LineBetween.rx, LineBetween.ry)
        else: 
            start = get_2d(self.obj1.pos - LineBetween.camera, LineBetween.rx, LineBetween.ry)

            
        if isinstance(self.obj2, tuple): 
            end = get_2d(Vec3(self.obj2[0], self.obj2[1], self.obj2[2]) - LineBetween.camera, LineBetween.rx, LineBetween.ry)
        elif isinstance(self.obj2, Vec3): 
            end = get_2d(self.obj2 - LineBetween.camera, LineBetween.rx, LineBetween.ry)
        else: 
            end = get_2d(self.obj2.pos - LineBetween.camera, LineBetween.rx, LineBetween.ry)

        return (start, end)







        


    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry