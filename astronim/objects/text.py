import pygame
from astronim.utils.tools import distance, get_2d, Vec3
from astronim.utils.constants import DEPTH, WIDTH, HEIGHT
import time

class Text: 
    def __init__(self, message: str, pos: Vec3 =  Vec3(0, 0, 50), base_size: int = 32, color = (255, 255, 255), type_out = False, start_time = 0, char_delay = 50): 
        self.pos = pos
        self.message = message
        self.base_size = base_size
        self.color = color

        self.type_out = type_out
        self.current_message = ""
        self.char_delay = char_delay
        self.start_time = start_time

    def draw(self, screen): 
        dist = distance(self.pos, Text.camera)
        size = max(2, min(5000, int(self.base_size * DEPTH / dist)))
        font = pygame.font.SysFont('Times New Roman', size)

        if self.type_out: 
            self.update_message()
            text_surface = font.render(self.current_message, True, self.color)
        else:
            text_surface = font.render(self.message, True, self.color)

        textRect = text_surface.get_rect()
        # textRect.center = (WIDTH //2, HEIGHT//2)
        textRect.center = (3840 //2, 2160//2)
        #Change width and heigth back!!!!

        text_pos = get_2d(self.pos - Text.camera, Text.rx, Text.ry)

        if text_pos: 
            textRect.center = text_pos
            screen.blit(text_surface, textRect)

    def update_message(self):

        elapsed = pygame.time.get_ticks() - self.start_time
        

        chars_to_show = elapsed // self.char_delay

        self.current_message = self.message[:chars_to_show]

        if chars_to_show >= len(self.message): 
            self.type_out = False
        


    def attach_to(self): 
        """
        Attaches the text object to another object. 
        """
        pass




    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry