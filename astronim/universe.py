import pygame
from astronim.simulation import Simulation
from astronim.renderer import Renderer
from astronim.recorder import Recorder
from astronim.utils.tools import Vec3 
import numpy as np


class Universe:
    def __init__(self, width=1920, height=1080, output_file = "output"):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Astronim')

        self.simulation = Simulation()
        self.renderer = Renderer(self.screen, width, height)
        self.recorder = Recorder()

        self.running = True
        self.clock = pygame.time.Clock()

        self.speed = 0.2
        self.shift_speed_factor = 10

        self.output_file = output_file + '.mp4'

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False



    def main_loop(self):
        while self.running:
            dt = 0.1 * 86400 # seconds per frame
            self.clock.tick(120) # seconds per frame
            self.handle_events()


            keys = pygame.key.get_pressed()
            self.controls(keys)

            dx, dy = pygame.mouse.get_rel()
            self.renderer.rx += np.radians(dx / 5)
            self.renderer.ry -= np.radians(dy / 5)

            self.simulation.update(dt)
            self.renderer.draw(self.simulation)
            self.recorder.save_frame(self.screen)

        pygame.quit()
        self.recorder.stop(output_file=self.output_file)

    

    def controls(self, keys):

        if keys[pygame.K_w]:
            self.renderer.camera.x += np.sin(self.renderer.rx) * self.speed 
            self.renderer.camera.z += np.cos(self.renderer.rx) * self.speed 
        if keys[pygame.K_s]:
            self.renderer.camera.x -= np.sin(self.renderer.rx) * self.speed 
            self.renderer.camera.z -= np.cos(self.renderer.rx) * self.speed 
        if keys[pygame.K_a]:
            self.renderer.camera.x += np.sin(self.renderer.rx - np.pi / 2) * self.speed 
            self.renderer.camera.z += np.cos(self.renderer.rx - np.pi / 2) * self.speed 
        if keys[pygame.K_d]:
            self.renderer.camera.x += np.sin(self.renderer.rx + np.pi / 2) * self.speed 
            self.renderer.camera.z += np.cos(self.renderer.rx + np.pi / 2) * self.speed 
        if keys[pygame.K_SPACE]:
            self.renderer.camera.y += self.speed 
        if keys[pygame.K_LCTRL]:
            self.renderer.camera.y -= self.speed 


        if keys[pygame.K_w] and keys[pygame.K_RSHIFT]:
            self.renderer.camera.x += np.sin(self.renderer.rx) * self.speed * self.shift_speed_factor
            self.renderer.camera.z += np.cos(self.renderer.rx) * self.speed * self.shift_speed_factor
        if keys[pygame.K_s] and keys[pygame.K_RSHIFT]:
            self.renderer.camera.x -= np.sin(self.renderer.rx) * self.speed * self.shift_speed_factor
            self.renderer.camera.z -= np.cos(self.renderer.rx) * self.speed * self.shift_speed_factor
        if keys[pygame.K_a] and keys[pygame.K_RSHIFT]:
            self.renderer.camera.x += np.sin(self.renderer.rx - np.pi / 2) * self.speed * self.shift_speed_factor
            self.renderer.camera.z += np.cos(self.renderer.rx - np.pi / 2) * self.speed * self.shift_speed_factor
        if keys[pygame.K_d] and keys[pygame.K_RSHIFT]:
            self.renderer.camera.x += np.sin(self.renderer.rx + np.pi / 2) * self.speed * self.shift_speed_factor
            self.renderer.camera.z += np.cos(self.renderer.rx + np.pi / 2) * self.speed * self.shift_speed_factor
