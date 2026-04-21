import os
import runpy
import traceback

import pygame
from astronim.simulation import Simulation
from astronim.renderer import Renderer
from astronim.recorder import Recorder
from astronim.utils.tools import Vec3
import numpy as np


class Universe:
    """Handles the core loop of astronim. 

        -Initializes pygame and the rendering window. 
        -Manages the simulation and updates bodies. 
        -Runs the renderer (camera, drawing, depth sorting)
        -Records frames to an output .mp4 file using recorder. 
        -Handles user input to move around scene. 

        Attributes
        ----------
        screen : pygame.Surface
            The pygame screen that the simulation will draw on. 

        simulation : Simulation
            Contains all the objects in the scene and update logic

        renderer : Renderer
            Contains draw order and handles drawing logic. 

        recorder : Recorder
            Captures screen frame-by-frame and uses ffmpeg to save video to a specified output file. 

        running : bool
            Main loop will run while true. 

        clock : pygame.time.Clock
            Controls frame timing

        speed : float
            Controls fly control speeds. 

        shift_speed : float
            Controls fly control speeds while holding down right shift. 

        Methods
        -------
        handle_events(): 
            Processes pygame events. 

        main_loop(): 
            Runs the simulation, rendering, and recording. Must be called in any project file. 

        controls():
            Handles all the controls for moving through the scene (WASD, space, ctrl, shift)

    """
    def __init__(self, width: int = 1920, height: int = 1080,
                 output_file: str =  "output", on_tab: callable = None):


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

        self.output_file = output_file 
        self.static_mouse = False

        self.on_tab = on_tab
        

        
    def handle_events(self):
        '''
        Handles all of the pygame events, quits running on window close. 
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.pause_sim()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                if self.on_tab is not None:
                    self.on_tab()



    def main_loop(self, watch=None):
        '''
        The main loop that updates our simulation, draws to the screen, and records the scene.

        If ``watch`` is a path to a script defining ``build(u)``, the loop polls its
        mtime and re-executes ``build(u)`` in place whenever the file changes, so
        edits take effect without closing the pygame window.
        '''
        watch_mtime = self._watch_mtime(watch) if watch else None
        poll_every = 10
        frame = 0

        while self.running:
            self.dt = 0.1 * 86400 # seconds per frame
            # self.clock.tick(60) # seconds per frame
            self.handle_events()

            if watch and frame % poll_every == 0:
                current = self._watch_mtime(watch)
                if current is not None and current != watch_mtime:
                    watch_mtime = current
                    self._reload_scene(watch)

            keys = pygame.key.get_pressed()
            self.controls(keys)

            frame += 1


            # if not self.static_mouse:
            #     dx, dy = pygame.mouse.get_rel()
            #     self.renderer.rx += np.radians(dx / 5)
            #     self.renderer.ry -= np.radians(dy / 5)

            rotation_speed = np.radians(1)  # degrees per frame, adjust for sensitivity

            if keys[pygame.K_q]:
                self.renderer.rx -= rotation_speed
            if keys[pygame.K_e]:
                self.renderer.rx += rotation_speed
            if keys[pygame.K_p]: 
                self.renderer.ry += rotation_speed
            if keys[pygame.K_l]: 
                self.renderer.ry -= rotation_speed

            self.simulation.update(self.dt)
            self.renderer.draw(self.simulation)
            self.recorder.save_frame(self.screen)

        pygame.quit()
        if self.output_file[-3:] == '.mp4':
            self.recorder.stop(output_file=self.output_file)
        else:
            self.recorder.stop(output_file=self.output_file + '.mp4')
        
        

    
    def controls(self, keys):
        '''
        Handles all of the pygame controls to move through the scene. 
        '''
        

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
        if keys[pygame.K_LSHIFT]:
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

    # def pause_sim(self):
    #     for obj in self.simulation.star_objects:
    #         obj.static = True
    def pause_sim(self):
        self.paused = not getattr(self, "paused", False)
        for obj in self.simulation.star_objects:
            obj.static = self.paused

    @staticmethod
    def _watch_mtime(path):
        try:
            return os.path.getmtime(path)
        except (FileNotFoundError, OSError):
            return None

    def _reload_scene(self, watch_path):
        '''Re-execute the watched script and rebuild the scene in place.'''
        cam = self.renderer.camera
        rx, ry = self.renderer.rx, self.renderer.ry
        try:
            ns = runpy.run_path(watch_path, run_name="__reload__")
        except Exception:
            print(f"[astronim] reload failed to execute {watch_path}:")
            traceback.print_exc()
            return

        build = ns.get("build")
        if not callable(build):
            print(f"[astronim] {watch_path} defines no build(u); keeping current scene")
            return

        self.simulation.clear()
        try:
            build(self)
        except Exception:
            print(f"[astronim] build(u) raised during reload:")
            traceback.print_exc()
        finally:
            self.renderer.camera = cam
            self.renderer.rx = rx
            self.renderer.ry = ry

        




