import numpy as np
import pygame
import math
from astronim.utils.tools import distance, Vec3, get_2d
from astronim.utils.constants import DEPTH
from astronim.objects.line_between import LineBetween
import time



class Graph: 

    def __init__(self, width_height:tuple, x_data: np.ndarray, y_data: np.ndarray,
                 pos:tuple = (0, 0, 0), points_per_second = 30, frame_color = (255, 255, 255),
                 point_color = (255, 255, 255), alpha:int = 255, color_fn: callable = None,
                 delay: float = 0.0, point_radius: int = 3,
                 instant: bool = False):

        self.width, self.height = width_height
        self.x_data = x_data
        self.y_data = y_data
        self.pos = Vec3(pos[0], pos[1], pos[2])
        self.base_size = 0.02
        self.box_animation = False
        self.frame_color = frame_color
        self.point_color = point_color
        self.alpha = alpha
        self.color_fn = color_fn
        self.current_colors = []

        self.current_x_data, self.current_y_data = ([], [])
        self.data_i = 0
        self.last_time = time.time()
        self.points_per_second = points_per_second
        self.delay = float(delay)   # seconds (frame-paced at 60 fps) before anything draws
        self.point_radius = int(point_radius)
        # instant=True: frame + all data fully drawn from the first frame
        # (no draw-on animation), and the frame follows self.pos if it is
        # moved afterwards — for graphs that slide around a scene.
        self.instant = bool(instant)
        self._frames = 0            # frame-paced clock (recording-safe)
        self._edges_at = None       # pos the edges were built for

        self.edges = []
        self.static = True


    def draw(self, screen):
        self._frames += 1
        if self._frames <= self.delay * 60.0:
            return

        dist = distance(self.pos, Graph.camera)
        scale = max(0, int(self.base_size * DEPTH / dist))
        pos_2d = get_2d(self.pos - Graph.camera, Graph.rx, Graph.ry)
        if pos_2d:
            self.rect = pygame.Rect(0, 0, self.width*scale, self.height*scale)
            self.rect.center = pos_2d
            
            if not self.edges:
                self.init_edges()
            elif self.instant and self._edges_at != (self.pos.x,
                                                     self.pos.y,
                                                     self.pos.z):
                self.init_edges()   # frame follows a moving pos

            for edge in self.edges: 
                edge.set_camera(Graph.camera, Graph.rx, Graph.ry)
                edge.draw(screen)

            self.scatter(screen)
            # pygame.draw.rect(screen, (255, 255, 255), rect, width = 2)
            # print(self.rect.left)
            # print(self.rect.right)

        

    def init_edges(self):
        hw = self.width / 2
        hh = self.height / 2

        # Get 3d coordinates for each line
        self.bl = self.pos + Vec3(-hw, -hh, 0)  # bottom-left
        self.tl = self.pos + Vec3(-hw,  hh, 0)  # top-left
        self.tr = self.pos + Vec3( hw,  hh, 0)  # top-right
        self.br = self.pos + Vec3( hw, -hh, 0)  # bottom-right

        animate = not self.instant
        self.edges = [
            LineBetween(self.bl, self.tl, animate=animate, speed=0.008, color = self.frame_color),
            LineBetween(self.tl, self.tr, animate=animate, speed=0.008, color = self.frame_color),
            LineBetween(self.tr, self.br, animate=animate, speed=0.008, color = self.frame_color),
            LineBetween(self.br, self.bl, animate=animate, speed=0.008, color = self.frame_color),
        ]
        self._edges_at = (self.pos.x, self.pos.y, self.pos.z)

    def scatter(self, screen, margin = 0.01):
         
        self.update_data_animation()
        if not hasattr(self, "bl") or not hasattr(self, "tl"):
            return  # edges not initialized yet
        
        if isinstance(self.current_x_data, np.ndarray):
            self.current_x_data = self.current_x_data.tolist()
        if isinstance(self.current_y_data, np.ndarray):
            self.current_y_data = self.current_y_data.tolist()

        if not hasattr(self, "_scatter_surface"):
            self._scatter_surface = pygame.Surface(
                screen.get_size(), pygame.SRCALPHA
            )

        self._scatter_surface.fill((0, 0, 0, 0))  

        if self.current_x_data and self.current_y_data:

            # Normalize against the FULL dataset, not just the points
            # streamed so far — otherwise the axes rescale every frame as
            # data arrives and already-plotted points slide around.
            x_min, x_max = np.min(self.x_data), np.max(self.x_data)
            y_min, y_max = np.min(self.y_data), np.max(self.y_data)

            # do not want to divide by zero
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1

            usable_width = self.width * (1 - 2*margin)
            usable_height = self.height * (1 - 2*margin)
            
            if self.color_fn is None:
                color = (*self.point_color[:3], self.alpha)

            for j, (x, y) in enumerate(zip(self.current_x_data, self.current_y_data)):
                # Normalize into [-hw, hw], [-hh, hh]
                x_norm = ((x - x_min) / x_range) * usable_width - usable_width/2
                y_norm = ((y - y_min) / y_range) * usable_height - usable_height/2

                # Get 3D position inside box
                point_3d = self.pos + Vec3(x_norm, y_norm, 0)

                # Project into 2D
                point_2d = get_2d(point_3d - Graph.camera, Graph.rx, Graph.ry)

                # if point_2d:
                #     if self.color_fn is not None:
                #         color = self.color_fn(point_2d, self.rect)

                if point_2d:
                    pygame.draw.circle(
                        self._scatter_surface,
                        self.current_colors[j] if self.color_fn else color,
                        point_2d,
                        self.point_radius
                    )
            screen.blit(self._scatter_surface, (0, 0))


    def update_data_animation(self):
        if self.instant:
            # Everything visible from frame one
            if self.data_i < len(self.x_data):
                self.current_x_data = list(self.x_data)
                self.current_y_data = list(self.y_data)
                if self.color_fn is not None:
                    self.current_colors = [self.color_fn(i)
                                           for i in range(len(self.x_data))]
                self.data_i = len(self.x_data)
            return
        # Frame-paced (one draw = 1/60 s of scene time) so recordings get
        # the same reveal speed regardless of how slow real ticks are.
        # Points hold until the frame's edge draw-on completes, then stream.
        if self.edges and any(e.progress < 1 for e in self.edges):
            self._points_started_at = self._frames
            return
        started = getattr(self, "_points_started_at", 0)
        elapsed_s = (self._frames - started) / 60.0
        target = min(len(self.x_data), int(elapsed_s * self.points_per_second))

        while self.data_i < target:
            self.current_x_data.append(self.x_data[self.data_i])
            self.current_y_data.append(self.y_data[self.data_i])

            if self.color_fn is not None:
                self.current_colors.append(
                    self.color_fn(self.data_i)
                )

            self.data_i += 1



    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry