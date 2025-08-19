import pygame 
import numpy as np

class Star:
    def __init__(self, obj_pos_2d, screen, color, obj_pos_3d):
        self.obj_pos_3d = obj_pos_3d
        self.base_radius = 0.01

        self.draw_star(obj_pos_2d, color, screen)

    def draw_glow_circle(self, surface, color, center, radius, glow_radius):
        
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)

        # multiple circles with decreasing alpha for glow
        for i in np.arange(glow_radius, radius, -1):
            t =  (i - radius) / (glow_radius - radius)
            alpha = int(255 * np.exp(-4 * t))
            pygame.draw.circle(glow_surf, (*color, alpha), (glow_radius, glow_radius), i)


        glow_rect = glow_surf.get_rect(center=center)
        surface.blit(glow_surf, glow_rect, special_flags = pygame.BLEND_ALPHA_SDL2 )

        pygame.draw.circle(surface, color, center, radius)


    def draw_star(self, obj_pos_2d, color, screen):

        dx = self.obj_pos_3d.x - Star.camera.x
        dy = self.obj_pos_3d.y - Star.camera.y
        dz = self.obj_pos_3d.z - Star.camera.z
        dist = (dx**2 + dy**2 + dz**2)**0.5 + 1

        self.radius = max(2, min(20, int(self.base_radius * 500 / dist)))



        self.draw_glow_circle(screen, color, obj_pos_2d, radius=self.radius, glow_radius=self.radius *8)
        pygame.draw.circle(screen, color, obj_pos_2d, 2)


    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry

