import pygame 
import numpy as np
from astronim.utils.tools import distance, get_2d, Vec3

class Star:
    def __init__(self, pos:Vec3, vel: Vec3, mass: float, radius: float = 0.01, color = (255, 255, 255), trail = False):
        self.pos = pos
        self.velocity = [vel.x, vel.y, vel.z]
        self.mass = mass
        self.base_radius = radius
        self.color = color
        self.trail_list = []
        self.trail_length = 50
        self.trail = trail

        

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




    def draw(self, screen):

        

        obj_pos_2d = get_2d(self.pos - Star.camera, Star.rx, Star.ry)
        if obj_pos_2d:
            dist = distance(self.pos, Star.camera)
            self.radius = max(2, min(20, int(self.base_radius * 500 / dist)))

            self.draw_glow_circle(
                screen, self.color, obj_pos_2d,
                radius=self.radius,
                glow_radius=self.radius * 8
            )
            pygame.draw.circle(screen, self.color, obj_pos_2d, 0.01)

    def draw_trail(self, screen): 
        obj_path_2d = [
                pt for pt in (get_2d(Vec3(*p) - Star.camera, Star.rx, Star.ry) for p in self.trail_list) if pt ]
        
        if len(obj_path_2d) >1:
            pygame.draw.lines(screen, (255, 255, 255), False, obj_path_2d, 1)



    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry

