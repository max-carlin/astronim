import pygame 
import numpy as np
import random
from astronim.utils.tools import get_2d, Vec3, distance
from astronim.utils.constants import DEPTH


class BlackHole: 
    def __init__(self, pos:Vec3, vel: Vec3, mass: float, radius: float = 0.01, color = (255, 255, 255), trail = False):
 
        self.velocity = vel
        self.pos = pos
        self.mass = mass
        self.base_radius = radius
        self.color = color

        self.trail = trail
        self.trail_list = []
        self.trail_length = 50
       


    def draw(self, screen): 

        dist = distance(self.pos, BlackHole.camera)

        obj_pos_2d = get_2d(self.pos, BlackHole.rx, BlackHole.ry)

        # scale radius with depth
        self.radius = max(2, int(self.base_radius * DEPTH / dist))

        self.draw_star(obj_pos_2d, (250, 226, 67), screen, radius=self.radius, glow_radius=self.radius )
        self.draw_star(obj_pos_2d, (255, 183, 0), screen, radius=self.radius, glow_radius=self.radius *2)

        pygame.draw.circle(screen, (0, 0, 0), obj_pos_2d, int(0.9*self.radius))


    def draw_glow_circle(self, surface, color, center, radius, glow_radius, width):
        
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)

        # multiple circles with decreasing alpha for glow
        for i in np.arange(glow_radius, radius, -1):
            t =  (i - radius) / (glow_radius - radius)
            alpha = int(255 * np.exp(-4 * t))
            pygame.draw.circle(glow_surf, (*color, alpha), (glow_radius, glow_radius), i, width=width)


        glow_rect = glow_surf.get_rect(center=center)
        surface.blit(glow_surf, glow_rect, special_flags = pygame.BLEND_ALPHA_SDL2 )

        pygame.draw.circle(surface, color, center, radius, width=width)


    def draw_star(self, obj_pos_2d, color, screen, width = 0, radius = 2, glow_radius = 20):
        self.draw_glow_circle(screen, color, obj_pos_2d, radius=radius, glow_radius=glow_radius, width = width)

    def draw_trail(self, screen): 
        obj_path_2d = [
                pt for pt in (get_2d(Vec3(*p) - BlackHole.camera, BlackHole.rx, BlackHole.ry) for p in self.trail_list) if pt ]
        
        if len(obj_path_2d) >1:
            pygame.draw.lines(screen, (255, 255, 255), False, obj_path_2d, 1)


    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry


    





















# class BlackHole: 
#     def __init__(self, BH_pos, screen, color, radius = 50):
#         self.BH_pos = BH_pos
#         self.base_radius = radius
#         self.front_disk = []
#         self.back_disk = []
#         self.half_disk_random_factors()
#         self.screen = screen

        

#     def update(self, camera_pos): 
#         bh_2d = get_2d(self.BH_pos - camera)
   
#         #update according to depth somehow
#         if bh_2d: 

#             dx = self.BH_pos.x - camera_pos.x
#             dy = self.BH_pos.y - camera_pos.y
#             dz = self.BH_pos.z - camera_pos.z

#             self.dist = (dx**2 + dy**2 + dz**2)**0.5
            
#             self.radius = int(self.base_radius *DEPTH / self.dist)
    


#             if self.radius > 0:
                
#                 self.draw_half_disk(self.front_angles, self.front_r_factors)#disk
#                 # pygame.draw.circle(screen, (255, 255,255), bh_2d, self.radius, width=2)
#                 # pygame.draw.circle(screen, (0, 0, 0), bh_2d, self.radius - 0.1)

#                 self.draw_star(bh_2d, (252, 250, 210), self.screen, radius=self.radius, glow_radius=100)
#                 pygame.draw.circle(self.screen, (0, 0, 0), bh_2d, self.radius - 0.1*self.radius)


#                 self.draw_half_disk(self.back_angles, self.back_r_factors) #disk


#     def draw_half_disk(self, angles, r_factors): 


#         for angle, r_factor in zip(angles, r_factors):

#             r = self.radius + r_factor * 2* DEPTH / self.dist

#             x = r * np.cos(angle) + self.BH_pos.x
#             z = r * np.sin(angle) + self.BH_pos.z
#             y = 0 + self.BH_pos.y #random.uniform(-2, 2)  # tiny vertical jitter

#             p = get_2d(Vec3(x, y, z) - camera)
#             if p: 
#                 # pygame.draw.circle(self.screen,  (200, 100, 50), p, 2)
#                 self.draw_star(p, self.get_color(r_factor), self.screen, glow_radius=20)

#     def get_color(self, r_factor): 
#         # scale radius contribution by thickness, not full DEPTH
#         r = self.radius + r_factor * (self.thickness / 2)
#         min_r = self.radius - self.thickness / 2
#         max_r = self.radius + self.thickness / 2
        
#         # normalize to 0..1
#         t = (r - min_r) / (max_r - min_r)
        
#         if t >= 0.9: 
#             return (252, 196, 63)
#         if t >= 0.8: 
#             return (255, 208, 0)    
#         if t >= 0.6: 
#             return (252, 226, 109)  
#         if t >= 0.5: 
#             return (250, 244, 160)  


#         return (252, 250, 210)  
        

#     def half_disk_random_factors(self, n_points=1000, radius=50, thickness=15):

#         # all of the points in the disk are generated at a random
#         # angle and radius + random r_factor. We only want to generate those once. 
#         # We generate the front of the accretion disk and back separatley so we can get depth right.

#         self.front_angles = []
#         self.front_r_factors = []
#         self.thickness = thickness


#         for _ in range(int(n_points / 2)):
#             angle = random.uniform(0, np.pi)
#             r_factor = random.uniform(-thickness/2, thickness/2)
#             self.front_angles.append(angle)
#             self.front_r_factors.append(r_factor)

#         self.back_angles = []
#         self.back_r_factors = []

#         for _ in range(int(n_points / 2)):
#             angle = random.uniform(np.pi, 2*np.pi)
#             r_factor = random.uniform(-thickness/2, thickness/2)
#             self.back_angles.append(angle)
#             self.back_r_factors.append(r_factor)



#     def draw_glow_circle(self, surface, color, center, radius, glow_radius, width):
        
#         glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)

#         # multiple circles with decreasing alpha for glow
#         for i in np.arange(glow_radius, radius, -1):
#             t =  (i - radius) / (glow_radius - radius)
#             alpha = int(255 * np.exp(-4 * t))
#             pygame.draw.circle(glow_surf, (*color, alpha), (glow_radius, glow_radius), i, width=width)


#         glow_rect = glow_surf.get_rect(center=center)
#         surface.blit(glow_surf, glow_rect, special_flags = pygame.BLEND_ALPHA_SDL2 )

#         pygame.draw.circle(surface, color, center, radius, width=width)


#     def draw_star(self, obj_pos_2d, color, screen, width = 0, radius = 2, glow_radius = 20):
#         self.draw_glow_circle(screen, color, obj_pos_2d, radius=radius, glow_radius=glow_radius, width = width)

        

