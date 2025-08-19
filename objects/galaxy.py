import numpy as np
import pygame
import math
from utils.tools import gaussianRandom, clamp, spiral, Vec3, get_2d
import random

NUM_STARS = 500
NUM_ARMS = 4

#std in z
GALAXY_THICKNESS = 5

#std of the core
CORE_X_DIST = 15 #33
CORE_Y_DIST = 15 #33

OUTER_CORE_X_DIST = 50 #100
OUTER_CORE_Y_DIST = 50 #100

ARM_X_DIST = 50 #100
ARM_Y_DIST = 25#50
ARM_X_MEAN = 100 #200
ARM_Y_MEAN = 50 #100

SPIRAL = 1.5 #3.0
ARMS = 2.0

HAZE_RATIO = 0.5

starTypes = {
    "percentage" : [76.45, 12.1, 7.6, 3.0, 0.6, 0.13],
    "colors" : [(255, 205, 111), (255, 210, 161), (255, 244, 234), (248, 247, 255), (202, 215, 255), (170, 191, 255)]
}


class Galaxy: 
    def __init__(self, galaxy_center_3d, screen, color, camera, rx, ry, galaxy_type):
        # init function should draw the galaxy using only these args 
        self.galaxy_type = galaxy_type
        self.screen = screen
        self.color = color
        self.galaxy_center_3d = Vec3(galaxy_center_3d[0], galaxy_center_3d[1], galaxy_center_3d[2])

        types = {'spiral_galaxy': self.spiral_positions(), 'irregular_galaxy':self.irregular_positions(), "elliptical_galaxy":self.elliptical_positions()}
        self.positions = types[galaxy_type]

        self.camera = camera
        self.rx = rx
        self.ry = ry

    def update(self, camera, rx, ry):
        # just re-draw based on camera position/rotation
        self.draw(self.galaxy_center_3d, camera, rx, ry)

    def spiral_positions(self): 
        stars = [] #pos and color of star in the form [[vec, color], [vec, color], ...]

        for i in range(NUM_STARS // 4):
            x = gaussianRandom(0, CORE_X_DIST)
            y = gaussianRandom(0, CORE_Y_DIST)
            z = gaussianRandom(0, GALAXY_THICKNESS)
            stars.append([Vec3(x, y, z), self.get_star_color()])

        for i in range(NUM_STARS // 4):
            x = gaussianRandom(0, OUTER_CORE_X_DIST)
            y = gaussianRandom(0, OUTER_CORE_Y_DIST)
            z = gaussianRandom(0, GALAXY_THICKNESS)
            stars.append([Vec3(x, y, z), self.get_star_color()])

        for arm in range(int(ARMS)): 
            offset = arm *2 * math.pi / ARMS

            for i in range(NUM_STARS // 4):
                pos = spiral(
                    gaussianRandom(ARM_X_MEAN, ARM_X_DIST), 
                    gaussianRandom(ARM_Y_MEAN, ARM_Y_DIST), 
                    gaussianRandom(0, GALAXY_THICKNESS), 
                    offset
                )
                stars.append([pos, self.get_star_color()])

        self.stars = stars
        return self.stars
    
    def irregular_positions(self): 
        stars = [] #pos and color of star in the form [[vec, color], [vec, color], ...]

        for i in range(NUM_STARS // 4):
            x = gaussianRandom(0, CORE_X_DIST)
            y = gaussianRandom(0, CORE_Y_DIST)
            z = gaussianRandom(0, GALAXY_THICKNESS)
            stars.append([Vec3(x, y, z), self.get_star_color()])

        for i in range(NUM_STARS // 4):
            x = gaussianRandom(0, OUTER_CORE_X_DIST)
            y = gaussianRandom(0, OUTER_CORE_Y_DIST)
            z = gaussianRandom(0, GALAXY_THICKNESS)
            stars.append([Vec3(x, y, z), self.get_star_color()])

        

        for i in range(NUM_STARS // 4):

            x = gaussianRandom(ARM_X_MEAN, ARM_X_DIST)
            y = gaussianRandom(ARM_Y_MEAN, ARM_Y_DIST)
            z = gaussianRandom(0, GALAXY_THICKNESS)
            stars.append([Vec3(x, y, z), self.get_star_color()])



        self.stars = stars
        return self.stars
    
    def elliptical_positions(self):
        stars = []

        MAJOR_AXIS = 75  
        MINOR_AXIS = 50  
        Z_THICKNESS = 5  

        for _ in range(NUM_STARS):
            
            angle = random.uniform(0, 2 * math.pi)
            radius_major = abs(gaussianRandom(0, MAJOR_AXIS))
            radius_minor = radius_major * (MINOR_AXIS / MAJOR_AXIS)

            x = radius_major * math.cos(angle)
            y = radius_minor * math.sin(angle)
            z = gaussianRandom(0, Z_THICKNESS)

            stars.append([Vec3(x, y, z), self.get_star_color()])

        self.stars = stars
        return self.stars
    
    def draw(self, galaxy_center_2d, camera, rx, ry):
        for local_star in self.positions:
            # Convert local star position to world space
            world_star = Vec3(
                galaxy_center_2d.x + local_star[0].x,
                galaxy_center_2d.y + local_star[0].y,
                galaxy_center_2d.z + local_star[0].z
            )

            # Project to 2D
            star_2d = get_2d(world_star - camera, rx, ry)
            if star_2d:
                self.draw_glow_circle(self.screen, local_star[1], star_2d, radius=2, glow_radius=20)
  

        

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
        self.draw_glow_circle(screen, color, obj_pos_2d, radius=2, glow_radius=20)
        pygame.draw.circle(screen, color, obj_pos_2d, 2)

    def get_star_color(self): 

        if self.color: 
            p = np.random.randint(0, 2)
            if p == 0:
                return self.color

        r = random.uniform(0, 100)
        cumulative = 0

        for pct, color in zip(starTypes["percentage"], starTypes["colors"]): 
            cumulative += pct
            if r <= cumulative: 
                return color
            
        return starTypes["colors"][np.random.randint(0, len(starTypes["colors"]))]

