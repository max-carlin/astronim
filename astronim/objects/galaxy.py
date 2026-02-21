import numpy as np
import pygame
import math
from astronim.utils.tools import gaussianRandom, clamp, spiral, Vec3, get_2d, rotation_matrix
from .star import Star
import random

# NUM_STARS = 500
# NUM_ARMS = 4

# #std in z
# GALAXY_THICKNESS = 5

# #std of the core
# CORE_X_DIST = 15 #33
# CORE_Y_DIST = 15 #33

# OUTER_CORE_X_DIST =  50 #100
# OUTER_CORE_Y_DIST = 50 #100

# ARM_X_DIST =  50 #100
# ARM_Y_DIST = 25#50
# ARM_X_MEAN = 100 #200
# ARM_Y_MEAN = 50 #100

# SPIRAL = 1.5 #3.0
# ARMS = 4

# HAZE_RATIO = 0.5

starTypes = {
    "percentage" : [76.45, 12.1, 7.6, 3.0, 0.6, 0.13],
    "colors" : [(255, 205, 111), (255, 210, 161), (255, 244, 234), (248, 247, 255), (202, 215, 255), (170, 191, 255)]
}


class Galaxy: 

    DEFAULTS = {
        "NUM_STARS": 500,
        "NUM_ARMS": 4,
        "GALAXY_THICKNESS": 5,
        "CORE_X_DIST": 15,
        "CORE_Y_DIST": 15,
        "OUTER_CORE_X_DIST": 50,
        "OUTER_CORE_Y_DIST": 50,
        "ARM_X_DIST": 50,
        "ARM_Y_DIST": 25,
        "ARM_X_MEAN": 100,
        "ARM_Y_MEAN": 50,
        "SPIRAL": 1.5,
        "ARMS": 4,
        "MAJOR_AXIS" : 75,  
        "MINOR_AXIS" : 50,  
        "Z_THICKNESS": 5 
    }

    def __init__(self, pos:Vec3, galaxy_type: str = 'spiral_galaxy', 
                 color = None, rotation: Vec3 = Vec3(0, 0, 0), **kwargs):
        # init function should draw the galaxy using only these args 
        self.galaxy_type = galaxy_type
        self.color = color
        
        self.pos = Vec3(pos.x, pos.y, pos.z)
        self.rotation = rotation

        # Update default dictionary with kwargs
        self.params = self.DEFAULTS.copy()
        self.params.update(kwargs)

        types = {'spiral_galaxy': self.spiral_positions(), 
                 'irregular_galaxy':self.irregular_positions(), 
                 "elliptical_galaxy":self.elliptical_positions()}
        stars = types[galaxy_type]
        # Apply rotation
        if any([self.rotation.x, self.rotation.y, self.rotation.z]):
            stars = [[self.rotate_vector(star[0]), star[1]] for star in stars]

        self.positions = stars

        self.static = True

        

    def update(self, camera, rx, ry):
        # just re-draw based on camera position/rotation
        self.draw(self.pos, camera, rx, ry)

    def spiral_positions(self): 
        stars = [] #pos and color of star in the form [[vec, color], [vec, color], ...]

        for i in range(self.params["NUM_STARS"] // 4):
            x = gaussianRandom(0, self.params["CORE_X_DIST"])
            y = gaussianRandom(0, self.params["CORE_Y_DIST"])
            z = gaussianRandom(0, self.params["GALAXY_THICKNESS"])
            stars.append([Vec3(x, y, z), self.get_star_color()])
            

        for i in range(self.params["NUM_STARS"] // 4):
            x = gaussianRandom(0, self.params["OUTER_CORE_X_DIST"])
            y = gaussianRandom(0, self.params["OUTER_CORE_Y_DIST"])
            z = gaussianRandom(0, self.params["GALAXY_THICKNESS"])
            stars.append([Vec3(x, y, z), self.get_star_color()])



        for arm in range(int(self.params["ARMS"])): 
            offset = arm *2 * math.pi / self.params["ARMS"]

            for i in range(self.params["NUM_STARS"] // 4):
                pos = spiral(
                    gaussianRandom(self.params["ARM_X_MEAN"], self.params["ARM_X_DIST"]), 
                    gaussianRandom(self.params["ARM_Y_MEAN"], self.params["ARM_Y_DIST"]), 
                    gaussianRandom(0, self.params["GALAXY_THICKNESS"]), 
                    offset
                )
                stars.append([pos, self.get_star_color()])

        self.stars = stars
        return self.stars
    
    def irregular_positions(self): 
        stars = [] #pos and color of star in the form [[vec, color], [vec, color], ...]

        for i in range(self.params["NUM_STARS"] // 4):
            x = gaussianRandom(0, self.params["CORE_X_DIST"])
            y = gaussianRandom(0, self.params["CORE_Y_DIST"])
            z = gaussianRandom(0, self.params["GALAXY_THICKNESS"])
            stars.append([Vec3(x, y, z), self.get_star_color()])

        for i in range(self.params["NUM_STARS"] // 4):
            x = gaussianRandom(0, self.params["OUTER_CORE_X_DIST"])
            y = gaussianRandom(0, self.params["OUTER_CORE_Y_DIST"])
            z = gaussianRandom(0, self.params["GALAXY_THICKNESS"])
            stars.append([Vec3(x, y, z), self.get_star_color()])

        

        for i in range(self.params["NUM_STARS"] // 4):

            x = gaussianRandom(self.params["ARM_X_MEAN"], self.params["ARM_X_DIST"])
            y = gaussianRandom(self.params["ARM_Y_MEAN"], self.params["ARM_Y_DIST"])
            z = gaussianRandom(0, self.params["GALAXY_THICKNESS"])
            stars.append([Vec3(x, y, z), self.get_star_color()])



        self.stars = stars
        return self.stars
    
    def elliptical_positions(self):
        stars = []


        for _ in range(self.params["NUM_STARS"]):
            
            angle = random.uniform(0, 2 * math.pi)
            radius_major = abs(gaussianRandom(0, self.params["MAJOR_AXIS"]))
            radius_minor = radius_major * (self.params["MINOR_AXIS"] / self.params["MAJOR_AXIS"])

            x = radius_major * math.cos(angle)
            y = radius_minor * math.sin(angle)
            z = gaussianRandom(0, self.params["Z_THICKNESS"])

            stars.append([Vec3(x, y, z), self.get_star_color()])

        self.stars = stars
        return self.stars
    
    def rotate_vector(self, v: Vec3):
        """Rotate vector v by self.rotation (in radians) around x, y, z axes."""
        x, y, z = v.x, v.y, v.z
        rx, ry, rz = self.rotation.x, self.rotation.y, self.rotation.z

        # Around x-axis
        y, z = np.matmul(rotation_matrix(rx), np.array([y, z]))

        # Around y-axis
        z, x = np.matmul(rotation_matrix(ry), np.array([z, x]))

        # Around z-axis
        x, y = np.matmul(rotation_matrix(rz), np.array([x, y]))

        return Vec3(x, y, z)
    
    def draw(self, screen):
        for local_star in self.positions:
            # Convert local star position to world space
            world_star = Vec3(
                self.pos.x + local_star[0].x,
                self.pos.y + local_star[0].y,
                self.pos.z + local_star[0].z
            )

            # Project to 2D
            star_2d = get_2d(world_star - Galaxy.camera, Galaxy.rx, Galaxy.ry)
            if star_2d:
                self.draw_glow_circle(screen, local_star[1], star_2d, radius=2, glow_radius=20)
  

        

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

    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry
