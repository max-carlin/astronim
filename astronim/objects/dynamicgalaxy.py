import numpy as np
import pygame
import math
from astronim.utils.tools import gaussianRandom, clamp, spiral, Vec3, get_2d, rotation_matrix
from .star import Star
import random

starTypes = {
    "percentage" : [76.45, 12.1, 7.6, 3.0, 0.6, 0.13],
    "colors" : [(255, 205, 111), (255, 210, 161), (255, 244, 234), (248, 247, 255), (202, 215, 255), (170, 191, 255)]
}


class DynamicGalaxy: 

    DEFAULTS = {
        "NUM_STARS": 400,
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
                 color = None, rotation: Vec3 = Vec3(0, 0, 0),
                 seed: int = None, **kwargs):
        # init function should draw the galaxy using only these args
        self.galaxy_type = galaxy_type
        self.color = color

        self.pos = Vec3(pos.x, pos.y, pos.z)
        self.rotation = rotation

        # Update default dictionary with kwargs
        self.params = self.DEFAULTS.copy()
        self.params.update(kwargs)

        # Optional reproducibility: snapshot the global RNG state, seed
        # both numpy + Python random for the duration of star
        # generation (positions, colors, IMF mass sampling), then
        # restore. Same pattern as Galaxy. Lets the morph's
        # sandbox-extracted target and the live scene rebuild produce
        # identical star configurations so the morph lands on the real
        # final layout with no jump.
        _np_state = _py_state = None
        if seed is not None:
            _np_state = np.random.get_state()
            _py_state = random.getstate()
            np.random.seed(seed)
            random.seed(seed)
        try:
            stars = self.spiral_positions()
            if any([self.rotation.x, self.rotation.y, self.rotation.z]):
                stars = [[self.rotate_vector(star[0]), star[1]] for star in stars]

            self.positions = stars
            self.star_objects = self.create_star_objects(self.positions)
        finally:
            if _np_state is not None:
                np.random.set_state(_np_state)
                random.setstate(_py_state)

  

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
    
    def create_star_objects(self, position_color):
        star_objects = []

        for pos_col in position_color:
            pos = pos_col[0] + self.pos     # offset galaxy center
            color = pos_col[1]

            # --- Mass from Kroupa IMF ---
            m_solar = self.sample_mass()
            mass = m_solar * 1.989e30

            # --- Velocity from rotation curve ---
            vel = self.compute_orbital_velocity(pos)

            star = Star(pos, vel=vel, mass=mass, color=color, trail=True)
            star_objects.append(star)

        return star_objects

    
    def sample_mass(self):
        # Kroupa IMF parameters
        if np.random.rand() < 0.7:
            alpha = 1.3
            m_min, m_max = 0.08, 0.5
        else:
            alpha = 2.3
            m_min, m_max = 0.5, 50.0   # upper limit for your sim

        r = np.random.rand()
        return ((m_max**(1-alpha) - m_min**(1-alpha))*r + m_min**(1-alpha))**(1/(1-alpha))

    def compute_orbital_velocity(self, pos, v0=2e5):  # 200 km/s in m/s
        x, y, z = pos.x, pos.y, pos.z
        r = math.sqrt(x*x + y*y)

        if r < 1e-6:
            return Vec3(0,0,0)

        # Tangential unit vector
        vx = -y / r
        vy =  x / r

        # Scale by circular velocity
        return Vec3(vx * v0, vy * v0, 0)


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
