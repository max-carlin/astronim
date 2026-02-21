import pygame 
import numpy as np
from astronim.utils.tools import distance, get_2d, Vec3

class DarkMatter:
    '''Constructs a dark matter halo.

    Parameters
    ----------
    pos : Vec3
        The (x, y, z) position of the halo. 
    base_radius : int
        The radius of the halo.
    '''
    def __init__(self, pos : Vec3, base_radius : int):

        self.static = True
        # self.pos = pos
        self.base_radius = base_radius
        self.positions = self.ellipsoid()
            
        

    def draw(self, screen):
        points_2d = []
        

        try:
            for point in self.positions:
                obj_pos_2d = get_2d(point - DarkMatter.camera, 
                                    DarkMatter.rx, DarkMatter.ry)

                if obj_pos_2d is not None:
                    points_2d.append(obj_pos_2d)
        except ValueError: 
            return
        
        # dist = distance(Vec3(0, 0, 0), DarkMatter.camera)
        # self.radius = max(2, min(20, int(self.base_radius * 500 / dist)))
        
        # pygame.draw.polygon(screen, (0, 0, 255), points_2d, width=1)
        for p in points_2d:
            pygame.draw.circle(screen, (0, 0, 255), p, 1)

    def ellipsoid(self):
        points = []

        # roughly match the galaxy scale:
        # (use MAJOR_AXIS, MINOR_AXIS, and Z_THICKNESS if available)
        major = 250    # horizontal radius
        minor = 200     # vertical radius
        thickness = 20 # z-direction "thickness"

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        u, v = np.meshgrid(u, v)

        x = major * np.cos(u) * np.sin(v)
        y = minor * np.sin(u) * np.sin(v)
        z = thickness * np.cos(v)

        # Flatten and zip to Vec3s, offset by the galaxy position
        for x_val, y_val, z_val in zip(x.ravel(), y.ravel(), z.ravel()):
            points.append(Vec3(x_val,
                            y_val,
                            z_val))

        return points




    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry