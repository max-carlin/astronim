import pygame 
import numpy as np
from astronim.objects.line_between import LineBetween
from astronim.utils.tools import get_2d, Vec3, distance, rotation_matrix
from astronim.utils.constants import G
import matplotlib.pyplot as plt
import time



class Grid:
    '''Creates an n x m grid that can be distorted to represent spacetime. 
    '''
    def __init__(self, n:int, m:int, center:Vec3, 
                 spacing:int = 6, color = (255, 255, 255), rotation:Vec3 = Vec3(180, 0, 0), 
                 curvature = None):

        self.m = m
        self.n = n
        self.color = color
        self.center = center
        self.spacing = spacing
        self.static = True
        self.rotation = rotation
        self.curvature = curvature
        self.finished_animating = False
        

        self.init_lines_animate()

        if (type(self.m ) != int) or (type(self.n ) != int) : 
            print('Dimensions of grid must be integers.')
            raise TypeError

    def draw(self, screen): 
        # We'll have to call line.draw for line in self.lines

        for line in self.lines:
            if line.progress < 1:
                self.finished_animating = False
            elif line.progress == 1: 
                self.finished_animating = True

        if self.finished_animating:
            self.init_lines()
        
        for line in self.lines:
            line.set_camera(Grid.camera, Grid.rx, Grid.ry)
            line.draw(screen)

    def init_lines_animate(self):
        """
        Initialize LineBetween objects for a full grid:
        - all horizontal lines between adjacent x-points in each row
        - all vertical lines between adjacent y-points in each column
        """
        xv, yv = self.init_grid()
        self.lines = []

        n_rows, n_cols = xv.shape


        # Horizontal lines (left - right for every row)
        for i in range(n_rows):
            for j in range(n_cols - 1):
                p1 = Vec3(xv[i, j], yv[i, j], self.center.z)
                p2 = Vec3(xv[i, j + 1], yv[i, j + 1], self.center.z)

                p1 = self.apply_curvature(p1)
                p2 = self.apply_curvature(p2)


                p1 = self.rotate_vector(p1)
                p2 = self.rotate_vector(p2)

                self.lines.append(LineBetween(p1, p2, color=self.color,animate=True, speed=0.008))

        # Vertical lines (top - bottom for every column)
        for j in range(n_cols):
            for i in range(n_rows - 1):
                p1 = Vec3(xv[i, j], yv[i, j], self.center.z)
                p2 = Vec3(xv[i + 1, j], yv[i + 1, j], self.center.z)

                p1 = self.apply_curvature(p1)
                p2 = self.apply_curvature(p2)

                p1 = self.rotate_vector(p1)
                p2 = self.rotate_vector(p2)

                self.lines.append(LineBetween(p1, p2, color=self.color, animate=True, speed=0.008))

    def init_lines(self):
        """
        Initialize LineBetween objects for a full grid:
        - all horizontal lines between adjacent x-points in each row
        - all vertical lines between adjacent y-points in each column
        """
        xv, yv = self.init_grid()
        self.lines = []

        n_rows, n_cols = xv.shape


        # Horizontal lines (left - right for every row)
        for i in range(n_rows):
            for j in range(n_cols - 1):
                p1 = Vec3(xv[i, j], yv[i, j], self.center.z)
                p2 = Vec3(xv[i, j + 1], yv[i, j + 1], self.center.z)

                p1 = self.apply_curvature(p1)
                p2 = self.apply_curvature(p2)


                p1 = self.rotate_vector(p1)
                p2 = self.rotate_vector(p2)

                self.lines.append(LineBetween(p1, p2, color=self.color))

        # Vertical lines (top - bottom for every column)
        for j in range(n_cols):
            for i in range(n_rows - 1):
                p1 = Vec3(xv[i, j], yv[i, j], self.center.z)
                p2 = Vec3(xv[i + 1, j], yv[i + 1, j], self.center.z)

                p1 = self.apply_curvature(p1)
                p2 = self.apply_curvature(p2)

                p1 = self.rotate_vector(p1)
                p2 = self.rotate_vector(p2)

                self.lines.append(LineBetween(p1, p2, color=self.color))



    def init_grid(self): 
        starting_x = self.center.x - (self.n // 2)
        ending_x = self.center.x + (self.n // 2)

        starting_y = self.center.y - (self.m // 2)
        ending_y = self.center.y + (self.m // 2)

        x = np.linspace(starting_x, ending_x, self.spacing)
        y = np.linspace(starting_y, ending_y, self.spacing)

        xv, yv = np.meshgrid(x, y)
        
        return xv, yv
    
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
    
    def apply_curvature(self, v):
        '''r = -k (r - r_obj) / (r - r_obj)^p

        k is the strength of the curvature
        p controls the falloff
        ''' 
        if self.curvature is None: 
            return v
        
        obj_pos = self.curvature.pos
        mass = self.curvature.mass
        
        dx = v.x - obj_pos.x
        dy = v.y - obj_pos.y
        dz = v.z - obj_pos.z

        r = (dx**2 + dy**2 + dz**2)**0.5
        
        # avoid singularity
        eps = 1e-6
        if r<eps: 
            return v
        
        k = 0.07 # * np.sqrt(mass)
        p = 2.0  #+ 0.05 * np.log10(mass + 1)

        # direction from object to v; negative points toward the object
        # displacement = -k * (direction) / r**p
        # (This gives a displacement that scales like r^(1-p). For p=2 the magnitude ~ 1/r.)
        disp_x = -k * dx / (r**p)
        disp_y = -k * dy / (r**p)
        disp_z = -k * dz / (r**p)

        return Vec3(v.x + disp_x, v.y + disp_y, v.z + disp_z)


    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry


# if __name__ == "__main__": 
#     test = Grid(6, 6, Vec3(0, 0, 0), spacing=10)

#     xv, yv = test.init_grid()

#     plt.plot(xv, yv, marker = 'o')
#     plt.show()


