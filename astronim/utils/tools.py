
import numpy as np
import math
from dataclasses import dataclass


#Settings
WIDTH = 1200
HEIGHT = 800
DEPTH = 500


# Vec3 Class
@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __sub__(self, other): 
        return Vec3(self.x - other.x, self.y - other.y, self.z-other.z)


# docstring wikipedia links in get_2d
def rotation_matrix(theta): 
    return np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])

#Getting 2d Point coordinates from 3d vectors
def get_2d(pos, rx, ry): 

    '''
    Takes 3D coordinates and converts them into 2d. 

    parameters
    ----------
    pos : Vec3
        3 dimensional vector containing x, y, z coordinates respectively. 
    rx : float
        rotation angle about x-axis, in radians
    ry : float
        rotation angle about y-axis, in radians

    returns
    -------
    2-dimensional (x, y) coordinates as integers. 


    Mainly taken from here under the "In Two Dimensions section": 
    https://en.wikipedia.org/wiki/Rotation_matrix#Common_2D_rotations

    And here under "Weak perspective projection" for depth stuff: 
    https://en.m.wikipedia.org/wiki/3D_projection 
    '''

    #horizonatl rotation (about x-axis)
    pos.x, pos.z = np.matmul(rotation_matrix(rx), np.array([pos.x, pos.z]))

    #vertical rotation (about y-axis)
    pos.y, pos.z = np.matmul(rotation_matrix(ry), np.array([pos.y, pos.z]))

    #Get rid of points behind the camera
    if pos.z <= 0.1: 
        return None
    
    pos.x = pos.x* DEPTH/pos.z
    pos.y = pos.y * DEPTH / pos.z

    pos.x += WIDTH / 2
    pos.y = HEIGHT / 2 - pos.y

    return (int(pos.x), int(pos.y))

def gaussianRandom(mean = 0, stdev =1): 
    u = 1 - np.random.random()
    v = np.random.random()
    z = np.sqrt(-2 * np.log(u)) * np.cos(2.0 * np.pi * v)
    return z * stdev + mean

def clamp(value, minimum, maximum): 
    return np.minimum(maximum, np.maximum(minimum, value))

def spiral(x, y, z, offset, ARM_X_DIST = 100, SPIRAL = 3.0):
    r = math.sqrt(x**2 + y**2)
    theta = offset
    if x > 0:
        theta += math.atan(y / x)
    else:
        theta += math.atan(y / x) + np.pi
    theta += (r / ARM_X_DIST) * SPIRAL
    return Vec3(r * math.cos(theta), r * math.sin(theta), z)

def distance(obj_pos_3d, camera):
    dx = obj_pos_3d.x - camera.x
    dy = obj_pos_3d.y - camera.y
    dz = obj_pos_3d.z - camera.z
    dist = (dx**2 + dy**2 + dz**2)**0.5 + 1

    return dist


