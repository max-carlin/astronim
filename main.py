import pygame 
import numpy as np
from dataclasses import dataclass, replace
import random
from utils.leapfrog import updateParticles


pygame.init()

# Vec3 Class
@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __sub__(self, other): 
        return Vec3(self.x - other.x, self.y - other.y, self.z-other.z)
    
#Settings
WIDTH = 1200
HEIGHT = 800
DEPTH = 500

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Astronim')


camera = Vec3(0, 0, 0)
rx = 0
ry = 0
speed = 0.1


def rotation_matrix(theta): 
    return np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])

#Getting 2d Point coordinates from 3d vectors
def get_2d(pos): 

    '''
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

clock = pygame.time.Clock()
running = True
pygame.mouse.set_visible(True)
pygame.event.set_grab(False)

def disk(n_points=1000, radius=50, thickness=30):
    points = []
    for _ in range(n_points):
        angle = random.uniform(0, 2*np.pi)
        r = radius + random.uniform(-thickness/2, thickness/2)  # fuzzy band
        x = r * np.cos(angle)
        z = r * np.sin(angle)
        y = 0 #random.uniform(-2, 2)  # tiny vertical jitter
        points.append(Vec3(x, y, z))
    return points

def circle(n_points =1000, radius =2): 
    points = []
    for _ in range(n_points): 
        angle = random.uniform(0, 2*np.pi)
        r = radius
        x = r*np.cos(angle)
        y = r*np.sin(angle)
        z = 0
        points.append(Vec3(x, y, z))

    return points

def parabola(x, a = 0.9): 
    return a * x**2

def eye_shape(n_points = 200, r = 50, center = (0, 0, 0), height = 1.75): 
    points = []
    x0 = -r
    xf = r


    # First, the top half
    for x in np.linspace(x0, xf, int(n_points / 2)): 
        y = -height * (1 - (x / r)**2)
        z = 0
        points.append(Vec3(x, y, z))

    # bottom half
    for x in np.linspace(x0, xf, int(n_points / 2)): 
        y = height * (1 - (x / r)**2)
        z = 0
        points.append(Vec3(x, y, z))

    return points

def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

        


        
    


hole_points = eye_shape()
disk_points = disk()


# To create a BH: Let's first create a circle

disk_pos = Vec3(0, 0, 0)



#______________________________ FROM SCRATCH___________________________________


BH_pos = Vec3(0, 0, 200)

class BlackHole: 
    def __init__(self, BH_pos, radius):
        self.BH_pos = BH_pos
        self.base_radius = radius
        self.front_disk = []
        self.back_disk = []
        self.half_disk_random_factors()

        
        

    def update(self, camera_pos): 
        bh_2d = get_2d(self.BH_pos - camera)
   
        #update according to depth somehow
        if bh_2d: 

            dx = self.BH_pos.x - camera_pos.x
            dy = self.BH_pos.y - camera_pos.y
            dz = self.BH_pos.z - camera_pos.z

            self.dist = (dx**2 + dy**2 + dz**2)**0.5
            fov_scale = 500 # this is just depth
            self.radius = int(self.base_radius *fov_scale / self.dist)
    


            if self.radius > 0:
                
                self.draw_half_disk(self.front_angles, self.front_r_factors)#disk
                # pygame.draw.circle(screen, (255, 255,255), bh_2d, self.radius, width=2)
                # pygame.draw.circle(screen, (0, 0, 0), bh_2d, self.radius - 0.1)

                self.draw_star(bh_2d, (252, 250, 210), screen, radius=self.radius, glow_radius=100)
                pygame.draw.circle(screen, (0, 0, 0), bh_2d, self.radius - 0.1*self.radius)


                self.draw_half_disk(self.back_angles, self.back_r_factors) #disk


    def draw_half_disk(self, angles, r_factors): 

        for angle, r_factor in zip(angles, r_factors):

            r = self.radius + r_factor * 2* DEPTH / self.dist

            x = r * np.cos(angle) + self.BH_pos.x
            z = r * np.sin(angle) + self.BH_pos.z
            y = 0 + self.BH_pos.y #random.uniform(-2, 2)  # tiny vertical jitter

            p = get_2d(Vec3(x, y, z) - camera)
            if p: 
                # pygame.draw.circle(screen,  (200, 100, 50), p, 2)
                self.draw_star(p, self.get_color(r_factor), screen, glow_radius=20)

    def get_color(self, r_factor): 
        # scale radius contribution by thickness, not full DEPTH
        r = self.radius + r_factor * (self.thickness / 2)
        min_r = self.radius - self.thickness / 2
        max_r = self.radius + self.thickness / 2
        
        # normalize to 0..1
        t = (r - min_r) / (max_r - min_r)
        
        if t >= 0.9: 
            return (252, 196, 63)
        if t >= 0.8: 
            return (255, 208, 0)    
        if t >= 0.6: 
            return (252, 226, 109)  
        if t >= 0.5: 
            return (250, 244, 160)  


        return (252, 250, 210)  
        


        



    def half_disk_random_factors(self, n_points=1000, radius=50, thickness=15):

        self.front_angles = []
        self.front_r_factors = []
        self.thickness = thickness


        for _ in range(int(n_points / 2)):
            angle = random.uniform(0, np.pi)
            r_factor = random.uniform(-thickness/2, thickness/2)
            self.front_angles.append(angle)
            self.front_r_factors.append(r_factor)

        self.back_angles = []
        self.back_r_factors = []

        for _ in range(int(n_points / 2)):
            angle = random.uniform(np.pi, 2*np.pi)
            r_factor = random.uniform(-thickness/2, thickness/2)
            self.back_angles.append(angle)
            self.back_r_factors.append(r_factor)



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

        







        
        

blackhole = BlackHole(BH_pos, 25)

while running: 
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    dx, dy = pygame.mouse.get_rel()
    rx += np.radians(dx / 5)
    ry -= np.radians(dy / 5)

    if keys[pygame.K_w]:
        camera.x += np.sin(rx) * speed*10
        camera.z += np.cos(rx) * speed*10
    if keys[pygame.K_s]:
        camera.x -= np.sin(rx) * speed*10
        camera.z -= np.cos(rx) * speed*10
    if keys[pygame.K_a]:
        camera.x += np.sin(rx - np.pi / 2) * speed*10
        camera.z += np.cos(rx - np.pi / 2) * speed*10
    if keys[pygame.K_d]:
        camera.x += np.sin(rx + np.pi / 2) * speed*10
        camera.z += np.cos(rx + np.pi / 2) * speed*10
    if keys[pygame.K_SPACE]:
        camera.y += speed*10
    if keys[pygame.K_LCTRL]:
        camera.y -= speed*10

    screen.fill((0, 0, 0))

    


    blackhole.update(camera)




    pygame.display.flip()
    clock.tick(60)

pygame.quit()





























# white = (255, 255, 255)
# green = (0, 255, 0)
# blue = (0, 0, 128)
# font = pygame.font.Font('freesansbold.ttf', 32)
# text_surface = font.render('max', True, green, blue)
# textRect = text_surface.get_rect()
# textRect.center = (WIDTH // 2, HEIGHT // 2)
# text_pos3d = Vec3(0, 0, 0)

#     text_pos = get_2d(text_pos3d - camera)
    
#     if text_pos:
#         textRect.center = text_pos
#         screen.blit(text_surface, textRect)





    # hole_poly = []
    # for p in hole_points: 
    #     dp = get_2d( p - camera)
    #     if dp:
    #         hole_poly.append(dp)



    # for p in disk_points: 
    #     dp = get_2d( p - camera)
    #     if dp and (len(hole_poly) == 0 or not point_in_poly(dp[0], dp[1], hole_poly)):
    #         pygame.draw.circle(screen,  (200, 100, 50), dp, 2)


    # for dp in hole_poly:
    #     pygame.draw.circle(screen, (200, 100, 50), dp, 2)