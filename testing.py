from astronim import *
import os
import ffmpeg
from utils.tools import Vec3

u = Universe()
G = 6.674e-11
AU = 1.496e11


# sun = u.star((-3e-6, 0, 0), mass = 1.989e30, vel=[0.0, -8.94e-2, 0.0], trail = True, color = (248, 252, 3))
# star2 = u.star((0.999997, 0.2, 0), mass = 5.972e24, vel=[0.0, 2.98e4, 0.0], trail = True)
# star3 = u.star((0.5, 0.4, 0), mass = 5.972e24, vel=[0.0, 2.98e4, 0.0], trail =True)

# star2.trail_length= 200

# # galaxy = u.elliptical_galaxy((0, 0, 0))
# # galaxy2 = u.spiral_galaxy((100, 100, -300), (201, 226, 255))

# # u.record_scene(output_name= "test_4k")

# u.black_hole((0, 0, 15), mass = 1.989e30 * 1e3 , vel=[0.0, 0, 0.0])


bhCenter = (0, 0, 0)

M = 1.989e30          # 1 Msun
MBH = 1e6 * M         # 10^6 Msun SMBH

a = 0.5                # binary separation in AU (big so it’s visible!)
R_init = 500          # COM distance from BH in AU

# orbital velocity for circular orbit (each star around COM at a/2)
a_m = a * AU
v_orb = np.sqrt(G * M / (2 * a_m))   # m/s

print("Orbital speed:", v_orb, "m/s")

# put COM far away from BH
offset = np.array([R_init, 0, 0])  

# star positions relative to COM
r1 = np.array([+a/2, 0, 0])  # AU
r2 = np.array([0, -a/2, 0])  # AU

# star velocities relative to COM
v1_rel = np.array([0, +v_orb, 0])   # m/s
v2_rel = np.array([0, -v_orb, 0])   # m/s

# bulk COM velocity toward BH
v_cm = np.array([-2e3, 0, 0])       # m/s

# add BH
u.black_hole(bhCenter, mass=MBH, vel=(0,0,0))

# add stars
star1 = u.star((r1).tolist(), mass=M, vel=(v1_rel + v_cm).tolist(), 
               trail=True, color=(200,200,255))
star2 = u.star((r2).tolist(), mass=M, vel=(v2_rel + v_cm).tolist(), 
               trail=True, color=(255,200,200))

def move_camera(camera): 

    camera.x += 0.1
    camera.z -= 0.3

u.camerafunction = move_camera


u.camera = Vec3(x=490, y=10, z=-25.191541689769413)
# u.camera = Vec3(x=480, y=0, z=-3.191541689769413)
# u.rx = -1.1170107212763714
# u.ry = -0.25481807079117247

u.speed = 0.1

# u.record_scene('binary_2')
u.main_loop()






