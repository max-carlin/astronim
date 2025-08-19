from astronim import *
from utils.tools import Vec3

u = Universe()
G = 6.674e-11
AU = 1.496e11


bhCenter = (0, 0, 0)

M = 1.989e30       
MBH = 1e6 * M         

a = 0.5          
R_init = 500       

# orbital velocity for circular orbit (each star around COM at a/2)
a_m = a * AU
v_orb = np.sqrt(G * M / (2 * a_m))   

offset = np.array([R_init, 0, 0])  

# star positions relative to COM
r1 = np.array([+a/2, 0, 0])  # AU
r2 = np.array([0, -a/2, 0])  # AU

# star velocities relative to COM
v1_rel = np.array([0, +v_orb, 0])   
v2_rel = np.array([0, -v_orb, 0])   

# velocity toward BH
v_cm = np.array([-2e3, 0, 0])       

# add BH
u.black_hole(bhCenter, mass=MBH, vel=(0,0,0))

# add stars
star1 = u.star((r1 + offset).tolist(), mass=M, vel=(v1_rel + v_cm).tolist(), 
               trail=True, color=(200,200,255))
star2 = u.star((r2 +offset).tolist(), mass=M, vel=(v2_rel + v_cm).tolist(), 
               trail=True, color=(255,200,200))


# Changes our camera position
def move_camera(camera): 
    camera.x += 0.1
    camera.z -= 0.3

u.camerafunction = move_camera

# initial position of the camera
u.camera = Vec3(x=490, y=10, z=-25.191541689769413)

# slows down the control speed
u.speed = 0.1

u.record_scene('Hills_Mechanism')
u.main_loop()






