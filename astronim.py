import pygame 
import numpy as np
from dataclasses import dataclass
from utils.leapfrog import updateParticles
from utils.tools import get_2d
from utils.tools import Vec3
from objects.star import Star
from objects.galaxy import Galaxy
from objects.blackhole import BlackHole
import os
import ffmpeg
import tempfile
from utils.constants import WIDTH, HEIGHT, DEPTH


class Universe:
    '''
    Astronim: An interactive N-body astronomy animation engine.
    Astronim implements a leapfrog integration scheme to update bodies in real-time, with rendering done by Pygame. 
    It is interactive and includes recording funtionality. 

    Controls:
    ----------
    W/A/S/D - move in XY plane
    SPACE / CTRL - move up/down
    Mouse - look around
    SHIFT - speed boost

    Methods:
    -------
    main_loop()
        Loops through all objects in the scene and updates the scene. 

    star(pos, mass = 1.989e30, vel = [0.0, -8.94e-2, 0.0],  trail = False, color = (255, 255, 255))
        Adds a star object to the scene. 

    blackhole(pos, mass = 1.989e30, vel = [0.0, -8.94e-2, 0.0],  trail = False, color = (255, 255, 255))
        Adds a blackhole object to the scene. 

    spiral_galaxy(pos, color = None)
        Adds a galaxy object to the scene. 

    elliptical_galaxy(pos, color = None)
        Adds a galaxy object to the scene. 

    irregular_galaxy(pos, color = None)
        Adds a galaxy object to the scene. 

    record_scene(output_name = 'output'): 
        Records the scene.

    camera_movement()
        Applys a specified Universe.camerafunction to the camera. 
 
    '''
    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Astronim')
        self.background_color = (0, 0, 0)

        #recording stuff
        self.frame_count = 0
        self.recording = False
        self.frames_dir = None 
        

        #camera init
        self.camera = Vec3(0, 0, 0)
        self.camerafunction = None
        self.rx = 0
        self.ry = 0
        self.t = 0



        #pygame init
        self.clock = pygame.time.Clock()
        self.running = True
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

        # All objects that act as a point mass and are generated as a single body will be stored in these arrays
        # e.g. stars, planets, BH's, etc. 
        self.num_objs = 0 # everytime we add an object to the scene, this number increases
        self.star_objects = []
        self.star_masses = []
        self.star_vels = []
        self.star_positions = []
        
        self.delta_t = 0.1 * 86400 #Time step (rn 0.1 days) 

        #All static objects (will think of a way to make them better / not static later)
        self.static_objects = []
        self.num_static_objs = 0

        #control speeds
        self.speed = 0.2
        self.shift_speed_factor = 10
        self.enable_mouse = True

        

        


    def main_loop(self):

       while self.running: 
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
            if self.enable_mouse:
                    
                dx, dy = pygame.mouse.get_rel()
                self.rx += np.radians(dx / 5)
                self.ry -= np.radians(dy / 5)

            # ------------CONTROLS--------------
            self.controls(keys)

            self.screen.fill(self.background_color)

            # update all objects in the scene, static objects are not updated using updatebodies
            self.update_star_objects()
            self.update_static_objects()
            self.camera_movement()


            pygame.display.flip()

            if self.recording and self.frames_dir:
                frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_count:04d}.png")
                pygame.image.save(self.screen, frame_path)
                self.frame_count += 1
            self.clock.tick(120)

       pygame.quit()
       if self.recording and self.frames_dir: 
            os.system(
                f"ffmpeg -framerate 60 -i {self.frames_dir}/frame_%04d.png "
                f"-c:v libx264 -crf 15 -preset slow -pix_fmt yuv420p {self.filename}.mp4"
            )
           

    def star(self, pos, mass = 1.989e30, vel = [0.0, -8.94e-2, 0.0],  trail = False, color = (255, 255, 255)): 
        obj_type = 'star'
        obj = ObjGenerator(pos, mass, vel, self.num_objs, obj_type, trail, color = color)
        self.num_objs += 1
        self.star_objects.append(obj)
        return obj
    
    def black_hole(self, pos, mass = 1.989e30, vel = [0.0, -8.94e-2, 0.0],  trail = False, color = (255, 255, 255)): 
        obj_type = 'blackhole'
        obj = ObjGenerator(pos, mass, vel, self.num_objs, obj_type, trail, color = color)
        self.num_objs +=1
        self.star_objects.append(obj)
    
    def spiral_galaxy(self, pos, color = None): 
        obj_type = 'spiral_galaxy'
        obj = Galaxy(pos,  self.screen, color, self.camera, self.rx, self.ry, obj_type )
        self.num_static_objs += 1
        self.static_objects.append(obj)
        return obj
    
    def elliptical_galaxy(self, pos, color = None): 
        obj_type = 'elliptical_galaxy'
        obj = Galaxy(pos,  self.screen, color, self.camera, self.rx, self.ry, obj_type )
        self.num_static_objs += 1
        self.static_objects.append(obj)
        return obj
    
    def irregular_galaxy(self, pos, color = None): 
        obj_type = 'irregular_galaxy'
        obj = Galaxy(pos,  self.screen, color, self.camera, self.rx, self.ry, obj_type )
        self.num_static_objs += 1
        self.static_objects.append(obj)
        return obj
    


    def update_arrays(self): 
        au = 1.496e11
        self.star_positions = []
        self.star_masses = []
        self.star_vels = []
        for object in self.star_objects:
         
            self.star_positions.append([object.pos.x * au  , object.pos.y * au , object.pos.z * au])
            self.star_masses.append(object.mass)
            self.star_vels.append(object.velocity)

    def update_star_objects(self): 
        # -------------LEAPFROG AND UPDATE BODIES SCHEME-----------------------
        if self.star_objects and len(self.star_masses) < len(self.star_objects):
            self.update_arrays() #check if the number of star_masses is less than the number of objects.
            # If so, we need to reconstruct our star_masses and velocity arrays for the update particles function.
      
        # run one step of the integrator and convert to au
        if self.star_objects:
            leapfrog_pos, leapfrog_vel = updateParticles(np.array(self.star_masses), 
                                                            np.array(self.star_positions), 
                                                            np.array(self.star_vels), self.delta_t )
        
            # re-write star_positions and velocities so we're actually updating
            self.star_positions = leapfrog_pos.tolist()
            self.star_vels = leapfrog_vel.tolist()

        #move and update each object   
        draw_list = []      
        for obj in self.star_objects:

            px, py, pz = np.array(self.star_positions[obj.objID]) / 1.496e11
            obj.pos.x = px
            obj.pos.y = py
            obj.pos.z = pz


            obj_pos_2d = get_2d(obj.pos - self.camera, self.rx, self.ry)
            if obj_pos_2d:
                # need to know order in which to render (mainly just for BH which is annoying)
                dx = obj.pos.x - self.camera.x
                dy = obj.pos.y - self.camera.y
                dz = obj.pos.z - self.camera.z
                dist = (dx**2 + dy**2 + dz**2)**0.5
                draw_list.append((dist, obj, obj_pos_2d))

        draw_list.sort(key = lambda x: x[0], reverse = True) #sort by distance (x[0])


        

        for dist, obj, obj_pos_2d in draw_list:
            BlackHole.set_camera(self.camera, self.rx, self.ry)
            Star.set_camera(self.camera, self.rx, self.ry)

            obj.draw_object(obj_pos_2d, self.screen) 

    

            # if trail = true, we just append to the trail_list and draw with pygame.draw.lines
            if obj.trail:                
                obj.trail_list.append([obj.pos.x, obj.pos.y, obj.pos.z])
                obj_path_2d = [
                    pt for pt in (get_2d(Vec3(*p) - self.camera, self.rx, self.ry) for p in obj.trail_list) if pt ]
                if len(obj_path_2d) > 1:
                    if len(obj_path_2d) > obj.trail_length: 
                        # we only want to plot the trail if its greater than 1 and less than desired trail length
                        obj_path_2d = obj_path_2d[-obj.trail_length :]
                        obj.trail_list = obj.trail_list[-obj.trail_length:]
                        
                    pygame.draw.lines(self.screen, (255, 255, 255), False, obj_path_2d, 1)

    def update_static_objects(self): 
        for object in self.static_objects: 
            object.update(self.camera, self.rx, self.ry)



    def controls(self, keys):


        if keys[pygame.K_w]:
            self.camera.x += np.sin(self.rx) * self.speed 
            self.camera.z += np.cos(self.rx) * self.speed 
        if keys[pygame.K_s]:
            self.camera.x -= np.sin(self.rx) * self.speed 
            self.camera.z -= np.cos(self.rx) * self.speed 
        if keys[pygame.K_a]:
            self.camera.x += np.sin(self.rx - np.pi / 2) * self.speed 
            self.camera.z += np.cos(self.rx - np.pi / 2) * self.speed 
        if keys[pygame.K_d]:
            self.camera.x += np.sin(self.rx + np.pi / 2) * self.speed 
            self.camera.z += np.cos(self.rx + np.pi / 2) * self.speed 
        if keys[pygame.K_SPACE]:
            self.camera.y += self.speed 
        if keys[pygame.K_LCTRL]:
            self.camera.y -= self.speed 


        if keys[pygame.K_w] and keys[pygame.K_RSHIFT]:
            self.camera.x += np.sin(self.rx) * self.speed * self.shift_speed_factor
            self.camera.z += np.cos(self.rx) * self.speed * self.shift_speed_factor
        if keys[pygame.K_s] and keys[pygame.K_RSHIFT]:
            self.camera.x -= np.sin(self.rx) * self.speed * self.shift_speed_factor
            self.camera.z -= np.cos(self.rx) * self.speed * self.shift_speed_factor
        if keys[pygame.K_a] and keys[pygame.K_RSHIFT]:
            self.camera.x += np.sin(self.rx - np.pi / 2) * self.speed * self.shift_speed_factor
            self.camera.z += np.cos(self.rx - np.pi / 2) * self.speed * self.shift_speed_factor
        if keys[pygame.K_d] and keys[pygame.K_RSHIFT]:
            self.camera.x += np.sin(self.rx + np.pi / 2) * self.speed * self.shift_speed_factor
            self.camera.z += np.cos(self.rx + np.pi / 2) * self.speed * self.shift_speed_factor

    def record_scene(self, output_name = 'output'): 
        self.recording = True
        self.filename = output_name
        self.frame_count = 0
        self.frames_tempdir = tempfile.TemporaryDirectory()
        self.frames_dir = self.frames_tempdir.name

    def camera_movement(self):
        if self.camerafunction != None:
            self.camerafunction(self.camera)



class ObjGenerator():
    def __init__( self, pos, mass, velocity,  objID, obj_type, trail = False, color = (255, 255, 255)):
        self.mass = mass
        self.velocity = velocity
        self.pos = Vec3(pos[0] , pos[1] , pos[2] )
        self.objID = objID
        self.color = color
        self.obj_type = obj_type


        #trail stuff
        self.trail = trail
        self.trail_list = []
        self.trail_length = 50


    def update_trail(self):

        self.trail_list.append([self.pos.x, self.pos.y, self.pos.z])

    def draw_object(self, obj_pos_2d, screen): 
        #draws the object according to obj_type
        self.object_dict = {'star' : Star, 
                            'blackhole': BlackHole }
        # EVERY OBJECT CLASS MUST TAKE POS, SCREEN, COLOR, 3dPOS
        obj_kind = self.object_dict[self.obj_type](obj_pos_2d, screen, self.color, self.pos)
        


class StaticGenerator: 
    def __init__(self, center_3d, color, screen, camera, rx, ry, obj_type, obj_id):

        self.center_3d = center_3d
        self.screen = screen
        self.color = color
        self.camera = camera
        self.rx = rx
        self.ry = ry
        self.obj_type = obj_type
        self.obj_id = obj_id

    def draw_object(self): 
        static_obj_dict = {'galaxy': Galaxy}
        obj_kind = static_obj_dict[self.obj_type](self.center_3d, self.screen, self.color, self.camera, self.rx, self.ry)



        





