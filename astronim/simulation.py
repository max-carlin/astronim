from astronim.utils.leapfrog import updateParticles
from astronim.utils.tools import distance, Vec3
import numpy as np
from astronim.utils.constants import AU

class Simulation: 
    def __init__(self):
        self.star_objects = []
        self.star_masses = []
        self.star_vels = []
        self.star_positions = []
        
        self.static_objects = []

    def add_star(self, star): 
        self.star_objects.append(star)
        self.star_masses.append(star.mass)
        self.star_vels.append(star.velocity)
        self.star_positions.append([star.pos.x, star.pos.y, star.pos.z])

    def add_static(self, obj): 
        self.static_objects.append(obj)

    def update(self, dt): 
        
        if not self.star_objects: 
            return
        
        
        leapfrog_pos, leapfrog_vel = updateParticles(np.array(self.star_masses), 
                                                     np.array(self.star_positions) * AU, 
                                                     np.array(self.star_vels), 
                                                     dt)
        
        for i, obj in enumerate(self.star_objects): 
            px, py, pz = leapfrog_pos[i]
            
            obj.pos.x = px  / AU
            obj.pos.y = py / AU
            obj.pos.z = pz / AU 
            obj.velocity = leapfrog_vel[i]
            

            self.star_positions[i] = [obj.pos.x, obj.pos.y, obj.pos.z]
            self.star_vels[i] = obj.velocity

            

            obj.trail_list.append([obj.pos.x, obj.pos.y, obj.pos.z])

            if len(obj.trail_list) > obj.trail_length: 
                obj.trail_list.pop(0)






