from astronim.utils.leapfrog import updateParticles
from astronim.utils.tools import distance, Vec3
import numpy as np
from astronim.utils.constants import AU

class Simulation: 
    '''Handles the N-body simulation and all objects in our scene. 

    Attributes
    ----------
    star_objects : list
        A list of all the star objects in our scene. 
    
    star_masses : list
        A list of all the star masses in our scene.

    star_vels : list
        A list of all the star velocities in our scene. 

    star_positions : list
        A list of all the star positions in our scene. 

    static_objects : list
        A list of all the objects whose positions will not be updated according to the N-body simulation.

    Methods
    -------
    add_star(obj): 
        Adds a star object to our simulation. 
    
    add_static(obj): 
        Adds a static object to our simulation

    update(dt): 
        Updates the simulation by one specified time step, dt. 
    '''
    def __init__(self):
        self.star_objects = []
        self.star_masses = []
        self.star_vels = []
        self.star_positions = []
        
        self.static_objects = []

    def add_star(self, star): 
        '''Adds a star object to our simulation. 

        params
        ------
        star : astronim.object
            The object whose position, mass, and velocity will be added to our simulation.
            
        '''
        self.star_objects.append(star)
        self.star_masses.append(star.mass)
        self.star_vels.append(star.velocity)
        self.star_positions.append([star.pos.x, star.pos.y, star.pos.z])

    def add_static(self, obj): 
        '''Adds a static object to our simulation. 

        params
        ------
        obj : astronim.object
            The static object to be added to the scene.
        '''
        self.static_objects.append(obj)

    def update(self, dt): 
        '''Runs one step of our leapfrog integrator and updates the positions and velocities of every particle. 
        Adds the positions of each object to their trail lists.

        params
        ------
        dt : float
            The time step applied to the integrator. 
        '''
        
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






