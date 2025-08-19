import numpy as np
from utils.forces import calculateForceVectors


def updateParticles(masses, positions, velocities, delta_time):
    """
        Evolve particles in time via leap-frog integrator scheme.

        This function takes masses, positions, velocities, and a
        time step (delta_time), calculates the next position and
        velocity, and then returns the updated (next) particle
        positions and velocities.

        Parameters
        ----------
        masses : np.ndarray
            1D array containing masses for all particles in kg.
            Its length is the number of particles (AKA "N").

        positions : np.ndarray
            2D array containing (x, y, z) positions for each particle.
            Shape is (N, 3) where N is the number of particles.

        velocities : np.ndarray
            2D array containing (x, y, z) velocities for each particle.
            Shape is (N, 3) where N is the number of particles.

        delta_time : float
            Evolve system for time delta_time in seconds.

        Returns
        -------
        updated positions and velocities : (2D positions np.array, 2D velocities np.array)
            Each being a 2D array with shape (N, 3), where N is the
            number of particles.
    """
    # Make copies of the (starting) positions and velocities
    starting_positions = np.array(positions)
    starting_velocities = np.array(velocities)

    num_of_particles, _ = starting_positions.shape

       # Make sure the three input arrays have consistent shapes
    if starting_velocities.shape != starting_positions.shape:
        raise ValueError("velocities and positions have different shapes")

    # Make sure the number of masses matches the number of particles
    if len(masses) != num_of_particles:
        raise ValueError("Length of masses differs from the first dimension of positions")

    # Calculate net force vectors on all particles at the starting positions
    starting_forces = np.array(calculateForceVectors(masses, starting_positions))

    # Calculate the acceleration due to gravity at the starting positions
    # Equation: acceleration = force / mass
    starting_accelerations = starting_forces / np.array(masses).reshape(num_of_particles, 1)

    #position = velocity_0 * time + 0.5 * acceleration * time**2
    nudge = starting_velocities * delta_time + 0.5 * starting_accelerations * delta_time ** 2
    ending_positions = starting_positions + nudge

    # Calculate net force vectors on all particles at the ending positions
    ending_forces = np.array(calculateForceVectors(masses, ending_positions))

    #acceleration = force / mass
    ending_accelerations = ending_forces / np.array(masses).reshape(num_of_particles, 1)

    #velocity = velocity_0 + 0.5 * acceleration * time
    ending_velocities = (starting_velocities + 0.5 * (ending_accelerations + starting_accelerations) * delta_time)

    return ending_positions, ending_velocities


def calculateTrajectories(masses, initial_positions, initial_velocities, delta_t, total_t): 
    '''
    A function that updates the trajectories of n-particles given a total time to evolve the system and a time step. 
    At each time step, a leapfrog integrator is employed to calculate the updated positions and velocities.
    
    Parameters
    ----------
    
    masses : np.ndarray
        1D array containing the masses for all particles in kg. 
        Length is N for N particles
    
    initial_positions : np.ndarray
        2D array containing the initial positions of each particle in (x,y,z) coordinates. 
        Shape is (N, 3) where there are three coordinates for each particle
        
    initial_velocities : np.ndarray
        2D array containing the initial velocities of each particle (v_x, v_y, v_z). 
        Shape is (N, 3) where there are three velocities for each particle. 
        
    delta_t : float
        The time step used for each step of the simulation, in seconds.
        
    total_t : float
        The total time to evolve the system, in seconds 
    
    Returns
    -------
    times : a 1D numpy array of times
    
    positions : a 3D numpy array of positions (n_times x n_particles x n_positions)
    
    velocities : a 3D numpy array of velocities (n_times x n_particles x n_positions
    '''
    
    # Create an array of times 

    
    # Create position and velocity lists
    positions = [] 
    velocities = []
    
    # Set current position and velocity
    currentPositions = initial_positions
    currentVelocities = initial_velocities
    
    
    #update particles
    particles = updateParticles(masses, currentPositions, currentVelocities, delta_t)
    
    #replace current position and velocity
    currentPositions = particles[0]
    currentVelocities = particles[1]
    
    # add new positions to arrays
    positions.append(currentPositions)
    velocities.append(currentVelocities)
        
        #return times, positions, and velocities
    return  np.array(positions), np.array(velocities)


def old(masses, initial_positions, initial_velocities, delta_t, total_t): 
    '''
    A function that updates the trajectories of n-particles given a total time to evolve the system and a time step. 
    At each time step, a leapfrog integrator is employed to calculate the updated positions and velocities.
    
    Parameters
    ----------
    
    masses : np.ndarray
        1D array containing the masses for all particles in kg. 
        Length is N for N particles
    
    initial_positions : np.ndarray
        2D array containing the initial positions of each particle in (x,y,z) coordinates. 
        Shape is (N, 3) where there are three coordinates for each particle
        
    initial_velocities : np.ndarray
        2D array containing the initial velocities of each particle (v_x, v_y, v_z). 
        Shape is (N, 3) where there are three velocities for each particle. 
        
    delta_t : float
        The time step used for each step of the simulation, in seconds.
        
    total_t : float
        The total time to evolve the system, in seconds 
    
    Returns
    -------
    times : a 1D numpy array of times
    
    positions : a 3D numpy array of positions (n_times x n_particles x n_positions)
    
    velocities : a 3D numpy array of velocities (n_times x n_particles x n_positions
    '''
    
    # Create an array of times 
    times = np.arange(0, total_t, delta_t)
    
    # Create position and velocity lists
    positions = [] 
    velocities = []
    
    # Set current position and velocity
    currentPositions = initial_positions
    currentVelocities = initial_velocities
    
    for i in times:
        #update particles
        particles = updateParticles(masses, currentPositions, currentVelocities, delta_t)
        
        #replace current position and velocity
        currentPositions = particles[0]
        currentVelocities = particles[1]
        
        # add new positions to arrays
        positions.append(currentPositions)
        velocities.append(currentVelocities)
        
        #return times, positions, and velocities
    return times, np.array(positions), np.array(velocities)