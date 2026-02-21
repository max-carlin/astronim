import numpy as np
from astronim.utils.forces import calculateForceVectors


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

