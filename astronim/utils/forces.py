import numpy as np


def forceMagnitude(mass_i, mass_j, separation):
    """
        Compute magnitude of gravitational force between two particles.

        Parameters
        ----------
        mass_i, mass_j : float
            Particle masses in kg

        separation : float
            Particle separation (distance between particles) in meters

        Returns
        -------
        force : float
            Gravitational force between particles in Newtons

        Example
        -------
            mass_earth = 6.0e24     # kg
            mass_person = 70.0      # kg
            radius_earth = 6.4e6    # meters
            print(magnitudeOfForce(mass_earth, mass_person, radius_earth))

            Output: 683.935546875
    """
    grav_const = 6.67e-11  # m3 kg-1 s-2
    return grav_const * mass_i * mass_j / separation**2  # Newtons


def magnitude(vector):
    """
        Compute magnitude of any vector with an arbitrary number of elements.

        Parameters
        ----------
        vector : numpy array
            Any vector

        Returns
        -------
        magnitude : float
            The magnitude of that vector.

        Example
        -------
            print(magnitude(np.array([3.0, 4.0, 0.0])))

            Output: 5.0
    """
    return np.sqrt(np.sum(vector**2))


def unitDirectionVector(position_i, position_j):
    """
        Create unit direction vector from position_i to position_j.

        Parameters
        ----------
        position_i : numpy array
            Vector of the start position

        position_j : numpy array
            vector of the end position

        Returns
        -------
        unit direction vector : one numpy array (same size input vectors)
            The unit direction vector from position_i toward position_j

        Example
        -------
            start_position = np.array([3.0, 2.0, 5.0])
            end_position = np.array([1.0, -4.0, 8.0])
            print(unitDirectionVector(start_position, end_position))

            Output: [-0.28571429, -0.85714286, 0.42857143]
    """
    # Calculate the separation between the two vectors
    separation = position_j - position_i

    # Divide vector components by vector magnitude to make unit vector
    return separation / magnitude(separation)


def forceVector(mass_i, mass_j, position_i, position_j):
    """
        Compute gravitational force vector exerted on particle i by particle j.

        mass_i is the mass for the particle at position_i.
        mass_j is the mass for the particle at position_j.

        Parameters
        ----------
        mass_i : float
            Particle mass in kg

        mass_j : float
            Particle mass in kg

        position_i : numpy array
            Particle position in cartesian coordinates in meters

        position_j : numpy array
            Particle position in cartesian coordinates in meters

        Returns
        -------
        force vector : numpy array
            Components of gravitational force vector in Newtons

        Example
        -------
            mass_earth = 6.0e24     # kg
            mass_person = 70.0      # kg
            radius_earth = 6.4e6 # m
            center_earth = np.array([0, 0, 0])
            surface_earth = np.array([0, 0, 1]) * radius_earth
            print(forceVector(mass_earth, mass_person, center_earth, surface_earth))

            Output: [0. 0. 683.93554688]
    """
    # Compute the magnitude of the distance between positions in meters
    distance = magnitude(position_i - position_j)

    # Compute the magnitude of the force in Newtons
    force = forceMagnitude(mass_i, mass_j, distance)

    # Calculate the unit direction vector of the force (unitless)
    direction = unitDirectionVector(position_i, position_j)

    # A numpy array with units of Newtons
    return force * direction

def calculateForceVectors(masses, positions):
    """
        Compute net gravitational force vectors on particles
        given a list of masses and positions for all of them.

        Parameters
        ----------
        masses : list (or 1D numpy array) of floats
            Particle masses in kg

        positions : list (or numpy array) of 3-element numpy arrays
            Particle positions in cartesian coordinates, in meters,
            in the same order as the masses are listed.

            Each element in the list (a single particle's position)
            should be a 3-element numpy array, referring to its
            X, Y, Z position.

        Returns
        -------
        force vectors : list of 3-element numpy arrays
            A list containing the net force vectors for each particle.
            Each element in the list is a 3-element numpy array that
            represents the net 3D force acting on a particle after summing
            over the individual force vectors induced by every other particle.

        Example
        -------
            meters_per_au = 1.496e+11
            masses = [1.0e24, 40.0e24, 50.0e24, 30.0e24, 2.0e24]
            positions = [np.array([ 0.5,  2.6,  0.05]) * meters_per_au,
                         np.array([ 0.8,  9.1,  0.10]) * meters_per_au,
                         np.array([-4.1, -2.4,  0.80]) * meters_per_au,
                         np.array([10.7,  3.7,  0.00]) * meters_per_au,
                         np.array([-2.0, -1.9, -0.40]) * meters_per_au]
            # Calculate and print the force vectors for all particles
            the_forces = calculateForceVectors(masses, positions)

            print("{:>10} | {:>10} | {:>10} | {:>10}".format("Particle", "Fx", "Fy", "Fz"))
            print("{:>10} | {:>10} | {:>10} | {:>10}".format("(#)", "(N)", "(N)", "(N)"))
            print("-" * 49)

            for index in range(len(the_forces)):
                force_x, force_y, force_z = the_forces[index]
                print("{:10.0f} | {:10.1e} | {:10.1e} | {:10.1e}".format(index, force_x, force_y, force_z))

            Output: particle |         Fx |         Fy |         Fz
                         (#) |        (N) |        (N) |        (N)
                    -------------------------------------------------
                           0 |   -1.3e+15 |    3.8e+14 |    3.5e+14
                           1 |    9.2e+15 |   -5.3e+16 |    1.8e+15
                           2 |    7.5e+16 |    5.4e+16 |   -2.7e+16
                           3 |   -4.2e+16 |    6.4e+15 |    1.1e+15
                           4 |   -4.0e+16 |   -7.5e+15 |    2.4e+16
        """
    # How many particles are there?
    num_of_particles = len(positions)

    # Create an empty list to be filled with force vectors
    force_vectors = []

    # Loop over particles for which we want the force vector
    for particle_i in range(num_of_particles):
        # Create a force vector with all three elements as zero
        force_vector = np.zeros(3)

        # Loop over all the particles we need to include in the force sum
        for particle_j in range(num_of_particles):
            # As long as particles i and j are not the same...
            if particle_j != particle_i:
                # Add in the force vector of particle j acting on particle i
                force_vector += forceVector(masses[particle_i], masses[particle_j], positions[particle_i], positions[particle_j])

        # Append this force vector into the list of force vectors
        force_vectors.append(force_vector)

    # Return the list of force vectors out of the function
    return force_vectors

def test():
    """ This function tests the force calculations. """

    print("This test function should produce the following output:")
    print()
    print("  particle |         Fx |         Fy |         Fz")
    print("       (#) |        (N) |        (N) |        (N)")
    print("-------------------------------------------------")
    print("         0 |   -1.3e+15 |    3.8e+14 |    3.5e+14")
    print("         1 |    9.2e+15 |   -5.3e+16 |    1.8e+15")
    print("         2 |    7.5e+16 |    5.4e+16 |   -2.7e+16")
    print("         3 |   -4.2e+16 |    6.4e+15 |    1.1e+15")
    print("         4 |   -4.0e+16 |   -7.5e+15 |    2.4e+16")

    print("Here is what it outputs after running calculateForceVectors:")
    print()

    meters_per_au = 1.496e11
    masses = [1.0e24, 40.0e24, 50.0e24, 30.0e24, 2.0e24]
    positions = [np.array([ 0.5,  2.6,  0.05]) * meters_per_au,
                 np.array([ 0.8,  9.1,  0.10]) * meters_per_au,
                 np.array([-4.1, -2.4,  0.80]) * meters_per_au,
                 np.array([10.7,  3.7,  0.00]) * meters_per_au,
                 np.array([-2.0, -1.9, -0.40]) * meters_per_au]

    # Calculate and print the force vectors for all particles
    forces = calculateForceVectors(masses, positions)

    print("{:>10} | {:>10} | {:>10} | {:>10}".format("particle", "Fx", "Fy", "Fz"))
    print("{:>10} | {:>10} | {:>10} | {:>10}".format("(#)", "(N)", "(N)", "(N)"))
    print("-" * 49)
    
    for index in range(len(forces)):
        force_x, force_y, force_z = forces[index]
        print("{:10.0f} | {:10.1e} | {:10.1e} | {:10.1e}".format(index, force_x, force_y, force_z))

    print()
    print("If those two tables are the same, it works!")


if __name__ == "__main__":
    test()

