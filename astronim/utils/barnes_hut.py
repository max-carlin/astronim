import numpy as np
from astronim.utils.constants import AU

THETA = 0.7       # Opening angle for Barnes–Hut
G = 6.67430e-11   # Gravitational constant

# near top of barnes_hut.py
G = 6.67430e-11
# softening length in meters (e.g., 100 pc)
pc = 3.085677581e16
SOFTENING = AU * 0.25 #150       # soften to 2 AU
EPS = SOFTENING
EPS2 = EPS * EPS


class OctreeNode:
    def __init__(self, center, half_size):
        self.center = center          # center of this cube
        self.half_size = half_size    # half-width of cube
        self.mass = 0.0
        self.com = np.zeros(3)        # center of mass
        self.children = [None]*8
        self.is_leaf = True
        self.index = -1               # index of particle if leaf


def insert_particle(node, pos, mass, index):
    """Recursively insert a particle into the octree."""
    if node.is_leaf and node.index == -1:
        node.index = index
        node.mass = mass
        node.com = pos.copy()
        return

    if node.is_leaf and node.index != -1:
        # Subdivide
        old_index = node.index
        node.is_leaf = False
        node.index = -1

        old_pos = node.com.copy()
        old_mass = node.mass

        # Create children
        for i in range(8):
            offset = np.array([ (1 if i&1 else -1),
                                (1 if i&2 else -1),
                                (1 if i&4 else -1)]) * 0.5
            child_center = node.center + offset * node.half_size
            node.children[i] = OctreeNode(child_center, node.half_size * 0.5)

        # Insert old particle
        oct_index = 0
        if old_pos[0] > node.center[0]: oct_index |= 1
        if old_pos[1] > node.center[1]: oct_index |= 2
        if old_pos[2] > node.center[2]: oct_index |= 4
        insert_particle(node.children[oct_index], old_pos, old_mass, old_index)

    # Insert new particle
    oct_index = 0
    if pos[0] > node.center[0]: oct_index |= 1
    if pos[1] > node.center[1]: oct_index |= 2
    if pos[2] > node.center[2]: oct_index |= 4

    insert_particle(node.children[oct_index], pos, mass, index)

    # Update COM + mass
    total_mass = node.mass + mass
    node.com = (node.com * node.mass + pos * mass) / total_mass
    node.mass = total_mass


# def compute_force(node, pos, accel):
#     """Recursively compute force using Barnes–Hut."""
#     if node.mass == 0:
#         return

#     r = node.com - pos
#     dist = np.linalg.norm(r) + 1e-10

#     # Opening criterion
#     if node.is_leaf or (node.half_size / dist) < THETA:
#         accel += G * node.mass * r / dist**3
#         return

#     for child in node.children:
#         if child is not None:
#             compute_force(child, pos, accel)

def compute_force(node, pos, accel):
    if node.mass == 0:
        return

    r = node.com - pos
    dist2 = np.dot(r, r) + 1e-30
    dist = np.sqrt(dist2)

    # A node contributes only when it terminates the recursion: it is a
    # leaf, or far enough away to approximate by its center of mass.
    if node.is_leaf or (node.half_size / dist) < THETA:
        inv_dist3 = 1.0 / ((dist2 + EPS2) * np.sqrt(dist2 + EPS2))
        accel += G * node.mass * r * inv_dist3
        return

    for child in node.children:
        if child is not None:
            compute_force(child, pos, accel)



def fast_updateParticles(masses, positions, velocities, dt):
    N = len(masses)
    masses = np.asarray(masses)
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)

    # Build bounding box
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_size = 0.5 * np.max(maxs - mins) + 1e-10

    # Build octree
    root = OctreeNode(center, half_size)
    for i in range(N):
        insert_particle(root, positions[i], masses[i], i)

    # First half-step acceleration
    accelerations = np.zeros_like(positions)
    for i in range(N):
        compute_force(root, positions[i], accelerations[i])

    # Leapfrog: position update
    positions_new = positions + velocities * dt + 0.5 * accelerations * dt**2

    # Rebuild octree at new positions
    mins = positions_new.min(axis=0)
    maxs = positions_new.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_size = 0.5 * np.max(maxs - mins) + 1e-10
    root_new = OctreeNode(center, half_size)
    for i in range(N):
        insert_particle(root_new, positions_new[i], masses[i], i)

    # Second half-step acceleration
    accelerations_new = np.zeros_like(positions)
    for i in range(N):
        compute_force(root_new, positions_new[i], accelerations_new[i])

    # Leapfrog: velocity update
    velocities_new = velocities + 0.5 * (accelerations + accelerations_new) * dt

    return positions_new, velocities_new
