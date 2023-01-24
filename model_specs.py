from utils import generate_dimension, get_absolute_displacement, prescribed_displacement
from utils import load_description
import numpy as np
import math

# Generate wall dimensions
brick_dim_num = [8, 2, 20]
# -------------------------------------------------------------------------------------------------
wall_dim = generate_dimension(brick_dim_num)
# -------------------------------------------------------------------------------------------------

# 3DEC setting
large_strain = 'on'
damping = 0.8

# Material properties and contact law
brick_density = 2000
E = 4e8
G = 0.15*E
friction_coef = 0.5
contact_model = 'mohr'
normal_stiffness = 5.87e9
shear_stiffness = 2.45e9
cohesion = 0.39e6
tension = 0.2e6
friction = math.degrees(math.atan(friction_coef))
# friction = 31.8
fric_res = friction

# Loading specs
# rotations = [0.25*math.pi, 0, 0]
# displacement_percentage = [i/100 for i in [10, 0, 0]]
# -------------------------------------------------------------------------------------------------
# displacements = get_absolute_displacement(displacement_percentage, wall_dim)
# displacements = np.concatenate((np.array(displacements), np.array(rotations)))  # Activate if not sampling
# -------------------------------------------------------------------------------------------------

# Solving specs
num_cycle = 3000
vertical_pressure = -1e6

# Storage specs
file_name = 'mytest'
geometry_file_name1 = 'geo_info1.txt'
geometry_file_name2 = 'geo_info2.txt'

# Processing
# if sampling
velocity_max = 0.01
threshold = [0.015, 0.01, 0]
displacements = prescribed_displacement(wall_dim, threshold)

# displacements = np.concatenate((np.array(displacements), np.array(rotations)))  # Activate if not sampling
load_descriptions = load_description(displacements, wall_dim[2])

# Boundary properties
boundary_spec = {
    'normal_stiffness': 5e10,
    'shear_stiffness': 5e10,
    'cohesion': 0.39e6,
    'tension': 2e7,
    'friction': 40,
    'fric_res': 40
}

