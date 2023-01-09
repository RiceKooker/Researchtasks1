from utils import generate_dimension, get_absolute_displacement

# Generate wall dimensions
brick_dim_num = [2, 2, 2]
# -------------------------------------------------------------------------------------------------
wall_dim = generate_dimension(brick_dim_num)
# -------------------------------------------------------------------------------------------------

# 3DEC setting
large_strain = 'on'
damping = 0.8

# Material properties and contact law
brick_density = 2548
contact_model = 'mohr'
normal_stiffness = 5.87e9
shear_stiffness = 2.45e9
cohesion = 0
tension = 0
friction = 31.8
fric_res = 31.8

# Loading specs
rotations = [0, 0, 0]
displacement_percentage = [i/100 for i in [5, 5, -2]]
# -------------------------------------------------------------------------------------------------
displacements = get_absolute_displacement(displacement_percentage, wall_dim)
# -------------------------------------------------------------------------------------------------
velocity_max = 0.01

# Solving specs
num_cycle = 3000

# Storage specs
file_name = 'mytest'
geometry_file_name1 = 'geo_info1.txt'
geometry_file_name2 = 'geo_info2.txt'

