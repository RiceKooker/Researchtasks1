from utils import generate_dimension, get_absolute_displacement, prescribed_displacement
from utils import load_description
import numpy as np
import math
from Const import brick_dims_UK
import utils_val

# Generate wall dimensions
brick_dim_num = [10, 2, 8]
# -------------------------------------------------------------------------------------------------
# wall_dim = generate_dimension(brick_dim_num)
wall_dim = [1, 0.25, 1.35]
# -------------------------------------------------------------------------------------------------

# 3DEC setting
large_strain = 'off'
damping = 0.8

# Material properties and contact law
joint_spacing = brick_dims_UK[2]
brick_density = 1652
E = 1.491e9
G = 0.5e9
friction_coef = 0.58
contact_model = 'example'
normal_stiffness = E/joint_spacing
shear_stiffness = G/joint_spacing
cohesion = 0.23e6
cohesion_res = 0
tension = 0.04e6
friction = math.degrees(math.atan(friction_coef))
# friction = 31.8
fric_res = friction
Gf_tension = 10
Gf_compression = 40e3
Gf_shear = 50
fc = 6.2e6

# -------------------------------------------------------------------------------------------------
# Loading specs, user specified
rotations = [math.radians(0.017), math.radians(-0.416), math.radians(0.091)]
displacement_percentage = [i/100 for i in [0.7, 0.85, 0]]
displacements = get_absolute_displacement(displacement_percentage, wall_dim)
displacements = np.concatenate((np.array(displacements), np.array(rotations)))  # Activate if not sampling
# -------------------------------------------------------------------------------------------------

# Solving specs
num_cycle = 1000
vertical_pressure = -0.6e6

# Cyclic test specs
disp_lim = 7e-3
increment = 0.25e-3
cyclic_specs = {
    'load_direction': 'x',
    'disp_amps': utils_val.displacement_amplitude(disp_lim, increment),
    'velocity': 0.01,
    'n_cycle': 100,
    'load_group': 'TOP'
}

# Storage specs
file_name = 'DIANA_val_test'
geometry_file_name1 = 'DIANA_val_test_geo1.txt'
geometry_file_name2 = 'geo_info2.txt'
gp_file_name = 'Gp_info_DIANA_val'

# -------------------------------------------------------------------------------------------------
# Loading specs, sampling
velocity_max = 1
threshold = [0.015, 0.01, 0]
# displacements = prescribed_displacement(wall_dim, threshold)
# -------------------------------------------------------------------------------------------------

# Text description generation
load_descriptions = load_description(displacements, wall_dim[2])

# Boundary properties
boundary_spec = {
    'normal_stiffness': normal_stiffness,
    'shear_stiffness': shear_stiffness,
    'cohesion': cohesion*100,
    'tension': tension*100,
    'friction': 44,
    'fric_res': 44,
    'cohesion_res': cohesion_res,
    'Gf_tension': Gf_tension*100,
    'Gf_compression': Gf_compression*100,
    'Gf_shear': Gf_shear*100,
    'fc': fc*100
}

# 3DEC variables
group_name_top = 'TOP'
group_name_bot = 'BOT'
group_name_body = 'None'
