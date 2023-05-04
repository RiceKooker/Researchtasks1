import Const
import numpy as np
import random
import math


def print_list(a_list):
    """This function is for printing lists in 3DEC script. The precision can be specified."""
    str_out = ''
    for i, num in enumerate(a_list):
        str_add = f'{num:.8f}, '
        if i+1 == len(a_list):
            str_add = f'{num:.8f}'
        str_out += str_add
    return str_out


def print_string_list(str_list):
    str_out = ''
    for i, value in enumerate(str_list):
        str_add = f"'{value}', "
        if i+1 == len(str_list):
            str_add = f"'{value}'"
        str_out += str_add
    return str_out

def print_location(a_list):
    """This function is for printing lists in 3DEC script. The precision can be specified."""
    str_out = ''
    for i, num in enumerate(a_list):
        str_add = f'{num:.8f} '
        if i+1 == len(a_list):
            str_add = f'{num:.8f}'
        str_out += str_add
    return str_out



def load_description(displacement, height):
    """
    This function returns the relative displacements and rotations in degree for displaying loading conditions in the code.
    :param displacement: prescribed displacement, 6 dimensional
    :param height: height of the brick wall - exclusive of boundary blocks
    :return:
    """
    description = []
    for i in range(3):
        description.append(100*displacement[i]/height) # Calculate the percentage displacement
    for i in range(3, 6):
        description.append(math.degrees(displacement[i]))
    return description


def stop_detection(displacement):
    """
    This function chooses the detection axis with a given displacement vector.
    It only looks at the translational displacements.
    If the prescribed displacement at the y axis is the largest, the function would allow
    the 3DEC programme to stop by constantly detecting the displacement in the y direction.
    :param displacement: 6 dimensional displacement vector
    :return:
    """
    displacement = displacement[0:3]
    max_magnitude = 0
    max_index = 0
    for i, dis_axis in enumerate(displacement):
        if np.abs(dis_axis) > max_magnitude:
            max_magnitude = np.abs(dis_axis)
            max_index = i
    return max_index, Const.axis_dict[max_index]


def sample_displacements(dims, thresholds):
    """
    This function randomly generates the displacement vector based on the wall dimensions and prescribed thresholds.
    The displacement in the z direction is forced to be 0.
    The sampling process follows a uniform distribution.
    :param dims: 3 dimensional vector indicating the dimensions of the wall
    :param thresholds: 3 dimensional vectors indicating the maximum displacements that can be possible sampled in each direction.
    :return:
    """
    displacements = []
    max_dis = [thresholds[0]*dims[2], thresholds[1]*dims[2], 0]
    for max_dis_axis in max_dis:
        displacements.append(random.uniform(-max_dis_axis, max_dis_axis))
    displacements[2] = 0
    return displacements, max_dis


def sample_rotations(dims, displacements, max_dis):
    """
    This function randomly generates the rotation vectors based on the displacement vector and the dimensions of the wall.
    :param dims:
    :param displacements:
    :param max_dis:
    :return:
    """
    theta_x_max = 1.5*displacements[1]/dims[2]
    theta_y_max = 1.5*displacements[0]/dims[2]
    theta_z_max = max_dis[1]/(0.5*dims[0])
    rotations = [random.uniform(-theta_x_max, theta_x_max), random.uniform(-theta_y_max, theta_y_max), random.uniform(-theta_z_max, theta_z_max)]
    return rotations


def prescribed_displacement(dims, thresholds):
    dis_vec, max_dis = sample_displacements(dims, thresholds)
    rot_vec = sample_rotations(dims, dis_vec, max_dis)
    return dis_vec + rot_vec


def get_velocities(displacements, velocity, dims):
    """
    This function combines the displacement and rotation vectors and scales the resulting vector such that the largest entry
    has a magnitude equal to the velocity specified.
    :param displacements: total displacement vector - 6 dimensional
    :param velocity: magnitude of the largest applied velocity in all 6 degrees of freedom.
    :param dims: dimensions of the wall
    :return:
    """
    dis = displacements[0:3]
    rot = displacements[3:6]
    rotation_dis = [rot[0]*dims[1], rot[1]*dims[0], rot[2]*dims[0]]
    D = list(dis) + list(rot)
    D_ = list(dis) + list(rotation_dis)
    scale = max([abs(i) for i in D_]) / abs(velocity)
    return [i / scale for i in D]


def generate_dimension(dimensions, block_num=True):
    """
    This function generates the required dimensions to generate a wall if number of bricks is given.
    :param dimensions:
    :param block_num:
    :return:
    """
    if block_num:
        wall_dim = np.multiply(np.array(dimensions), np.array(Const.brick_dims_UK))
    else:
        wall_dim = dimensions
    return wall_dim


# def create_geometry(dimensions=Const.brick_dim_num, block_num=True):
#     if block_num:
#         wall_dim = np.multiply(np.array(dimensions), np.array(Const.brick_dims_UK))
#     else:
#         wall_dim = dimensions
#     wall = BlockWall(wall_dim)
#     return wall.three_DEC_create(), create_boundary_bricks(wall.find_vertices())


def create_boundary_bricks(vertices):
    """
    This function generates the commands needed for the top and bottom boundary blocks.
    And returns the corresponding z coordinates for labeling purposes.
    :param vertices:
    :return:
    """
    x_lim = [vertices[0][0], vertices[3][0]]
    y_lim = [vertices[0][1], vertices[4][1]]
    z_lim = [vertices[0][2], vertices[1][2]]
    z_bot = [z_lim[0] - Const.brick_dims_UK[2], z_lim[0]]
    z_top = [z_lim[1], z_lim[1] + Const.brick_dims_UK[2]]
    return f"block create brick {x_lim[0]:.10f} {x_lim[1]:.10f} {y_lim[0]:.10f} {y_lim[1]:.10f} {z_bot[0]:.10f} {z_bot[1]:.10f} \n" \
           f"block create brick {x_lim[0]:.10f} {x_lim[1]:.10f} {y_lim[0]:.10f} {y_lim[1]:.10f} {z_top[0]:.10f} {z_top[1]:.10f} \n", \
           [z_bot, z_top]


class BricksBoundary:
    """
    This class stores information about top and bottom boundary blocks. Some post processing functions are added to extract detailed information.
    """

    def __init__(self, vertices, thickness=0.1):
        """
        Initialize the limits in x, y, and z direction. The 2 blocks share the same x,y limits but have different z limits.
        :param vertices:
        """
        self.x_lim = [vertices[0][0], vertices[3][0]]
        self.y_lim = [vertices[0][1], vertices[4][1]]
        self.z_lim = [vertices[0][2], vertices[1][2]]

        # Define the thickness of the boundary blocks
        self.z_bot = [self.z_lim[0] - thickness*Const.brick_dims_UK[2], self.z_lim[0]]
        self.z_top = [self.z_lim[1], self.z_lim[1] + thickness*Const.brick_dims_UK[2]]

        # Additional information
        self.top_gp = [(self.x_lim[0] + self.x_lim[1]) * 0.5, (self.y_lim[0] + self.y_lim[1]) * 0.5,
                       self.z_top[1]]  # Coordinates of the top grid point
        self.commands = f"block create brick {self.x_lim[0]:.10f} {self.x_lim[1]:.10f} {self.y_lim[0]:.10f} {self.y_lim[1]:.10f} {self.z_bot[0]:.10f} {self.z_bot[1]:.10f} \n" \
                        f"block create brick {self.x_lim[0]:.10f} {self.x_lim[1]:.10f} {self.y_lim[0]:.10f} {self.y_lim[1]:.10f} {self.z_top[0]:.10f} {self.z_top[1]:.10f} \n"
        self.z_lims = [self.z_bot, self.z_top]



def vector_rotation(alpha, beta, gamma, mode='tait-bryan'):
    """
    This function generates the general rotation matrix in 3D, where alpha, beta, and gamma are the angles to the axis.
    :param alpha:
    :param beta:
    :param gamma:
    :param mode: if mode is yaw, then it uses the more traditional way of defining the angles. If mode is tait-bryan,
    the angles are the intrinsic rotation angles about z, y, x.
    :return:
    """
    if mode == 'yaw':
        temp = alpha
        alpha = gamma
        gamma = temp
    angles = [alpha, beta, gamma]
    sine = np.zeros((3,))
    cosine = np.zeros((3,))
    for i, angle in enumerate(angles):
        sine[i] = np.sin(angle)
        cosine[i] = np.cos(angle)

    R_z = np.array([[cosine[2], -sine[2], 0],
                    [sine[2], cosine[2], 0],
                    [0, 0, 1]])
    R_y = np.array([[cosine[1], 0, sine[1]],
                    [0, 1, 0],
                    [-sine[1], 0, cosine[1]]])
    R_x = np.array([[1, 0, 0],
                    [0, cosine[0], -sine[0]],
                    [0, sine[0], cosine[0]]])
    return np.matmul(R_z, np.matmul(R_y, R_x))


def translate_and_rotate(angles, translation):
    """
    This function returns the displacement vector which consists of rotation and displacement
    :param angles:
    :param translation:
    :return:
    """
    return np.matmul(vector_rotation(angles[0], angles[1], angles[2], mode='yaw'), translation) + translation


def get_absolute_displacement(dis_percent, wall_dims):
    """
    This function returns the absolute displacement based on the percentage displacement and wall dimensions provided.
    :param dis_percent:
    :param wall_dims:
    :return:
    """
    return [i*wall_dims[2] for i in dis_percent ]


if __name__ == '__main__':
    a = [1, 2, 3, 4]
    print(print_list(a))


