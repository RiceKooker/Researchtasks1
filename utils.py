import Const
import numpy as np


def get_velocities(displacements, rotations, velocity):
    """
    This function combines the displacement and rotation vectors and scales the resulting vector such that the largest entry
    has a magnitude equal to the velocity specified.
    :param displacements:
    :param rotations:
    :param velocity: magnitude of the largest applied velocity in all 6 degrees of freedom.
    :return:
    """
    D = list(displacements) + list(rotations)
    scale = max([abs(i) for i in D]) / abs(velocity)
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

    def __init__(self, vertices):
        """
        Initialize the limits in x, y, and z direction. The 2 blocks share the same x,y limits but have different z limits.
        :param vertices:
        """
        self.x_lim = [vertices[0][0], vertices[3][0]]
        self.y_lim = [vertices[0][1], vertices[4][1]]
        self.z_lim = [vertices[0][2], vertices[1][2]]
        self.z_bot = [self.z_lim[0] - Const.brick_dims_UK[2], self.z_lim[0]]
        self.z_top = [self.z_lim[1], self.z_lim[1] + Const.brick_dims_UK[2]]

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
    return np.multiply(np.array(dis_percent), np.array(wall_dims))


if __name__ == '__main__':
    a = np.array([[1, 2, 4],
                  [5, 2, 1],
                  [1, 1, 1]])
    b = np.array([1, 2, 3])
    c = np.array([-4, 5, -6])
    print(get_velocities(b, c, 0.5))
