import Const
import numpy as np
from model_specs import brick_dim_num


def generate_dimension(dimensions=brick_dim_num, block_num=True):
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
    z_bot = [z_lim[0]-Const.brick_dims_UK[2], z_lim[0]]
    z_top = [z_lim[1], z_lim[1]+Const.brick_dims_UK[2]]
    return f"block create brick {x_lim[0]:.10f} {x_lim[1]:.10f} {y_lim[0]:.10f} {y_lim[1]:.10f} {z_bot[0]:.10f} {z_bot[1]:.10f} \n" \
           f"block create brick {x_lim[0]:.10f} {x_lim[1]:.10f} {y_lim[0]:.10f} {y_lim[1]:.10f} {z_top[0]:.10f} {z_top[1]:.10f} \n", \
           [z_bot, z_top]


def vector_rotation(alpha, beta, gamma):
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
    return np.matmul(vector_rotation(angles[0], angles[1], angles[2]), translation) + translation


if __name__ == '__main__':
    a = np.array([[1, 2, 4],
                 [5, 2, 1],
                 [1, 1, 1]])
    b = np.array([5, 6, 9])
    print(np.matmul(a, b))




