import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from Const import side_dict


def find_vertices(CoM, Dims):
    """
    This function returns the coordinates of the 8 vertices of a rectangle given the center of mass and the dimensions.
    z axis is taken to be the height axis.
    :param CoM: Center of mass in 3-dimensional vectors
    :param Dims: Dimensions in 3-dimensional vectors
    :return:
    """
    vertices = np.zeros([8, 3])
    half_Dims = 0.5*Dims
    x_min = CoM[0] - half_Dims[0]
    x_max = CoM[0] + half_Dims[0]
    y_min = CoM[1] - half_Dims[1]
    y_max = CoM[1] + half_Dims[1]
    z_min = CoM[2] - half_Dims[2]
    z_max = CoM[2] + half_Dims[2]
    vertices[0] = [x_min, y_min, z_min]
    vertices[1] = [x_min, y_min, z_max]
    vertices[2] = [x_max, y_min, z_max]
    vertices[3] = [x_max, y_min, z_min]
    vertices[4] = [x_min, y_max, z_min]
    vertices[5] = [x_min, y_max, z_max]
    vertices[6] = [x_max, y_max, z_max]
    vertices[7] = [x_max, y_max, z_min]
    return vertices


def transform_vertices(vertices):
    """
    This function transforms the order of the vertices so that they can be visualised easily.
    :param vertices:
    :return:
    """
    vertices = np.array(vertices)
    new_vertices = np.array(vertices)
    new_vertices[1] = vertices[3]
    new_vertices[2] = vertices[7]
    new_vertices[3] = vertices[4]
    new_vertices[4] = vertices[1]
    new_vertices[5] = vertices[2]
    new_vertices[7] = vertices[5]
    return new_vertices


def draw_blocks(block_list):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    for block in block_list:
        vertices = transform_vertices(block.Vertices)
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        # vertices = [list(zip(x, y, z))]
        ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        verts = [[vertices[0], vertices[1], vertices[2], vertices[3]],
                 [vertices[4], vertices[5], vertices[6], vertices[7]],
                 [vertices[0], vertices[1], vertices[5], vertices[4]],
                 [vertices[2], vertices[3], vertices[7], vertices[6]],
                 [vertices[1], vertices[2], vertices[6], vertices[5]],
                 [vertices[4], vertices[7], vertices[3], vertices[0]]]
        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    # ax.get_zaxis().set_ticks([])
    plt.show()


def find_axis_and_direction(side):
    side_info = side_dict[side]
    axis = side_info[0]
    direction = side_info[1]
    return axis, direction


def dependency_test():
    print('Test done!')

