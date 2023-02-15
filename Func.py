import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from Const import side_dict, grid_point_reader, vertex_index_dict
import pandas as pd


def check_vertex(pos, lims):
    """
    This function checks if a grid point is one of the vertices of a block.
    :param pos: coordinates of the grid point
    :param lims: extreme coordinate values of the block - [x_min, y_min, z_min, x_max, y_max, z_max]
    :return: state variable that determine if this grid point is a vertex
    """
    is_vertex = False
    extreme_index = 0
    num_dims = len(pos)
    vertex_index = []  # An index used to identify the conventional order of the vertex
    for i, pos_axis in enumerate(pos):
        axis_lim = [lims[i], lims[i+num_dims]]
        if pos_axis in axis_lim:
            extreme_index += 1
            v_i = 0
            if pos_axis == axis_lim[1]:
                v_i = 1
            vertex_index.append(v_i)
    if extreme_index == num_dims:
        is_vertex = True
    if len(vertex_index) < num_dims:
        return is_vertex, None
    else:
        return is_vertex, vertex_index


def check_vert_df(df):
    """
    This function checks if any of the grid points in the dataframe are vertices.
    :param df: dataframe of grid points belong to the same block.
    :return: a boolean array indicating which grid points are the vertices and a vertex index list.
    """
    gp_position = df[grid_point_reader['Position_labels']].values
    lim_lower, lim_upper = find_lims_from_gps(gp_position)
    vertex_bool_list = []
    vertex_index_list = []
    for gp_pos in gp_position:
        is_vertex, vertex_index = check_vertex(gp_pos, (lim_lower+lim_upper))
        vertex_bool_list.append(is_vertex)
        if vertex_index is not None:
            vertex_index_list.append(vertex_index)
    return vertex_bool_list, vertex_index_list


def get_vert_df(df_block):
    """
    This function returns the dataframe with ordered vertices, given the dataframe of all grid points of a block.
    """
    vertex_bool_list, vertex_index_list = check_vert_df(df_block)
    df_vertices = pd.DataFrame(df_block.loc[vertex_bool_list].reset_index(drop=True))
    new_df_index = []
    for key, value in vertex_index_dict.items():
        for i, vertex_index in enumerate(vertex_index_list):
            if value == vertex_index:
                new_df_index.append(i)
    df_vertices = df_vertices.reindex(new_df_index)
    return df_vertices


def find_lims_from_gps(gps):
    """
    This function finds the limits of the block in the x,y and z directions, it assumes the rotation is zero.
    :param gps: list of grid point coordinates [number of grid points, 3] - 3 dimensional coordinates
    :return: 3 dimensional upper bound list, and lower bound list
    """
    # Initialize the max and min values in the 3 axis
    a_large_num = 100000
    a_small_num = 0
    mins = [a_large_num for i in range(len(gps[0]))]
    maxs = [a_small_num for i in range(len(gps[0]))]
    for gp in gps:
        for i, axis_value in enumerate(gp):
            if axis_value > maxs[i]:
                maxs[i] = axis_value
            if axis_value < mins[i]:
                mins[i] = axis_value
    return mins, maxs


def find_lims_from_points(points):
    """
    This function finds the limits of all the points in 'points' in the x,y and z directions.
    :param points: list of point coordinates [number of points, 3] - 3 dimensional coordinates
    :return: 3 dimensional upper bound list, and lower bound list
    """
    # Initialize the max and min values in the 3 axis
    a_large_num = 100000
    a_small_num = 0
    mins = [a_large_num for i in range(len(points[0]))]
    maxs = [a_small_num for i in range(len(points[0]))]
    for point in points:
        for i, axis_value in enumerate(point):
            if axis_value > maxs[i]:
                maxs[i] = axis_value
            if axis_value < mins[i]:
                mins[i] = axis_value
    return mins, maxs


def find_vertices(CoM, Dims):
    """
    This function returns the coordinates of the 8 vertices of a rectangle given the center of mass and the dimensions.
    z axis is taken to be the height axis.
    :param CoM: Center of mass in 3-dimensional vectors
    :param Dims: Dimensions in 3-dimensional vectors
    :return:
    """
    vertices = np.zeros([8, 3])
    half_Dims = 0.5*np.array(Dims)
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


def find_vertices_from_max(min_dims, max_dims):
    x_min = min_dims[0]
    x_max = max_dims[0]
    y_min = min_dims[1]
    y_max = max_dims[1]
    z_min = min_dims[2]
    z_max = max_dims[2]
    vertices = np.zeros([8, 3])
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


def draw_blocks(block_list, wall_vert=None):
    """
    This function draws all the block objects given in the block_list.
    :param block_list:
    :param wall_vert: optional. This is to highlight the corners of the wall.
    :return:
    """
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    if wall_vert is not None:
        # This is to highlight the vertices of the wall.
        ax.scatter3D(wall_vert[:, 0], wall_vert[:, 1], wall_vert[:, 2], marker='^', c='#010812', s=50)
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


def draw_blocks2(block_list, wall_vert=None):
    """
    This function draws all the block objects given in the block_list.
    :param block_list:
    :param wall_vert: optional. This is to highlight the corners of the wall.
    :return:
    """
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    if wall_vert is not None:
        # This is to highlight the vertices of the wall.
        ax.scatter3D(wall_vert[:, 0], wall_vert[:, 1], wall_vert[:, 2], marker='^', c='#010812', s=50)
    for block in block_list:
        vertices = transform_vertices(block.vertices)
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


def verts_to_Dims_CoM(vertices):
    Dims = [vertices[3][0]-vertices[0][0], vertices[4][1]-vertices[0][1], vertices[1][2]-vertices[0][2]]
    Dims = np.array(Dims)
    CoM = np.array(vertices[0]) + 0.5*Dims
    return CoM, Dims


def find_block_CoM(vertices):
    n = 0
    CoM = 0
    for vertex in vertices:
        CoM += vertex
        n += 1
    return CoM / n


def find_row_CoM(block_list):
    CoM = 0
    v = 0
    for block in block_list:
        CoM += block.CoM * block.volume
        v += block.volume
    return CoM / v


def transform_vertex_3DEC(vertices):
    temp = vertices.copy()
    temp[1] = vertices[4]
    temp[2] = vertices[5]
    temp[3] = vertices[1]
    temp[4] = vertices[3]
    temp[5] = vertices[6]
    temp[6] = vertices[7]
    temp[7] = vertices[2]
    return temp



def dependency_test():
    print('Test done!')

