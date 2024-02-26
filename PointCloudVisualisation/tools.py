import numpy as np
from Const import brick_dims_UK
import open3d as o3d
import pynanoflann


def find_global_index(c_index, block_id, hull_index_list, bp_index_list):
    return bp_index_list[block_id][hull_index_list[block_id][c_index]]

def single_line_set(point1, point2, color=(1, 0, 0)):
    points = [point1, point2]
    lines = [[0, 1]]
    colors = [color]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def find_neighbours(points, queries, num=9, metric='L2'):
    nn = pynanoflann.KDTree(n_neighbors=num, metric=metric)
    nn.fit(points)
    distances, neighbour_indices_all = nn.kneighbors(queries)
    return distances, neighbour_indices_all


def compute_convex_hull(pcd, color=(0.447, 0.545, 0.741)):
    hull_block, index_list = pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull_block)
    hull_ls.paint_uniform_color(color)
    return hull_ls, index_list


def define_o3d_pc(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def find_min_to_point(target, group):
    """
    This function evaluate the minimum distance between a point and a group of point.
    """
    target = np.array(target)
    group = np.array(group)
    distance = 10000000
    group_index = 0
    for point_index, group_point in enumerate(group):
        distance_ = np.linalg.norm(group_point - target)
        if distance_ < distance:
            distance = distance_
            group_index = point_index
    return distance, group_index


def get_hull_points(block_id_, id_list_, point_cloud_, hull_point_list_):
    block_index_ = np.flatnonzero(id_list_ == block_id_)  # Indices of all the points belonging to a single block.
    target_block_points = point_cloud_[block_index_]
    return target_block_points[hull_point_list_[block_id_]]


def find_max_min(point_list):
    """
    This function finds the maximum and minimum in all dimensions of all the points in the list.
    """
    point_list = np.array(point_list)
    axis = 1
    needTranspose = True
    if len(point_list) != 3:
        axis = 0
        needTranspose = False
    max_coord = np.max(point_list, axis=axis)
    min_coord = np.min(point_list, axis=axis)
    if needTranspose:
        max_coord = max_coord.T
        min_coord = min_coord.T
    return max_coord, min_coord


def find_boundary_blocks(c_list):
    """
    This function determines if a block is at the boundaries or at the corner of a wall.
    :param c_list: list of centroid of all the blocks belonging to the same wall.
    If boundary index is the number of boundaries the block touches.
    """
    max_coord, min_coord = find_max_min(c_list)
    boundary_index_list = []
    range_factor = 0.2
    for centroid in c_list:
        boundary_count = 0
        for axis_value in centroid:
            # Approximate search to see if a block sits at any of the boundaries.
            for j, extreme_value in enumerate(zip(max_coord, min_coord)):
                max_value, min_value = extreme_value
                if max_value - brick_dims_UK[j] * range_factor <= axis_value <= max_value + brick_dims_UK[j] * range_factor:
                    boundary_count += 1
                    break
                if min_value - brick_dims_UK[j] * range_factor <= axis_value <= min_value + brick_dims_UK[j] * range_factor:
                    boundary_count += 1
                    break
        boundary_index_list.append(boundary_count)
    return boundary_index_list