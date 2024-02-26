from PointCloudVisualisation.Visualisation import read_xyz
import numpy as np
from PointCloudVisualisation.tools import find_boundary_blocks, get_hull_points, find_min_to_point\
    , define_o3d_pc, compute_convex_hull, find_neighbours
from PointCloudVisualisation.const import point_cloud_dir, noise_pc_dir, neighbour_color, target_color, displacement_map_noise_dir


def evaluate_hulls(points_dir, noise=True):
    """
    Given a point cloud with labels indicating the object each point belongs to, the function finds the centroid of each
    object, the local indices of each object's convex hull and the global indices of all the points belonging to the same
    object.
    """
    point_cloud, id_list = read_xyz(points_dir, delimiter=';', noise=noise)
    # Centroid of blocks in the same order as block id. i.e. The 0th element is the centroid of
    # the block with block id of 0.
    centroid_list, hull_index_list, bp_index_list = [], [], []

    # Evaluate the convex hull of the point clouds belonging to each block.
    for b_id in range(max(id_list) - 1):
        # Extract the points of each block
        block_index = np.flatnonzero(id_list == b_id)  # Indices of all the points belonging to a single block.
        points_block = point_cloud[block_index]
        bp_index_list.append(block_index)

        # Evaluate the centroid of the points
        centroid_block = np.mean(points_block, axis=0)
        centroid_list.append(centroid_block)

        # Generate the entire wall for visualisation
        pcd_block = define_o3d_pc(points_block)
        hull_ls, index_list = compute_convex_hull(pcd_block)
        hull_index_list.append(index_list)

    centroid_list = np.array(centroid_list)

    return centroid_list, hull_index_list, bp_index_list, point_cloud, id_list


def find_correspondence(block_id, centroid_list, hull_index_list, bp_index_list, point_cloud, id_list, num=8):
    """
    This function finds the indices of correspondence pairs in the block whose block id is given.
    The result is  a list with dimensions - num_neighbours X 2.
    """

    # Point correspondence between target and neighbouring blocks
    # Find the neighbouring blocks.
    distances, neighbour_indices_all = find_neighbours(centroid_list, centroid_list, num=num+1)

    # Find the correspondence in each neighbour (exclude the block itself)
    neighbour_indices = neighbour_indices_all[block_id]
    correspondence_index_list = []
    target_hull = get_hull_points(block_id, id_list, point_cloud, hull_index_list)
    for neighbour_index in neighbour_indices[1:]:
        # Extract the coordinates of all the convex hull points of the target and the neighbour
        neighbour_hull = get_hull_points(neighbour_index, id_list, point_cloud, hull_index_list)
        distance_list, s_index_list = [], []
        for i, point in enumerate(target_hull):
            # Find the shortest distance between the point in the target convex hull and the entire neighbour convex
            # hull.
            s_distance, s_index = find_min_to_point(point, neighbour_hull)
            distance_list.append(s_distance)
            s_index_list.append(s_index)
        # Find the index of the correspondence pair in the target and neighbouring convex hull
        correspondence_index_target = np.argmin(np.array(distance_list))
        correspondence_index_neighbour = s_index_list[correspondence_index_target]

        # Store the global indices of the correspondence points
        correspondence_index_list.append(
            [bp_index_list[block_id][hull_index_list[block_id][correspondence_index_target]],
             bp_index_list[neighbour_index][hull_index_list[neighbour_index][correspondence_index_neighbour]]])

    return correspondence_index_list


if __name__ == '__main__':
    centroid_list1, hull_index_list1, bp_index_list1, point_cloud1, id_list1 = evaluate_hulls(noise_pc_dir)
    c_list = find_correspondence(610, centroid_list1, hull_index_list1, bp_index_list1, point_cloud1, id_list1, num=8)
    print(c_list)