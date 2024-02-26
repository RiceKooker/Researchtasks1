from PointCloudVisualisation.Visualisation import read_xyz
import open3d as o3d
import numpy as np
from PointCloudVisualisation.tools import find_boundary_blocks, get_hull_points, find_min_to_point\
    , define_o3d_pc, compute_convex_hull, find_neighbours, single_line_set
from PointCloudVisualisation.const import point_cloud_dir, noise_pc_dir, neighbour_color, target_color, displacement_map_noise_dir
import pandas as pd


if __name__ == '__main__':
    # Read the point cloud.
    point_cloud, id_list = read_xyz(point_cloud_dir, delimiter=',')
    point_cloud, id_list = read_xyz(noise_pc_dir, delimiter=';', noise=True)
    # Centroid of blocks in the same order as block id. i.e. The 0th element is the centroid of
    # the block with block id of 0.
    geometry_list, centroid_list, hull_index_list, bp_index_list = [], [], [], []

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
        geometry_list.append(hull_ls)
        hull_index_list.append(index_list)

    # Identify boundary blocks
    centroid_list = np.array(centroid_list)
    boundary_indices = find_boundary_blocks(centroid_list)

    # Point correspondence between target and neighbouring blocks
    # Find the neighbouring blocks.
    distances, neighbour_indices_all = find_neighbours(centroid_list, centroid_list, num=9)

    block_id = 609
    block_id = 620

    # Visualize the neighbouring blocks
    neighbour_indices = neighbour_indices_all[block_id]
    for neighbour_index in neighbour_indices:
        geometry_list[neighbour_index].paint_uniform_color(neighbour_color)
    geometry_list[block_id].paint_uniform_color(target_color)

    # Find the correspondence in each neighbour (exclude the block itself)
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
        correspondence_line = single_line_set(target_hull[correspondence_index_target], neighbour_hull[correspondence_index_neighbour])
        geometry_list.append(correspondence_line)

        # Store the global indices of the correspondence points
        correspondence_index_list.append([bp_index_list[block_id][hull_index_list[block_id][correspondence_index_target]], bp_index_list[neighbour_index][hull_index_list[neighbour_index][correspondence_index_neighbour]]])
    o3d.visualization.draw_geometries(geometry_list)

    df = pd.read_csv(displacement_map_noise_dir, delimiter=',')
    rel_disp_list = []
    for c_pair in correspondence_index_list:
        target_disp = df.iloc[c_pair[0]]['Displacement']
        neighbour_disp = df.iloc[c_pair[1]]['Displacement']
        rel_disp = target_disp-neighbour_disp
        rel_disp_list.append(rel_disp)
    print(rel_disp_list)






