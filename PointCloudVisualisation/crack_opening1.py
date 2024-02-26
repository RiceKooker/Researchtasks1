from PointCloudVisualisation.Visualisation import read_xyz, single_line_set
import open3d as o3d
import numpy as np
import pynanoflann
from Const import brick_dims_UK

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
    for i, centroid in enumerate(c_list):
        boundary_count = 0
        for axis_value in centroid:
            # Approximate search to see if a block sits at any of the boundaries.
            for j, extreme_value in enumerate(zip(max_coord, min_coord)):
                max_value, min_value = extreme_value
                if max_value-brick_dims_UK[j]*range_factor <= axis_value <= max_value+brick_dims_UK[j]*range_factor:
                    boundary_count += 1
                    break
                if min_value-brick_dims_UK[j]*range_factor <= axis_value <= min_value+brick_dims_UK[j]*range_factor:
                    boundary_count += 1
                    break
        boundary_index_list.append(boundary_count)
    return boundary_index_list



if __name__ == '__main__':
    # Read the point cloud.
    point_cloud, id_list = read_xyz('C:\\Users\\mans3851\\PycharmProjects\\Researchtasks1\\DisplacementMapping'
                                    '\\TUD_COMP-4''-displacement.csv', delimiter=',')
    point_cloud, id_list = read_xyz('C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\Blender\\Blender projects'
                                    '\\TUD_COMP-4-31-12-2023\\scanned_points_deformed_noise_0_01std_frames_1_to_1.csv'
                                    , delimiter=';', noise=True)
    pcd_wall = o3d.geometry.PointCloud()
    pcd_wall.points = o3d.utility.Vector3dVector(point_cloud)
    geometry_list = []
    centroid_list = []  # Centroid of blocks in the same order as block id. i.e. The 0th element is the centroid of
    # the block with block id of 0.

    # Evaluate the convex hull of the point clouds belonging to each block.
    for block_id in range(max(id_list)-1):
        # Extract the points of each block
        block_index = np.flatnonzero(id_list == block_id)  # Indices of all the points belonging to a single block.
        points_block = point_cloud[block_index]

        # Evaluate the centroid of the points
        centroid_block = np.mean(points_block, axis=0)
        centroid_list.append(centroid_block)

        # Generate the entire wall for visualisation
        pcd_block = o3d.geometry.PointCloud()
        pcd_block.points = o3d.utility.Vector3dVector(points_block)
        hull_block, _ = pcd_block.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull_block)
        hull_ls.paint_uniform_color((0.447, 0.545, 0.741))
        geometry_list.append(hull_ls)

    # Identify boundary blocks
    centroid_list = np.array(centroid_list)
    boundary_indices = find_boundary_blocks(centroid_list)

    # Point correspondence between target and neighbouring blocks
    # Find the neighbouring blocks.
    num_neighbour = 9
    nn = pynanoflann.KDTree(n_neighbors=num_neighbour, metric='L2')
    nn.fit(centroid_list)
    distances, neighbour_indices_all = nn.kneighbors(centroid_list)
    for block_id in range(max(id_list) - 1):
        # Only process the non-boundary blocks
        if boundary_indices[block_id] > 1:
            continue

        geometry_list[block_id].paint_uniform_color((0, 1, 0.286))

        # Find the correspondence in each neighbour
        neighbour_indices = neighbour_indices_all[block_id]
        for neighbour_index in neighbour_indices:
            geometry_list[neighbour_index].paint_uniform_color((1, 0, 0))

        geometry_list[block_id].paint_uniform_color((0, 1, 0.286))

    o3d.visualization.draw_geometries(geometry_list)










    # Visualising the selection of boundary blocks.
    # for block_id in range(max(id_list)-1):
    #     block_index = np.flatnonzero(id_list == block_id)  # Indices of all the points belonging to a single block.
    #     points_block = point_cloud[block_index]
    #     pcd_block = o3d.geometry.PointCloud()
    #     pcd_block.points = o3d.utility.Vector3dVector(points_block)
    #     hull_block, hull_point_list = pcd_block.compute_convex_hull()
    #     hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull_block)
    #     hull_ls.paint_uniform_color((0.447, 0.545, 0.741))
    #     if boundary_indices[block_id] > 1:
    #         hull_ls.paint_uniform_color((1, 0, 0))
    #     geometry_list.append(hull_ls)
    # o3d.visualization.draw_geometries(geometry_list)





