"""
This script is to visualise the clustering of scanned points using convex hull.
This script also uses the idea of hausdorff distance to find cracks.
"""
import numpy as np
import open3d as o3d
import pandas as pd
from Const import brick_dims_UK


def read_xyz(filename, delimiter=',', noise=False):
    suffix = ''
    if noise is True:
        suffix = '_noise'
    df = pd.read_csv(filename, delimiter=delimiter)
    x = np.array(df['X'+suffix].tolist())
    y = np.array(df['Y'+suffix].tolist())
    z = np.array(df['Z'+suffix].tolist())
    block_id = np.array(df['categoryID'].tolist())
    points = np.array([x, y, z])
    return points.T, block_id.astype(int)


def visualize_pc(point_cloud):
    """
    This function visualize the input point cloud.
    :param point_cloud: shape in the form of tensors.
    :return:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])


def hausdorff_d(points_a, points_b):
    h, a_index, b_index = 0, 0, 0
    for i, a in enumerate(points_a):
        shortest_d = 10000000
        for j, b in enumerate(points_b):
            point_distance = np.linalg.norm(a - b)
            if point_distance < shortest_d:
                shortest_d = point_distance
                b_index = j
        if shortest_d > h:
            h = shortest_d
            a_index = i
    return points_a[a_index], points_b[b_index], h


def single_line_set(point1, point2, color='red'):
    points = [point1, point2]
    lines = [[0, 1]]
    # if color == 'red':
    #     colors = [1, 0, 0]
    # else:
    #     colors = color
    colors = [[1, 0, 0]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def hausdorff_line_set(points_a, points_b, color='red'):
    point_a, point_b, h = hausdorff_d(points_a, points_b)
    return single_line_set(point_a, point_b, color)



if __name__ == '__main__':
    point_cloud, id_list = read_xyz('C:\\Users\\mans3851\\PycharmProjects\\Researchtasks1\\DisplacementMapping'
                                    '\\TUD_COMP-4''-displacement.csv', delimiter=',')
    # point_cloud, id_list = read_xyz('C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\Blender\\Blender projects'
    #                                 '\\TUD_COMP-4-31-12-2023\\scanned_points_deformed_noise_0_01std_frames_1_to_1.csv'
    #                                 , delimiter=';', noise=True)
    # point_cloud = np.random.rand(100, 3)
    pcd_wall = o3d.geometry.PointCloud()
    pcd_wall.points = o3d.utility.Vector3dVector(point_cloud)
    geometry_list = []
    for block_id in range(max(id_list)+1):
        block_index = np.flatnonzero(id_list == block_id)
        points_block = point_cloud[block_index]
        pcd_block = o3d.geometry.PointCloud()
        pcd_block.points = o3d.utility.Vector3dVector(points_block)
        hull_block, _ = pcd_block.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull_block)
        hull_ls.paint_uniform_color((0.447, 0.545, 0.741))
        geometry_list.append(hull_ls)

        if block_id in list(range(100, 110)):
            if not block_id == max(id_list):
                block_index_next = np.flatnonzero(id_list == block_id + 1)
                hausdorff_line = hausdorff_line_set(points_block, point_cloud[block_index_next])
                geometry_list.append(hausdorff_line)


    # o3d.visualization.draw_geometries([pcd_wall]+hull_list)
    o3d.visualization.draw_geometries(geometry_list)


    # point_cloud = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.2]])
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # hull, _ = pcd.compute_convex_hull()
    # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    # hull_ls.paint_uniform_color((1, 0, 0))
    # o3d.visualization.draw_geometries([pcd, hull_ls])


