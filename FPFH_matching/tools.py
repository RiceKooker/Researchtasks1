import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def compare_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must be of equal length")

    total_elements = len(arr1)
    matching_elements = sum(1 for x, y in zip(arr1, arr2) if x == y)
    percentage = (matching_elements / total_elements) * 100
    return percentage


def local_feature_matching(pcd0, pcd1, fpsh0, fpsh1, radius):
    """
    All data should have the number of data points to be the first dimension.
    """
    correspondance0, correspondance1 = [], []

    # For each point in pcd0, find all the points in pcd1 that are within radius r of the point in pcd0.
    # local_neighbour_idx[i] contains the index of the points found in pcd1.
    tree0 = KDTree(pcd0)
    tree1 = KDTree(pcd1)
    local_neighbour_idx = tree0.query_ball_tree(other=tree1, r=radius)

    for i, neighbour_indices in enumerate(local_neighbour_idx):
        query_fpsh = fpsh0[i]
        fpsh1_neighbours = fpsh1[neighbour_indices]
        fpsh1_tree = KDTree(fpsh1_neighbours)
        distance_, closest_idx = fpsh1_tree.query(query_fpsh)
        closest_idx_global1 = neighbour_indices[closest_idx]
        correspondance0.append(i)
        correspondance1.append(closest_idx_global1)

    return correspondance0, correspondance1

def draw_hist_single_point(features, title):
    fig, ax = plt.subplots()
    ax.bar(np.array(list(range(len(features)))), features)
    ax.set_title(title)
    plt.show()


def calc_fpfh(pcd, radius, n_max=100):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius/2.5, max_nn=30))  # Empirical numbers from https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=n_max))
    return pcd_fpfh


def convert_o3d_pcd(pcd):
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_


def read_xyz(filename, delimiter=',', noise=False, voxel_size=None, openthreeD=True, offset=(0, 0, 0)):
    suffix = ''
    if noise is True:
        suffix = '_noise'
    points = []
    coord_list = ['X', 'Y', 'Z']

    df = pd.read_csv(filename, delimiter=delimiter)

    # Exclude the points scanned from the boundary blocks.
    block_id = np.array(df['categoryID'].tolist())
    boundary_block_id = np.max(block_id)
    df = df.loc[df['categoryID'] != boundary_block_id]
    df = df.loc[df['categoryID'] != boundary_block_id-1]
    
    for axis_name in coord_list:
        points.append(np.array(df[axis_name+suffix].tolist()))

    block_id = np.array(df['categoryID'].tolist())
    points = np.array(points)
    points = points.T
    points = points + np.array(offset)
    if openthreeD:
        points = convert_o3d_pcd(points)
        if voxel_size is not None:
            points = points.voxel_down_sample(voxel_size)
    return points, block_id.astype(int)


if __name__ == '__main__':
    pass