import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import json
from tqdm import tqdm
from sklearn.cluster import KMeans


def min_point_spacing(points_array, n_samples=10, n_neighbours=8):
    """
    This function estimates the average point spacing among a group of points by taking samples and measuring the
    distance between the samples and their neighbouring points.
    @param points_array: point cloud array - nxm where n is the number of points and m is the dimension of each point.
    @param n_samples: samples to take from the point cloud.
    @param n_neighbours: number of neighbours used to evaluate the average spacing.
    @return: scalar - average point spacing.
    """
    samples = sample_points(points_array, n_samples)
    tree = KDTree(points_array)
    spacing = []
    for sample in samples:
        distance_, idx = tree.query(sample, k=n_neighbours+1)
        neighbours = points_array[idx[1:]]
        spacing.append(find_ave_spacing(sample, neighbours))
    return sum(spacing)/len(spacing)



def sample_points(points, n_samples):
    # Convert points to numpy array
    points_array = np.array(points)

    # Initialize KMeans with n_samples clusters
    kmeans = KMeans(n_clusters=n_samples, random_state=0)

    # Fit KMeans to the data
    kmeans.fit(points_array)

    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_

    return centroids.tolist()


def find_ave_spacing(point_s, point_t):
    """
    This function evaluates the average euclidean distance between a group of target points and a source point.
    @param point_s: source point - shape of (m,) where m is the dimension of the data point
    @param point_t: group of target points - shape of (n,m)
    @return: a scalar indicating the average spacing
    """
    point_s = np.asarray(point_s)
    point_t = np.asarray(point_t)
    diff = point_t - point_s
    return sum([np.linalg.norm(i) for i in diff])/len(diff)


def save_list(file_name, list_data):
    with open(file_name, 'w') as f:
        json.dump(list_data, f, indent=2)


def load_list(file_name):
    with open(file_name, 'r') as f:
        loaded_file = json.load(f)
    return loaded_file


def compare_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must be of equal length")

    total_elements = len(arr1)
    matching_elements = sum(1 for x, y in zip(arr1, arr2) if x == y)
    percentage = (matching_elements / total_elements) * 100
    return percentage


def local_feature_matching(pcd0, pcd1, fpsh0, fpsh1, radius):
    """
    This function finds the correspondence between the target point cloud and the source point cloud by comparing their
    feature vectors.
    @param pcd0: source point cloud to be matched - dim: nx3, where n is the number of points
    @param pcd1: target point cloud - dim: nx3
    @param fpsh0: features of the source point cloud - dim: nxm, where m is the dimension of each feature vector
    @param fpsh1: features of the target point cloud - dim: nxm
    @param radius: local search radius - in meter
    @return: a list of indices of points in the source point cloud. i.e output[10] = 35 means that the 11th point in the
    target point cloud pcd1 corresponds to the 36th point in the source point cloud pcd0.
    """
    print('Neighbour searching in progress.')
    correspondance0, correspondance1 = [], []

    # For each point in pcd0, find all the points in pcd1 that are within radius r of the point in pcd0.
    # local_neighbour_idx[i] contains the index of the points found in pcd1.
    tree0 = KDTree(pcd0)
    tree1 = KDTree(pcd1)
    local_neighbour_idx = tree0.query_ball_tree(other=tree1, r=radius)
    print('Neighbours found.')

    # Perform feature matching within the neighbouring groupof each point in pcd0.
    print('Local matching search is in progress...')
    for i, neighbour_indices in tqdm(enumerate(local_neighbour_idx)):
        query_fpsh = fpsh0[i]
        fpsh1_neighbours = fpsh1[neighbour_indices]
        fpsh1_tree = KDTree(fpsh1_neighbours)
        distance_, closest_idx = fpsh1_tree.query(query_fpsh)
        closest_idx_global1 = neighbour_indices[closest_idx]
        correspondance1.append(closest_idx_global1)
    print('Local searching done')

    return correspondance1


def draw_hist_single_point(features, title):
    fig, ax = plt.subplots()
    ax.bar(np.array(list(range(len(features)))), features)
    ax.set_title(title)
    plt.show()


def calc_fpfh(pcd, radius, n_max=100):
    """
    This function evaluates the normalized fast point feature histogram of the o3d point cloud.
    """
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius/2.5, max_nn=30))  # Empirical numbers from https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=n_max))
    return np.array([i/np.sum(i) for i in pcd_fpfh.data.T])


def convert_o3d_pcd(pcd):
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_


def read_xyz(filename, delimiter=',', noise=False, voxel_size=None, openthreeD=True, boundary=True, offset=(0, 0, 0)):
    """
    This function reads the point cloud scanned from masonry stored in csv file format.
    @param filename: absolute directory of the csv file.
    @param delimiter: delimiter used in the csv file.
    @param noise: T or F, determine if the noise data or the noiseless data is read.
    @param voxel_size: for down-sampling
    @param openthreeD: if T, outputs open3D point cloud class otherwise outputs numpy array
    @param boundary: if T, all points belonging to the last 2 blocks are ignored.
    @param offset: displace all the points with this offset
    @return: point cloud data, a numpy array containing the block id each point belongs to
    """
    suffix = ''
    if noise is True:
        suffix = '_noise'
    points = []
    coord_list = ['X', 'Y', 'Z']

    df = pd.read_csv(filename, delimiter=delimiter)

    # Exclude the points scanned from the boundary blocks.
    if not boundary:
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
    def generate_evenly_spaced_points_with_noise(l, n, noise_std):
        # Generate evenly spaced points in each dimension
        x_points = np.linspace(0, l * (n - 1), n)
        y_points = np.linspace(0, l * (n - 1), n)
        z_points = np.linspace(0, l * (n - 1), n)

        # Create meshgrid from points
        xx, yy, zz = np.meshgrid(x_points, y_points, z_points)

        # Stack the meshgrid coordinates to form the points
        points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        # Generate zero-mean Gaussian noise
        noise = np.random.normal(loc=0, scale=noise_std, size=points.shape)

        # Add noise to the points
        points_with_noise = points + noise

        return points_with_noise

    points1 = generate_evenly_spaced_points_with_noise(2, 10, 0.001)
    print(min_point_spacing(points1, 10, 8))

