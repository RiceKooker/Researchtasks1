import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import norm
from DisplacementMapping.RBlockClass import procrustes_analysis
import random
from FPFH_matching.const import include_dict
import pickle


def find_nan_indices(lst):
    nan_indices = [i for i, x in enumerate(lst) if isinstance(x, float) and x != x]
    return nan_indices


def rigid_model_filter(pcd0, pcd1, correspondence_list, iteration, threshold, sample_n=3):
    correspondence_list = np.array(correspondence_list)
    pcd0 = np.array(pcd0)
    pcd1 = np.array(pcd1)

    # Get the corresponding points
    pcd0 = pcd0[correspondence_list[:, 0]]
    pcd1 = pcd1[correspondence_list[:, 1]]
    displacement = np.array([(p1-p0) for p0, p1 in zip(pcd0, pcd1)])

    best_inliers = []
    best_inliers_count = 0
    best_rot, best_trans = None, None
    # RANSAC
    for i in range(iteration):
        # Draw samples
        sample_idx = random.sample(list(range(len(pcd0))), sample_n)
        samples0, samples1 = pcd0[sample_idx], pcd1[sample_idx]  # The samples should dimension of sample_n x 3

        # Find the rigid model parameters and predictions (points after)
        predictions_, rot, trans = find_rigid_params_3d_fake(points_before=samples0, points_after=samples1, axis=1)
        predictions = np.array([transform_points_fake_3d(p0, rot, trans) for p0 in pcd0])

        # Evaluate the predicted displacement
        pred_disp = np.array([(p_p-p0) for p0, p_p in zip(pcd0, predictions)])

        # Perform inliers check
        current_inliers = []
        for j, disps in enumerate(zip(displacement, pred_disp)):
            disp, p_disp = disps
            if np.linalg.norm(disp - p_disp) < threshold:
                current_inliers.append(j)

        if len(current_inliers) > best_inliers_count:
            best_inliers = current_inliers
            best_inliers_count = len(current_inliers)
            best_rot = rot
            best_trans = trans

    return best_inliers, best_rot, best_trans


def find_rigid_params_3d_fake(points_before, points_after, axis=1):
    points_before = np.array(points_before)
    points_after = np.array(points_after)
    prediction_2d, rotation_matrix, trans_vec = find_rigid_params_2d(points_before[:, include_dict[axis]], points_after[:, include_dict[axis]])
    points_before[:, include_dict[axis]] = prediction_2d
    return points_before, rotation_matrix, trans_vec


def mutual_correspondence(list_a, list_b):
    """
    This function keeps the mutual correspondence between 2 correspondence index lists.
    list_a[2] = 5 means that the 3rd element in the array a corresponds to the 6th element in array b.
    @param list_a: correspondence index list bonded to array a
    @param list_b: correspondence index list bonded to array b
    @return: mutual correspondence list - mutual[2] = [2, 4] means that the 3rd element in array a
    corresponds to the 5th element in array b.
    """
    mutual = []
    for i, c_a in enumerate(list_a):
        if list_b[c_a] == i:
            mutual.append([i, c_a])
    return np.array(mutual)


def transform_points(points, rotation_matrix, translation_vector):
    """
    Transform points using rotation matrix and translation vector.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 2) containing 2D points.
        rotation_matrix (numpy.ndarray): 2x2 rotation matrix.
        translation_vector (numpy.ndarray): 1x2 translation vector.

    Returns:
        transformed_points (numpy.ndarray): Array of shape (N, 2) containing transformed points.
    """
    return np.dot(points, rotation_matrix.T) + translation_vector


def transform_points_fake_3d(points, rotation, translation, axis=1):
    points = np.array(points)
    points[include_dict[axis]] = transform_points(points=points[include_dict[axis]],
                                                  rotation_matrix=rotation, translation_vector=translation)
    return points


def find_rigid_params_2d(points_before, points_after):
    points_before = np.array(points_before)
    points_after = np.array(points_after)
    # If ignore y axis
    rotation_matrix, trans_matrix = procrustes_analysis(points_after, points_before)
    predicted_points = transform_points(points_before, rotation_matrix, trans_matrix)

    return predicted_points, rotation_matrix, trans_matrix


def fit_gaussian(your_data, xlabel, ylabel):
    mean, std_dev = norm.fit(your_data)
    plt.hist(your_data, bins=100, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std_dev)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("Fit results: Mean = %.2f,  Standard Deviation = %.2f" % (mean, std_dev))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    # plt.show()
    return mean, std_dev


def feature_quality_filter(fpfh, alpha):
    # Evaluate the mean FPFH
    mean_fpfh = np.mean(fpfh, axis=0)

    # Evaluate the KL divergence of all points
    kl_div = np.asarray([calc_kl_div(fpfh_i, mean_fpfh) for fpfh_i in fpfh])

    # Transform into log scale
    kl_div = np.log(kl_div)

    # Fit a Gaussian and find the mean and standard deviation
    miu, std = fit_gaussian(kl_div, xlabel='KL divergence value', ylabel='Probability')

    # Pick the points with more distinguished features
    return [i for i, value in enumerate(kl_div) if value < miu-std*alpha or value > miu+std*alpha]


def calc_kl_div(point_fpfh, mean_fpfh, threshold=0.0001):
    out = 0
    for i, bin_values in enumerate(zip(point_fpfh, mean_fpfh)):
        p, m = bin_values
        # Ignore zero values
        if p < threshold or m < threshold:
            continue
        out += (p-m)*np.log(p/m)
    return out


def get_block_points(pcd, ids, block_id):
    """
    This function extract all the points belonging to the same block.
    @param pcd: overall point cloud
    @param ids: list of block id - ids[i] tells which block does the ith point belongs to
    @param block_id: block id of the block interested
    @return: points array - nx3
    """
    pcd = np.array(pcd)
    ids = np.array(ids)
    point_idx = np.where(ids == block_id)
    return pcd[point_idx]


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


def save_list_json(file_name, list_data):
    with open(file_name, 'w') as f:
        json.dump(list_data, f, indent=2)


def load_list_json(file_name):
    with open(file_name, 'r') as f:
        loaded_file = json.load(f)
    return loaded_file


def save_list(file_name, list_data):
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(list_data, fp)


def load_list(file_name):
    with open(file_name, "rb") as fp:  # Pickling
        out = pickle.load(fp)
    return out


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


def feature_vector_match(f0, f1):
    """
    This function finds the correspondence between two sets of feature vectors.
    It gives the corresponding point index in the source feature vector.
    correspondence[5] = 10 means that the 6th data
    point in f1 corresponds to the 11th point in f0.
    @param f0: source feature vectors - n x m where n is the number of data points and m is the number of features.
    @param f1: target feature vector
    @return: correspondence
    """
    f_tree = KDTree(f0)
    correspondence = []
    for each_f in f1:
        distance_, closest_idx = f_tree.query(each_f)
        correspondence.append(closest_idx)
    return correspondence


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
    out = []
    exclude = []
    for i, feature in enumerate(pcd_fpfh.data.T):
        with np.errstate(invalid='raise'):
            try:
                out.append(feature/sum(feature))  # this gets caught and handled as an exception
            except FloatingPointError:
                exclude.append(i)
                continue

    return np.array(out), np.array(exclude)


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
    a = [1, 2, 3]
    b = ['a', 'b', 'c']
    print(random.sample(zip(a, b), 2))