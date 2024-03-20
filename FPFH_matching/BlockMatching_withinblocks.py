"""
This script is to match points belonging to two blocks that are
correspondence of each other.
Filters should be applied.
1. KL divergence filter - applied before matching features
2. Rigid body filter - applied after matching features
"""

from FPFH_matching.tools import load_list, read_xyz, calc_fpfh, get_block_points, \
    convert_o3d_pcd, calc_kl_div, feature_quality_filter, draw_hist_single_point, \
    feature_vector_match, mutual_correspondence, rigid_model_filter
import FPFH_matching.tools as tools
import numpy as np
from FPFH_matching.const import offset
import matplotlib.pyplot as plt


folder_dir = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\Blender\\Blender projects\\TUD_COMP-4-31-12-2023\\AutoGenTwoStates\\SmallWall\\CenterWall\\'
pcd0_name = 'h90-4800-v45-3600-std0_0005-undeformed_frames_1_to_1'
pcd1_name = 'h90-4800-v45-3600-std0_0005-deformed_frames_1_to_1'
suffix = '.csv'
pcd_dir0 = f'{folder_dir}{pcd0_name}{suffix}'
pcd_dir1 = f'{folder_dir}{pcd1_name}{suffix}'
average_point_separation = 1.5e-3
search_radius = average_point_separation * 1.2 * 6
search_radius = search_radius*3


if __name__ == '__main__':
    # Load correspondence and point clouds
    pcd0, ids0 = read_xyz(filename=pcd_dir0, delimiter=';', noise=False, openthreeD=False)
    pcd1, ids1 = read_xyz(filename=pcd_dir1, delimiter=';', noise=False, offset=offset, openthreeD=False)

    transformations = []

    # Match points block-wise.
    for i in range(max(ids0) + 1):
        # Extract the points belonging to the same block
        block_points0 = get_block_points(pcd0, ids0, i)
        block_points1 = get_block_points(pcd1, ids1, i)

        # Calculate the Fast Point Feature Histograms
        block_fpfh0, exc0 = calc_fpfh(convert_o3d_pcd(block_points0), radius=search_radius)
        block_fpfh1, exc1 = calc_fpfh(convert_o3d_pcd(block_points1), radius=search_radius)

        # Exclude points with zero features.
        nan_idx0 = exc0
        nan_idx1 = exc1
        block_points0 = np.array([block_points0[i] for i in range(len(block_points0)) if i not in nan_idx0])
        block_points1 = np.array([block_points1[i] for i in range(len(block_points1)) if i not in nan_idx1])
        block_fpfh0 = np.array([block_fpfh0[i] for i in range(len(block_fpfh0)) if i not in nan_idx0])
        block_fpfh1 = np.array([block_fpfh1[i] for i in range(len(block_fpfh1)) if i not in nan_idx1])

        # Pick the points whose features are distinguished enough - based on KL divergence.
        picked_point_idx0 = feature_quality_filter(block_fpfh0, alpha=1)
        picked_point_idx1 = feature_quality_filter(block_fpfh1, alpha=1)

        # Filter out the unwanted points
        block_points0 = block_points0[picked_point_idx0]
        block_points1 = block_points1[picked_point_idx1]
        block_fpfh0 = block_fpfh0[picked_point_idx0]
        block_fpfh1 = block_fpfh1[picked_point_idx1]

        # Match the feature vectors
        c1 = feature_vector_match(block_fpfh0, block_fpfh1)
        c0 = feature_vector_match(block_fpfh1, block_fpfh0)

        # Keep the mutual correspondence
        mutual_c = mutual_correspondence(c0, c1)

        # Rigid model filter
        n_iteration = 5000
        threshold = 1e-4
        best_inliers, rot, trans = rigid_model_filter(block_points0, block_points1, mutual_c, iteration=n_iteration, threshold=threshold)
        transformations.append([rot, trans])

        print(f'Block {i} - rigid model filter - pass rate: {100*len(best_inliers)/len(mutual_c)}%')

    tools.save_list(f'{folder_dir}Transformation_no_noise_dense_within_blocks', transformations)
