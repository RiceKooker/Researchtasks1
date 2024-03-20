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

    # Calculate the Fast Point Feature Histograms
    fpfh0 = calc_fpfh(convert_o3d_pcd(pcd0), radius=search_radius)
    fpfh1 = calc_fpfh(convert_o3d_pcd(pcd1), radius=search_radius)

    # Pick the points whose features are distinguished enough - based on KL divergence.
    picked_point_idx0 = feature_quality_filter(fpfh0, alpha=1)
    picked_point_idx1 = feature_quality_filter(fpfh1, alpha=1)

    # Filter out the unwanted points
    pcd0 = pcd0[picked_point_idx0]
    ids0 = ids0[picked_point_idx0]
    fpfh0 = fpfh0[picked_point_idx0]
    pcd1 = pcd1[picked_point_idx1]
    ids1 = ids1[picked_point_idx1]
    fpfh1 = fpfh1[picked_point_idx1]

    transformations = []

    # Match points block-wise.
    for i in range(max(ids0) + 1):
        # Extract the points belonging to the same block
        block_points0 = get_block_points(pcd0, ids0, i)
        block_points1 = get_block_points(pcd1, ids1, i)
        block_fpfh0 = get_block_points(fpfh0, ids0, i)
        block_fpfh1 = get_block_points(fpfh1, ids1, i)

        # Match the feature vectors
        c1 = feature_vector_match(block_fpfh0, block_fpfh1)
        c0 = feature_vector_match(block_fpfh1, block_fpfh0)

        # Keep the mutual correspondence
        mutual_c = mutual_correspondence(c0, c1)

        # Rigid model filter
        n_iteration = 10000
        threshold = 1e-4 * 0.5
        best_inliers, rot, trans = rigid_model_filter(block_points0, block_points1, mutual_c, iteration=n_iteration, threshold=threshold)
        transformations.append([rot, trans])

        print(f'Block {i} - rigid model filter - pass rate: {100*len(best_inliers)/len(mutual_c)}%')

    tools.save_list(f'{folder_dir}Transformation_no_noise_dense_lower_threshold', transformations)
