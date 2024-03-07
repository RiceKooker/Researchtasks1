from FPFH_matching.CalcVirtualPoints import original_points
from FPFH_matching.tools import calc_fpfh, convert_o3d_pcd, local_feature_matching, compare_arrays, read_xyz, \
    draw_hist_single_point, save_list, load_list
from FPFH_matching.const import offset

ave_point_sep = 1.66e-3
search_radius = ave_point_sep * 1.2 * 6
search_radius = search_radius * 3
folder_dir = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\Blender\\Blender projects\\TUD_COMP-4-31-12-2023\\AutoGenTwoStates\\SmallWall\\'
pcd0_name = 'h90-4800-v45-3600-std0_0005-smallwall_frames_1_to_1'
pcd1_name = 'h90-4800-v45-3600-std0_0005-smallwalldeformed_frames_1_to_1'
suffix = '.csv'
pcd_dir0 = f'{folder_dir}{pcd0_name}{suffix}'
pcd_dir1 = f'{folder_dir}{pcd1_name}{suffix}'

if __name__ == '__main__':
    # Load the point cloud
    pcd0, ids0 = read_xyz(filename=pcd_dir0, delimiter=';', noise=True)
    pcd1, ids1 = read_xyz(filename=pcd_dir1, delimiter=';', noise=True, offset=offset)

    print('Loading complete.')

    # Evaluate the FPFH of all points
    pcd_fpfh0 = calc_fpfh(pcd=pcd0, radius=search_radius)
    pcd_fpfh1 = calc_fpfh(pcd=pcd1, radius=search_radius)

    print('Feature evaluation complete.')

    # Perform local feature matching to find correspondence.
    c1 = local_feature_matching(pcd0=pcd0.points, pcd1=pcd1.points, fpsh0=pcd_fpfh0,
                                                  fpsh1=pcd_fpfh1, radius=search_radius)

    # Save the correspondence to local files
    # save_list('h90-4800-v45-3600-std0_0005_correspondence1.json', c1)
