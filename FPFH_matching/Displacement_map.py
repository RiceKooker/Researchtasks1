"""
This script uses the correspondence matching results to evaluate a displacement map.
"""
import FPFH_matching.tools as tools
import numpy as np
import pandas as pd


folder_dir = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\Blender\\Blender projects\\TUD_COMP-4-31-12-2023\\AutoGenTwoStates\\SmallWall\\CenterWall\\'
pcd0_name = 'h90-4800-v45-3600-std0_0005-undeformed_frames_1_to_1'
pcd1_name = 'h90-4800-v45-3600-std0_0005-deformed_frames_1_to_1'
suffix = '.csv'
pcd_dir0 = f'{folder_dir}{pcd0_name}{suffix}'
pcd_dir1 = f'{folder_dir}{pcd1_name}{suffix}'

if __name__ == '__main__':
    transformation = tools.load_list(f'{folder_dir}Transformation_no_noise_dense_within_blocks')
    pcd0, ids0 = tools.read_xyz(filename=pcd_dir0, delimiter=';', noise=False, openthreeD=False)
    disp_mag = []
    for point, block_id in zip(pcd0, ids0):
        points_after = tools.transform_points_fake_3d(point, transformation[block_id][0], transformation[block_id][1])
        displacement = np.linalg.norm(points_after-point)
        disp_mag.append(displacement)
    print(max(disp_mag))
    # Generate csv file
    prep_dict = {}
    key_names = ['X', 'Y', 'Z', 'D']
    for i, k_name in enumerate(key_names):
        if k_name == key_names[-1]:
            prep_dict[k_name] = disp_mag
        else:
            prep_dict[k_name] = pcd0[:, i]
    df = pd.DataFrame(data=prep_dict)
    df.to_csv(f'{folder_dir}Disp_from_correspondence_2filters_no_noise_dense_within_blocks.csv')