from Classes import GridpointReader
from DisplacementMapping import disp_utils
import pandas as pd

if __name__ == '__main__':
    undeformed_dir = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC ' \
                     'test\\Validation_tests\\Validation\\DELFT\\Long wall\\Cyclic\\2\\Undeformed\\Gp_info.txt'
    deformed_dir = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC ' \
                   'test\\Validation_tests\\Validation\\DELFT\\Long wall\\Cyclic\\2\\Gp_info.txt'
    points_dir = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\Blender\\Blender ' \
                 'projects\\TUD_COMP-4-31-12-2023\\scanned_points-no_noise_frames_1_to_1.csv'
    points_dir = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\Blender\\Blender projects' \
                   '\\TUD_COMP-4-31-12-2023\\scanned_points_deformed_noise_0_01std_frames_1_to_1.csv'
    df = pd.read_csv(points_dir, sep=';')
    undeformed_blocks = GridpointReader(undeformed_dir).block_list
    deformed_blocks = GridpointReader(deformed_dir).block_list
    scanned_points, ids = disp_utils.read_scanned_points(points_dir, noise=True)
    original_points = disp_utils.find_co_points(undeformed_blocks, deformed_blocks, scanned_points, ids)
    distance = disp_utils.points_distance(scanned_points, original_points)
    print(max(distance))
    df.insert(5, 'Displacement', distance, True)
    print(df.head())
    df.to_csv('TUD_COMP-4-displacement-noise-std0.01.csv')
