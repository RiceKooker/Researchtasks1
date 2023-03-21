side_dict = {'top': [2, 1], 'bot': [2, -1], 'left': [0, -1], 'right': [0, 1], 'front': [1, -1], 'back': [1, 1]}
origin = [0, 0, 0]
# brick_dims_UK = [215, 102.5, 65] / 1000000
brick_dims_UK = [0.215, 0.1025, 0.065]
output_geometry_file_separator = ['GRIDPOINTS', 'FACES', 'BLOCKS']
axis_dict = {0: 'x', 1: 'y', 2: 'z'}
grid_point_reader = {'ID_label': 'Block_ID', 'Position_labels': ['Pos_x', 'Pos_y', 'Pos_z'], 'Disp_labels': ['Disp_x', 'Disp_y', 'Disp_z']}
vertex_index_dict = {0: [0, 0, 0], 1: [0, 0, 1], 2: [1, 0, 1], 3: [1, 0, 0], 4: [0, 1, 0], 5: [0, 1, 1], 6: [1, 1, 1], 7: [1, 1, 0]}

directory_prefix = 'C:\\Users\\mans3851\\OneDrive - Nexus365'

