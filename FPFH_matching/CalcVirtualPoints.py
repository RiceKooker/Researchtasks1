from Classes import GridpointReader
from DisplacementMapping import disp_utils
from FPFH_matching.const import deformed_pc_dir, undeformed_pc_dir, undeformed_dir, deformed_dir
from FPFH_matching.tools import read_xyz
from FPFH_matching.const import offset


undeformed_blocks = GridpointReader(undeformed_dir).block_list
deformed_blocks = GridpointReader(deformed_dir).block_list
scanned_points, ids = read_xyz(deformed_pc_dir, delimiter=';', noise=True, openthreeD=False, offset=offset)
original_points = disp_utils.find_co_points(undeformed_blocks, deformed_blocks, scanned_points, ids)


if __name__ == '__main__':
    print(original_points.shape)
