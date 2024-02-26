from tools import calc_fpfh, read_xyz, draw_hist_single_point
from const import deformed_pc_dir, undeformed_pc_dir, average_point_separation, offset
import numpy as np

search_radius = average_point_separation * 1.2 * 6 # This is from Yiyan's report - page 155.
# Load the point clouds.
pcd0, ids0 = read_xyz(filename=undeformed_pc_dir, delimiter=';', noise=True)
pcd1, ids1 = read_xyz(filename=deformed_pc_dir, delimiter=';', noise=True, offset=offset)
pcd_fpfh0 = calc_fpfh(pcd=pcd0, radius=search_radius)
pcd_fpfh1 = calc_fpfh(pcd=pcd1, radius=search_radius)


if __name__ == '__main__':
    draw_hist_single_point(pcd_fpfh0.data[:, 0], title='Example of FPFH')
    