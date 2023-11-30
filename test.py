import pandas as pd
from Const import grid_point_reader
import numpy as np


class GridpointReader:
    def __init__(self, file_dir):
        self.df = pd.read_csv(file_dir, sep=" ")
        num_block = int(self.df.max(axis=0)[grid_point_reader['ID_label']])  # Read the number of blocks
        blocks = []
        for i in range(1, num_block+1):
            # Get the information of all grid points belong to a single block
            df_block = pd.DataFrame(self.df.loc[self.df[grid_point_reader['ID_label']] == i].reset_index(drop=True))
            # Read the positions of all grid points
            gp_position = df_block[grid_point_reader['Position_labels']].values
            # Read the displacements of all grid points
            gp_disp = df_block[grid_point_reader['Disp_labels']].values
            gp_pos_new = []
            for pos, disp in zip(gp_position, gp_disp):
                # Calculate the final position and append it to the list
                gp_pos_new.append(list(np.add(np.array(pos), np.array(disp))))


def find_vert_from_gps(gps):
    """
    This function finds the limits of the block in the x,y and z directions
    :param gps: list of grid point coordinates [number of grid points, 3] - 3 dimensional coordinates
    :return: 3 dimensional upper bound list, and lower bound list
    """
    # Initialize the max and min values in the 3 axis
    a_large_num = 100000
    a_small_num = 0
    mins = [a_large_num for i in range(len(gps[0]))]
    maxs = [a_small_num for i in range(len(gps[0]))]
    for gp in gps:
        for i, axis_value in enumerate(gp):
            if axis_value > maxs[i]:
                maxs[i] = axis_value
            if axis_value < mins[i]:
                mins[i] = axis_value
    return maxs, mins


def create_block_gps(gps):
    maxs, mins = find_vert_from_gps(gps)
    dims = []
    for lim_upper, lim_lower in zip(maxs, mins):
        dims.append(lim_upper - lim_lower)
    dims = np.array(dims)
    CoM = np.array(mins) + dims







if __name__ == '__main__':
    df1 = pd.read_csv("Position2_test.txt", sep=" ")
    df2 = pd.DataFrame(df1.loc[df1['Block_ID'] == 1].reset_index(drop=True))
    print(df2.head())
    sample_gp = GridpointReader("Position2_test.txt")

    sp = [[1,2,3],[3,4,5]] 
    a,b = find_vert_from_gps(sp)


