import pandas as pd
import numpy as np


def read_scanned_points(file_dir, sample_interval=1):
    """
    This function reads the csv that contains the scanned point information.
    It returns the coordinates of the scanned points - 'points'.
    It also returns the block id to which the scanned points belong - 'ids'.
    :param file_dir: directory of the csv file.
    :param sample_interval: sampling interval.
    """
    df = pd.read_csv(file_dir, sep=';')
    points = []
    ids = []
    n = 0
    for i, row in df.iterrows():
        x = float(row['X'])
        try:
            y = float(row['Y'])
        except ValueError:
            continue
        z = float(row['Z'])
        block_id = int(row['categoryID'])
        n += 1
        if n >= sample_interval:
            points.append([x, y, z])
            ids.append(block_id)
            n -= sample_interval
    points = np.array(points)
    ids = np.array(ids)
    return points, ids


def find_co_points(blocks, blocks_, points, ids):
    """
    This function finds the coordinates of the scanned points before deformation.
    :param blocks: list of blocks before deformation.
    :param blocks_: list of blocks after deformation.
    :param points: absolute coordinates of the scanned points after deformation.
    :param ids: block id to which the scanned points belong.
    """
    temp = []
    for point, b_id in zip(points, ids):
        point_rel = blocks_[b_id].find_relative_pos(point)
        point_abs = blocks[b_id].find_absolute_pos(point_rel)
        temp.append(point_abs)
    return np.array(temp)

