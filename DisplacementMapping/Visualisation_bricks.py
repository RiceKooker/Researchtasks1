from DisplacementMapping import RBlockClass as rB
import Func
import matplotlib.pyplot as plt
import numpy as np
from Const import brick_dims_UK
from Blender.WavefrontOBJ import save_obj_sep
import pandas as pd
import copy


if __name__ == '__main__':
    # Generation of a small wall unit
    block1 = rB.RBlock.dim_build(brick_dims_UK)
    block2 = rB.RBlock.dim_build(brick_dims_UK)
    block3 = rB.RBlock.dim_build(brick_dims_UK)
    block4 = rB.RBlock.dim_build(brick_dims_UK)
    block2.move([brick_dims_UK[0], 0, 0])
    block3.move([0, 0, brick_dims_UK[2]])
    block4.move([brick_dims_UK[0], 0, brick_dims_UK[2]])
    block_list = [block1, block2, block3, block4]
    # Func.draw_blocks5([block1, block2, block3, block4])

    block1_ = rB.RBlock.dim_build(brick_dims_UK)
    block2_ = rB.RBlock.dim_build(brick_dims_UK)
    block3_ = rB.RBlock.dim_build(brick_dims_UK)
    block4_ = rB.RBlock.dim_build(brick_dims_UK)
    block2_.move([brick_dims_UK[0], 0, 0])
    block3_.move([0, 0, brick_dims_UK[2]])
    block4_.move([brick_dims_UK[0], 0, brick_dims_UK[2]])
    deformed_blocks = [block1_, block2_, block3_, block4_]
    deformed_blocks[1].move([0.1*brick_dims_UK[0], 0, 0])
    deformed_blocks[1].rot([0, 15, 10])
    deformed_blocks[3].move([0.2*brick_dims_UK[0], 0, 0])
    deformed_blocks[3].rot([0, -10, -15])
    # Func.draw_blocks5(deformed_blocks)
    df = pd.read_csv('Blender/scanned_points_frames_1_to_1.csv', sep=';')
    points = []
    ids = []
    n = 0
    sample_interval = 20
    for i, row in df.iterrows():
        x = float(row['X'])
        try:
            y = float(row['Y'])
        except ValueError:
            continue
        z = float(row['Z'])
        block_id = int(row['categoryID'])
        if block_id == 0 or block_id == 2:
            continue
        n += 1
        if n >= sample_interval:
            points.append([x, y, z])
            ids.append(block_id)
            n -= sample_interval
    points = np.array(points)
    ids = np.array(ids)

    temp = []
    for point, b_id in zip(points, ids):
        point_rel = deformed_blocks[b_id].find_relative_pos(point)
        point_abs = block_list[b_id].find_absolute_pos(point_rel)
        temp.append(point_abs)
    points2 = np.array(temp)

    points_all = [points, points2]
    Func.draw_blocks5(block_list+deformed_blocks, highlight_points=points_all, show=True)
