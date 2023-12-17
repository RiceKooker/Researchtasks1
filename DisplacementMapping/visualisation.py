from DisplacementMapping import RBlockClass as rB
import Func
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    sample_block1 = rB.RBlock.dim_build([1, 3, 2])
    sample_block2 = rB.RBlock.dim_build([1, 3, 2])
    sample_block1.move([3, 4, 0])
    sample_block1.rot([90, 45, -15])
    points1 = sample_block1.generate_surface_points(4)

    # Back trace the points on the initial block
    temp = []
    for point in points1:
        point_rel = sample_block1.find_relative_pos(point)
        point2 = sample_block2.find_absolute_pos(point_rel)
        temp.append(point2)
    points2_rel = np.array(temp)


    # points2 = sample_block2.generate_surface_points(4)
    points = [points1, points2_rel]
    Func.draw_blocks5([sample_block1,sample_block2], highlight_points=points, show=False)
    plt.show()

