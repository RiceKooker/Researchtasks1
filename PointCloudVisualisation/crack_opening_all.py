from PointCloudVisualisation.crack_opening_anyblock import find_correspondence, evaluate_hulls
from PointCloudVisualisation.const import noise_pc_dir, undeformed_dir, deformed_dir
from PointCloudVisualisation.Visualisation import read_xyz
from PointCloudVisualisation.tools import find_boundary_blocks
import pandas as pd
from Classes import GridpointReader
from DisplacementMapping import disp_utils
import numpy as np


if __name__ == '__main__':
    # Identify the boundary blocks
    centroid_list, hull_index_list, bp_index_list, point_cloud, id_list = evaluate_hulls(noise_pc_dir)
    boundary_indices = find_boundary_blocks(centroid_list)
    correspondence_list = []
    for block_id in range(max(id_list)-1):
        correspondence_list_block = find_correspondence(block_id, centroid_list, hull_index_list, bp_index_list, point_cloud, id_list)
        correspondence_list.append(correspondence_list_block)

    df = pd.read_csv(noise_pc_dir, sep=';')
    undeformed_blocks = GridpointReader(undeformed_dir).block_list
    deformed_blocks = GridpointReader(deformed_dir).block_list
    scanned_points, ids = disp_utils.read_scanned_points(noise_pc_dir, noise=True)
    original_points = disp_utils.find_co_points(undeformed_blocks, deformed_blocks, scanned_points, ids)

    point_coordinates = []
    openings = []
    for block_correspondence in correspondence_list:
        for target_index, neighbour_index in block_correspondence:
            target_disp = scanned_points[target_index] - original_points[target_index]
            neighbour_disp = scanned_points[neighbour_index] - original_points[neighbour_index]
            rel_disp = neighbour_disp - target_disp
            point_coordinates.append(scanned_points[target_index])
            openings.append(np.linalg.norm(rel_disp))
    point_coordinates = np.array(point_coordinates)
    openings = np.array(openings)
    print('Max: ', max(openings))

    X = point_coordinates[:, 0]
    Y = point_coordinates[:, 1]
    Z = point_coordinates[:, 2]
    df = pd.DataFrame(list(zip(X, Y, Z, openings)),
                      columns=['X', 'Y', 'Z', 'Opening'])
    # df.to_csv('Openings_PointCor_TUD_COMP-4.csv')










