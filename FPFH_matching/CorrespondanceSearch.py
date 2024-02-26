from FPFH_matching.CalculateFPFH import pcd1, ids1, pcd_fpfh1, search_radius
from FPFH_matching.CalcVirtualPoints import original_points
from FPFH_matching.tools import calc_fpfh, convert_o3d_pcd, local_feature_matching, compare_arrays
import json

if __name__ == '__main__':
    load = True
    if not load:
        r = 200e-3
        original_points = convert_o3d_pcd(original_points)
        orig_fpfh = calc_fpfh(original_points, radius=search_radius)
        c_origin, c_deformed = local_feature_matching(pcd0=original_points.points, pcd1=pcd1.points, fpsh0=orig_fpfh.data.T,
                                                      fpsh1=pcd_fpfh1.data.T, radius=r)
        with open("correspondence.json", 'w') as f:
            json.dump([c_origin, c_deformed], f, indent=2)
    else:
        with open("correspondence.json", 'r') as f:
            correspondence = json.load(f)
        c_origin = correspondence[0]
        c_deformed = correspondence[1]
        print('Exact accuracy: ', compare_arrays(c_origin, c_deformed))
