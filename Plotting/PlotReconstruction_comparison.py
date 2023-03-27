import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice


def downsample_to_proportion(rows, proportion=1.0):
    return list(islice(rows, 0, len(rows), int(1/proportion)))


def three_dec_plot(file_name, label, alpha=1.0, prop=1.0):
    df_3DEC = pd.read_csv(file_name)
    col_names = []
    for col in df_3DEC.columns:
        col_names.append(col)
    x_3DEC = [i * 1000 for i in df_3DEC[col_names[0]].tolist()]
    y_3DEC = [i / 1000 for i in df_3DEC[col_names[1]].tolist()]

    plt.scatter(downsample_to_proportion(x_3DEC, prop), downsample_to_proportion(y_3DEC, prop), s=5, label=label, alpha=alpha)
    return x_3DEC, y_3DEC


if __name__ == '__main__':
    file_name1 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\Systematic\\test2\\force_displacement.csv'
    file_name2 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\Systematic\\test3\\force_displacement.csv'
    file_name4 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\Systematic\\test4\\force_displacement.csv'
    file_name5 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\Systematic\\test5\\force_displacement.csv'
    reduce_prop = 0.02
    alpha = 0.5
    df = pd.read_csv('../data/cyclic_test_low.csv', names=['x', 'y'])
    plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=8, label='Experiment')
    three_dec_plot(file_name1, 'Large strain - on', alpha=alpha, prop=reduce_prop)
    three_dec_plot(file_name2, 'Large strain - off', alpha=alpha, prop=reduce_prop)
    three_dec_plot(file_name4, 'High stiffness of bricks', alpha=alpha, prop=reduce_prop*0.5)
    three_dec_plot(file_name5, 'High stiffness and tensile strength', alpha=alpha, prop=reduce_prop*0.5)

    plt.axhline(y=0, color='gray')
    plt.axvline(x=0, color='gray')
    plt.legend(markerscale=4)
    plt.xlabel('Lateral displacement / mm')
    plt.ylabel('Shear force / kN')
    plt.title('Force-displacement curve')
    plt.xlim([-12, 12])
    # plt.ylim([-90, 90])
    plt.show()