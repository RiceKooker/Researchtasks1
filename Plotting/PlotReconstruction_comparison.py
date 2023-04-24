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

    plt.scatter(downsample_to_proportion(x_3DEC, prop), downsample_to_proportion(y_3DEC, prop), marker='o', s=5, label=label, alpha=alpha)
    return x_3DEC, y_3DEC


if __name__ == '__main__':
    file_name1 = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\Systematic\\test3\\New folder\\force_displacement.csv'
    file_name2 = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\Systematic\\test6\\force_displacement.csv'
    # file_name3 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Pushover_low_wall\\Pushover_systematic\\Pushover3\\force_displacement.csv'
    # file_name4 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Pushover_low_wall\\Pushover_systematic\\Pushover4\\force_displacement.csv'

    df = pd.read_csv('../data/cyclic_test_low.csv', names=['x', 'y'])
    plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=8, label='Experiment')
    three_dec_plot(file_name2, 'Linear stiffness degradation', alpha=0.5, prop=0.1)
    three_dec_plot(file_name1, 'No stiffness degradation', alpha=0.2, prop=0.1)
    # three_dec_plot(file_name3, 'Pushover3')
    # three_dec_plot(file_name4, 'Pushover4')

    plt.axhline(y=0, color='gray')
    plt.axvline(x=0, color='gray')
    plt.legend(markerscale=4)
    plt.xlabel('Lateral displacement / mm')
    plt.ylabel('Shear force / kN')
    plt.title('Force-displacement curve')
    plt.xlim([-12, 12])
    plt.ylim([-100, 100])
    plt.show()