import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice
import os


def downsample_to_proportion(rows, proportion=1.0):
    return list(islice(rows, 0, len(rows), int(1/proportion)))


def three_dec_plot(file_name, label, alpha=1.0, prop=1.0, s=5):
    df_3DEC = pd.read_csv(file_name)
    col_names = []
    for col in df_3DEC.columns:
        col_names.append(col)
    x_3DEC = [i * 1000 for i in df_3DEC[col_names[0]].tolist()]
    y_3DEC = [i / 1000 for i in df_3DEC[col_names[1]].tolist()]

    plt.scatter(downsample_to_proportion(x_3DEC, prop), downsample_to_proportion(y_3DEC, prop), marker='o', s=s, label=label, alpha=alpha)
    # return x_3DEC, y_3DEC


def read_plot_data(file_name, label, alpha=1.0, s=8):
    df = pd.read_csv(file_name, names=['x', 'y'])
    plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=s, label=label, alpha=alpha)


def plot_files(file_dir, prefix='force_displacement_', alpha=0.5, prop=0.05, s=5):
    csv_files = [img for img in os.listdir(file_dir) if img.endswith(".csv")]
    suffix = '.csv'
    for file in csv_files:
        label = file[len(prefix):-len(suffix)]
        file = os.path.join(file_dir, file)
        three_dec_plot(file, label=label, alpha=alpha, prop=prop, s=s)

if __name__ == '__main__':
    original_file = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\DELFT\\Long wall\\Backbone.csv'
    read_plot_data(original_file, label='Backbone', alpha=0.9, s=15)
    original_file2 = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\DELFT\\Long wall\\Full curve.csv'
    read_plot_data(original_file2, label='Full curve', alpha=0.9, s=15)
    file_name1 = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\DELFT\\Long wall\\Cyclic\\3 - Sinusoidal\\force_displacement.csv'
    file_name2 = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\DELFT\\Long wall\\Cyclic\\2\\force_displacement.csv'

    # df = pd.read_csv(original_file, names=['x', 'y'])
    # plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=8, label='Backbone curve')
    # plot_files(files_dir)
    three_dec_plot(file_name1, 'Sinusoidal forcing', alpha=0.8, prop=0.05)
    three_dec_plot(file_name2, 'Step forcing', alpha=0.8, prop=0.05)
    # three_dec_plot(file_name3, 'Pushover3')
    # three_dec_plot(file_name4, 'Pushover4')

    plt.grid(axis='x')
    plt.grid(axis='y')
    plt.legend(markerscale=4)
    plt.xlabel('Lateral displacement / mm')
    plt.ylabel('Shear force / kN')
    plt.title('TUD-COMP_4-3E-0.05ts')
    # plt.xlim([-12, 12])
    # plt.ylim([-100, 100])
    plt.show()