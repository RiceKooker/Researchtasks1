import pandas as pd
import matplotlib.pyplot as plt


def three_dec_plot(file_name, label):
    df_3DEC = pd.read_csv(file_name)
    col_names = []
    for col in df_3DEC.columns:
        col_names.append(col)
    x_3DEC = [i * 1000 for i in df_3DEC[col_names[0]].tolist()]
    y_3DEC = [i / 1000 for i in df_3DEC[col_names[1]].tolist()]
    plt.scatter(x_3DEC, y_3DEC, marker='o', s=8, label=label, alpha=0.5)
    return x_3DEC, y_3DEC


if __name__ == '__main__':
    file_name1 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Pushover_high\\Systematic\\Pushover1\\force_displacement.csv'
    file_name2 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Pushover_high\\Systematic\\Pushover2\\force_displacement.csv'

    df = pd.read_csv('../data/cyclic_test_high.csv', names=['x', 'y'])
    plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=8, label='Experiment')
    three_dec_plot(file_name1, 'Pushover1')
    three_dec_plot(file_name2, 'Pushover2')

    plt.axhline(y=0, color='gray')
    plt.axvline(x=0, color='gray')
    plt.legend(markerscale=4)
    plt.xlabel('Lateral displacement / mm')
    plt.ylabel('Shear force / kN')
    plt.title('Force-displacement curve')
    plt.xlim([-14, 14])
    plt.ylim([-100, 100])
    plt.show()