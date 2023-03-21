import pandas as pd
import matplotlib.pyplot as plt


file_name = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\Systematic\\test3\\force_displacement.csv'
df = pd.read_csv('data/cyclic_test_low.csv', names=['x', 'y'])
df_3DEC = pd.read_csv(file_name)
col_names = []
for col in df_3DEC.columns:
    col_names.append(col)
plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=8, label='Experiment')
x_3DEC = [i*1000 for i in df_3DEC[col_names[0]].tolist()]
y_3DEC = [i/1000 for i in df_3DEC[col_names[1]].tolist()]
plt.scatter(x_3DEC, y_3DEC, marker='o', color='r', s=0.1, label='Simulation')
plt.axhline(y=0, color='gray')
plt.axvline(x=0, color='gray')
plt.legend()
plt.xlabel('Lateral displacement / mm')
plt.ylabel('Shear force / kN')
plt.title('Force-displacement curve')
plt.xlim([-12, 12])
plt.ylim([-90, 90])
plt.show()