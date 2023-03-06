import matplotlib.pyplot as plt
import pandas as pd


file_name1 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\crack_opening_small_boundary.csv'
file_name2 = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\crack_opening_large_boundary.csv'
file_name3 = 'C:\\Users\\dgian\\PycharmProjects\\Researchtasks\\data\\crack_opening.csv'
df1 = pd.read_csv(file_name1)
df2 = pd.read_csv(file_name2)
df3 = pd.read_csv(file_name3, names=['time', 'opening'])
col_names = []
for col in df1.columns:
    col_names.append(col)
y1 = df1[col_names[1]].tolist()
y2 = df2[col_names[1]].tolist()
y = [b-a for a, b in zip(y1, y2)]
y = [i*1000 for i in y]  # Convert m into mm
x = df1[col_names[0]].tolist()
x_start = x[0]
x_end = x[-1]
x = [(i-x_start) for i in x]
x = [i*4000/(x_end-x_start) for i in x]  # Convert steps into time
plt.scatter(x, y, s=0.1, label='Simulation')

plt.scatter(df3['time'].tolist(), df3['opening'].tolist(), label='Experiment')
plt.xlabel('Time/s')
plt.ylabel('Crack opening size/mm')
plt.title('Crack width history at mid-height')
plt.legend()
plt.grid()
plt.show()
