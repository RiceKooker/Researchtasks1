import pandas as pd
import matplotlib.pyplot as plt


file_name = 'C:\\Users\\dgian\\Documents\\Itasca\\3dec700\\My Projects\\Pushover\\force_diplacement.csv'
df = pd.read_csv('cyclic_test.csv', names=['x', 'y'])
df_3DEC = pd.read_csv(file_name)
col_names = []
for col in df_3DEC.columns:
    col_names.append(col)
plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=5)
x_3DEC = [i*1000 for i in df_3DEC[col_names[0]].tolist()]
y_3DEC = [i/1000 for i in df_3DEC[col_names[1]].tolist()]
plt.scatter(x_3DEC, y_3DEC, marker='o', color='r', s=0.1)
plt.axhline(y=0, color='gray')
plt.axvline(x=0, color='gray')
plt.xlabel('Lateral displacement / mm')
plt.ylabel('Shear force / kN')
plt.title('Force-displacement curve')
plt.xlim([-12, 12])
plt.ylim([-90, 90])
plt.show()