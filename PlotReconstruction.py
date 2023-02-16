import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cyclic_test.csv', names=['x', 'y'])
plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=5)
plt.axhline(y=0, color='gray')
plt.axvline(x=0, color='gray')
plt.xlabel('Lateral displacement / mm')
plt.ylabel('Shear force / kN')
plt.xlim([-12, 12])
plt.ylim([-90, 90])
plt.show()