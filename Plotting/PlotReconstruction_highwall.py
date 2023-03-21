import pandas as pd
import matplotlib.pyplot as plt
from PlotReconstruction_comparison import three_dec_plot


file_name = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_high_wall\\Cyclic1\\force_displacement.csv'
df = pd.read_csv('../data/cyclic_test_high.csv', names=['x', 'y'])
df_3DEC = pd.read_csv(file_name)
col_names = []
for col in df_3DEC.columns:
    col_names.append(col)
plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=8, label='Experiment')
three_dec_plot(file_name, 'Simulation', alpha=0.5, prop=0.01)
plt.axhline(y=0, color='gray')
plt.axvline(x=0, color='gray')
plt.legend(markerscale=4)
plt.xlabel('Lateral displacement / mm')
plt.ylabel('Shear force / kN')
plt.title('Force-displacement curve')
plt.xlim([-14, 14])
plt.ylim([-100, 100])
plt.show()