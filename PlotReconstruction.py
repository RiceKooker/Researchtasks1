import pandas as pd
import matplotlib.pyplot as plt
from Plotting.PlotReconstruction_comparison import three_dec_plot


file_name = 'C:\\Users\\dgian\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\Cyclic_low_wall\\Systematic\\test5\\force_displacement.csv'
df = pd.read_csv('data/cyclic_test_low.csv', names=['x', 'y'])
plt.scatter(df['x'].tolist(), df['y'].tolist(), marker='o', s=8, label='Experiment')
three_dec_plot(file_name, 'Simulation', alpha=0.2, prop=0.1)
plt.axhline(y=0, color='gray')
plt.axvline(x=0, color='gray')
plt.legend()
plt.xlabel('Lateral displacement / mm')
plt.ylabel('Shear force / kN')
plt.title('Force-displacement curve')
plt.xlim([-12, 12])
#plt.ylim([-90, 90])
plt.show()