import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

dir = dir

# File names (replace with your actual file paths)
file1 = f'{dir}to_pdf_ctrl.csv'
file2 = f'{dir}to_pdf_sst.csv'
file3 = f'{dir}to_pdf_sst10.csv'

column_name = 'W'

# Read the files and extract the desired column (replace 'column_name' with your actual column name or index)
data1 = pd.read_csv(file1)[column_name]
data2 = pd.read_csv(file2)[column_name]
data3 = pd.read_csv(file3)[column_name]


plt.figure(figsize=(8,5))
# Create the histograms
plt.hist(data1, bins=30, alpha=0.5, density=True, histtype='step', label='Control',lw=2)
plt.hist(data2, bins=30, alpha=0.5, density=True, histtype='step', label='SST 15',lw=2)
plt.hist(data3, bins=30, alpha=0.5, density=True, histtype='step', label='SST 10',lw=2)

# Add labels and legend
plt.xlabel(r'W [$m/s$]',fontsize=14)
plt.ylabel('Density',fontsize=14)
plt.tick_params(labelsize=14)
plt.legend()

plt.savefig(f'{dir}/W_PDF.png',dpi=500,bbox_inches='tight')

# Show the plot
#plt.show()
