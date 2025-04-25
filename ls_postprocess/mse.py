import numpy as np
import os
import matplotlib.pyplot as plt

def load_series(path):
    return np.loadtxt(path)

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

# Load data
y0 = load_series('C_0_0075_z_disp.txt')
y1 = load_series('C_0_0105_z_disp.txt')
y2 = load_series('D4_0_3_z_disp.txt')
y3 = load_series('D4_0_6_z_disp.txt')

# Compute differences
diff_C = y1 - y0
diff_D4 = y3 - y2

# Compute metrics
mse_C = np.mean(diff_C**2)
rmse_C = np.sqrt(mse_C)
mae_C = np.mean(np.abs(diff_C))

mse_D4 = np.mean(diff_D4**2)
rmse_D4 = np.sqrt(mse_D4)
mae_D4 = np.mean(np.abs(diff_D4))

# Normalization by baseline mean (y0)
y0_mean = np.mean(y0)
rmse_C_pct = np.abs(rmse_C / y0_mean) * 100
rmse_D4_pct = np.abs(rmse_D4 / y0_mean) * 100

# Normalized RMSE (relative to baseline range)
y0_range = y0.max() - y0.min()
nrmse_C = rmse_C / y0_range
nrmse_D4 = rmse_D4 / y0_range

# Plotting
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.plot(diff_C)
plt.title('C Parameter Difference')
plt.xlabel('Sample Index')
plt.ylabel('Difference')

plt.subplot(2, 3, 2)
plt.plot(diff_D4)
plt.title('D4 Parameter Difference')
plt.xlabel('Sample Index')
plt.ylabel('Difference')

plt.subplot(2, 3, 4)
plt.bar(['C', 'D4'], [rmse_C, rmse_D4])
plt.title('Root Mean Square Error (RMSE) Comparison')
plt.ylabel('RMSE')

plt.subplot(2, 3, 5)
plt.bar(['C', 'D4'], [rmse_C_pct, rmse_D4_pct])
plt.title('Percent Root Mean Square Error (RMSE) Comparison')
plt.ylabel('Percent RMSE (%)')

# NRMSE Comparison
plt.subplot(2, 3, 6)
plt.bar(['C', 'D4'], [nrmse_C, nrmse_D4])
plt.title('Normalized Root Mean Square Error (NRMSE) Comparison')
plt.ylabel('Normalized RMSE')

plt.tight_layout()
fig = plt.gcf()                    # get current figure
fig.savefig('difference_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Print metrics
print(f"RMSE C: {rmse_C:.6e}, RMSE D4: {rmse_D4:.6e}")
print(f"Percent RMSE C: {rmse_C_pct:.2f}%, Percent RMSE D4: {rmse_D4_pct:.2f}%")
print(f"NRMSE C: {nrmse_C:.6e}, NRMSE D4: {nrmse_D4:.6e}")