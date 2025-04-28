"""
Author: Rong Jin, University of Kentucky
Date: 04-28-2025
"""
import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

# Constants and paths
base_dir = os.path.dirname(os.path.abspath(__file__))
mat_file = os.path.join(base_dir, 'MAT.txt')
original_k_file = os.path.join(base_dir, 'Run.k')
data_dir = os.path.join(base_dir, 'data')
plot_dir = os.path.join(base_dir, 'plot')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
lsdyna_executable = r"C:\\Program Files\\ANSYS Inc\\v251\\ansys\\bin\\winx64\\lsdyna_dp.exe"
ncpu = 4
memory = '256m'

# Parameters
param_labels = ['RA', 'RB', 'Rn', 'RC', 'Rm', 'RD1', 'RD2', 'RD3', 'RD4', 'RD5']
with open(mat_file, 'r') as f:
    params_true = [float(line.strip()) for line in f.readlines()]

# Adjustments (modifiable)
param_adjustment = 0.3  # 30%
selected_params = ['C', 'D4']  # example selection

# Helper functions
def modify_k_file(base_k_file, output_k_file, param_indices, param_values):
    with open(base_k_file, 'r') as f:
        lines = f.readlines()
    for idx, val in zip(param_indices, param_values):
        line_idx = 7 + idx
        lines[line_idx] = f"{param_labels[idx]}1,{val:.3e}\n"
    with open(output_k_file, 'w') as f:
        f.writelines(lines)

def run_lsdyna(input_file, ncpu, memory):
    subprocess.run([
        lsdyna_executable,
        f"i={input_file}",
        f"ncpu={ncpu}",
        f"memory={memory}"
    ])

def extract_z_disp(nodout_path, num_nodes=54, line_start=68, line_offset=60):
    with open(nodout_path, 'r') as f:
        lines = f.readlines()
    z_disp = []
    idx_start = line_start - 1
    while idx_start + num_nodes <= len(lines):
        block = [float(lines[idx_start + i][46:58]) for i in range(num_nodes)]
        z_disp.append(block)
        idx_start += line_offset
    return np.array(z_disp)  # Now shape (n_time_steps, n_nodes)

# Main workflow
results = {}
for param in selected_params:
    idx = ['A', 'B', 'n', 'C', 'm', 'D1', 'D2', 'D3', 'D4', 'D5'].index(param)
    for adj, label in zip([1 + param_adjustment, 1 - param_adjustment], ['max', 'min']):
        modified_params = params_true.copy()
        modified_params[idx] *= adj
        
        folder_name = f"{param}_{label}"
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Modify k-file
        output_k_file = os.path.join(folder_path, f"Run_{param}_{label}.k")
        modify_k_file(original_k_file, output_k_file, [idx], [modified_params[idx]])
        
        # Run LS-DYNA
        cwd = os.getcwd()
        os.chdir(folder_path)
        run_lsdyna(output_k_file, ncpu, memory)
        os.chdir(cwd)
        
        # Extract results
        nodout_path = os.path.join(folder_path, 'nodout')
        z_disp = extract_z_disp(nodout_path)
        results[f"{param}_{label}"] = z_disp

        # Save extracted displacement to data folder (txt and csv)
        txt_save_path = os.path.join(data_dir, f"{param}_{label}_z_disp.txt")
        np.savetxt(txt_save_path, z_disp.flatten(), fmt='%.6e')
        
        csv_save_path = os.path.join(data_dir, f"{param}_{label}_z_disp.csv")
        pd.DataFrame(z_disp.T).to_csv(csv_save_path, index=False, header=False)

# Load saved data and Plotting
plt.figure(figsize=(15, 8))

diffs = {}
metrics = {'param': [], 'MSE': [], 'RMSE': [], 'Percent RMSE': [], 'NRMSE': []}

for i, param in enumerate(selected_params):
    z_disp_max = np.loadtxt(os.path.join(data_dir, f"{param}_max_z_disp.txt")).reshape(-1, 54)
    z_disp_min = np.loadtxt(os.path.join(data_dir, f"{param}_min_z_disp.txt")).reshape(-1, 54)
    diff = (z_disp_max - z_disp_min).flatten()
    diffs[param] = diff

    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    baseline = np.mean(z_disp_min)
    rmse_pct = np.abs(rmse / baseline) * 100
    nrmse = rmse / (z_disp_min.max() - z_disp_min.min())

    metrics['param'].append(param)
    metrics['MSE'].append(mse)
    metrics['RMSE'].append(rmse)
    metrics['Percent RMSE'].append(rmse_pct)
    metrics['NRMSE'].append(nrmse)

    plt.subplot(2, 3, i+1)
    plt.plot(diff)
    plt.title(f'{param} Parameter Difference')
    plt.xlabel('Sample Index')
    plt.ylabel('Difference')

# Bar plots
plt.subplot(2, 3, 4)
plt.bar(metrics['param'], metrics['MSE'])
plt.title('Mean Square Error (MSE)')
plt.ylabel('MSE')

plt.subplot(2, 3, 5)
plt.bar(metrics['param'], metrics['RMSE'])
plt.title('Root Mean Square Error (RMSE)')
plt.ylabel('RMSE')

plt.subplot(2, 3, 6)
plt.bar(metrics['param'], metrics['NRMSE'])
plt.title('Normalized RMSE (NRMSE)')
plt.ylabel('NRMSE')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'difference_summary.png'), dpi=300, bbox_inches='tight')
plt.show()
