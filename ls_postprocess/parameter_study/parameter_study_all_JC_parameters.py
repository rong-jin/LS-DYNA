"""
Author: Rong Jin, University of Kentucky
Date: 04-29-2025
"""
import os
import numpy as np
import shutil
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
# Constants and paths
# ----------------------------------------------------------------------------
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

mat_file = os.path.join(base_dir, 'MAT.txt')
original_k_file = os.path.join(base_dir, 'Run.k')
data_dir = os.path.join(base_dir, 'data')
plot_dir = os.path.join(base_dir, 'plot')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
lsdyna_executable = r"C:\\Program Files\\ANSYS Inc\\v251\\ansys\\bin\\winx64\\lsdyna_dp.exe"
ncpu = 4
memory = '256m'

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
param_labels = ['RA', 'RB', 'Rn', 'RC', 'Rm', 'RD1', 'RD2', 'RD3', 'RD4', 'RD5']
with open(mat_file, 'r') as f:
    params_true = [float(line.strip()) for line in f]

param_adjustment = 0.1  # 50%
selected_params = ['A', 'B', 'n', 'C', 'm', 'D1', 'D2', 'D3', 'D4', 'D5']

# ----------------------------------------------------------------------------
# Utility: delete files starting with 'd3' in a given folder
# ----------------------------------------------------------------------------
def cleanup_d3_files(folder_path: str):
    if not os.path.isdir(folder_path):
        return
    for fname in os.listdir(folder_path):
        if fname.lower().startswith('d3'):
            full = os.path.join(folder_path, fname)
            try:
                os.remove(full)
                print(f"Deleted: {full}")
            except Exception as e:
                print(f"Failed to delete {full}: {e}")

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
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
        block = [float(lines[idx_start + i][34:46]) for i in range(num_nodes)]
        z_disp.append(block)
        idx_start += line_offset
    return np.array(z_disp)

# ----------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------
results = {}
prev_folder = None
for param in selected_params:
    idx = ['A', 'B', 'n', 'C', 'm', 'D1', 'D2', 'D3', 'D4', 'D5'].index(param)
    for adj, label in zip([1 + param_adjustment, 1 - param_adjustment], ['max', 'min']):
        # cleanup previous folder d3* files
        if prev_folder:
            cleanup_d3_files(prev_folder)

        modified_params = params_true.copy()
        modified_params[idx] *= adj

        folder_name = f"{param}_{label}"
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        output_k_file = os.path.join(folder_path, f"Run_{param}_{label}.k")
        modify_k_file(original_k_file, output_k_file, [idx], [modified_params[idx]])

        cwd = os.getcwd()
        os.chdir(folder_path)
        run_lsdyna(output_k_file, ncpu, memory)
        os.chdir(cwd)

        nodout_path = os.path.join(folder_path, 'nodout')
        z_disp = extract_z_disp(nodout_path)
        results[f"{param}_{label}"] = z_disp

        txt_save_path = os.path.join(data_dir, f"{param}_{label}_z_disp.txt")
        np.savetxt(txt_save_path, z_disp.flatten(), fmt='%.6e')

        csv_save_path = os.path.join(data_dir, f"{param}_{label}_z_disp.csv")
        pd.DataFrame(z_disp.T).to_csv(csv_save_path, index=False, header=False)

        prev_folder = folder_path
