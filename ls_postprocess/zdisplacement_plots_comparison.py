"""
Author: Rong Jin, University of Kentucky
Date: 04-25-2025
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------------------------------------------------
# 1. Define function to extract field data from LS-DYNA nodout file
# -------------------------------------------------------------------
def extract_nodout_field_data(
    file_path: str,
    start_line: int,
    line_offset: int,
    range_length: int,
    field: str
) -> list[list[str]]:
    """
    Reads an ASCII 'nodout' file and returns repeated blocks
    of the specified field (e.g., 'x_coor', 'z_disp').
    """
    fields = [
        "nodal_point", "x_disp", "y_disp", "z_disp",
        "x_vel", "y_vel", "z_vel",
        "x_accl", "y_accl", "z_accl",
        "x_coor", "y_coor", "z_coor"
    ]
    if field not in fields:
        raise ValueError(f"Field '{field}' is invalid. Valid options: {fields}")

    # Compute string-slice indices for this field
    if field == "nodal_point":
        start_idx, end_idx = 0, 10
    else:
        idx = fields.index(field)
        start_idx = 10 + (idx - 1) * 12
        end_idx   = start_idx + 12

    with open(file_path, 'r') as f:
        lines = f.readlines()

    blocks = []
    curr = start_line
    # Loop over blocks until you run out of lines
    while curr + range_length - 1 <= len(lines):
        blk = []
        for i in range(curr - 1, curr - 1 + range_length):
            raw = lines[i][start_idx:end_idx].strip()
            # Insert missing 'e' for scientific notation if needed
            if 'e' not in raw.lower():
                for j in range(1, len(raw)):
                    if raw[j] in ['+', '-']:
                        raw = f"{raw[:j]}e{raw[j:]}"
                        break
            try:
                blk.append(f"{float(raw):.6e}")
            except ValueError:
                print(f"Warning: failed to parse line {i+1}: '{raw}'")
        blocks.append(blk)
        curr += line_offset

    return blocks

# -------------------------------------------------------------------
# 2. Create directories for CSV data and plot images
# -------------------------------------------------------------------
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

data_dir = os.path.join(base_dir, 'data')
plot_dir = os.path.join(base_dir, 'plot')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# -------------------------------------------------------------------
# 3. Extract nodout data and save to CSV
# -------------------------------------------------------------------
case_dirs    = ["C_0_0075", "C_0_0105", "D4_0_3", "D4_0_6"]
fields       = ["z_disp", "x_coor"]
start_line   = 68
line_offset  = 60
range_length = 54

for case in case_dirs:
    nodout_path = os.path.join(base_dir, case, "nodout")
    if not os.path.isfile(nodout_path):
        print(f"Error: nodout file not found: {nodout_path}")
        continue

    for field in fields:
        blocks = extract_nodout_field_data(
            file_path    = nodout_path,
            start_line   = start_line,
            line_offset  = line_offset,
            range_length = range_length,
            field        = field
        )
        df = pd.DataFrame(blocks).transpose()
        csv_name = f"{case}_{field}.csv"
        df.to_csv(os.path.join(data_dir, csv_name),
                  index=False, header=False)
        print(f"Saved CSV: {case}_{field}.csv")

# -------------------------------------------------------------------
# 4. Prepare plotting parameters
# -------------------------------------------------------------------
cases_C    = ["C_0_0075", "C_0_0105"]
cases_D4   = ["D4_0_3",   "D4_0_6"]
fields_map = {"x": "x_coor", "z": "z_disp"}

num_steps    = 121
times        = np.linspace(0, 1.2e-5, num_steps)
time_indices = list(range(0, num_steps, 10))[:13]

# Use a qualitative 'tab20' palette for up to 20 distinct colors
N      = len(time_indices)
cmap   = plt.get_cmap('tab20', N)
colors = [cmap(i) for i in range(N)]

line_styles = {
    "C_0_0075": "solid", "C_0_0105": "dashed",
    "D4_0_3":   "solid", "D4_0_6":   "dashed"
}

# Map folder names to friendly legend labels
label_map = {
    "C_0_0075": "C = 0.0075",
    "C_0_0105": "C = 0.0105",
    "D4_0_3":   "D4 = 0.3",
    "D4_0_6":   "D4 = 0.6",
}

# -------------------------------------------------------------------
# 5a. Function to plot raw (unsymmetrized) data
# -------------------------------------------------------------------
def plot_raw(cases: list[str], title: str, save_name: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    for ti, idx in enumerate(time_indices):
        for case in cases:
            x = pd.read_csv(os.path.join(data_dir, f"{case}_{fields_map['x']}.csv"),
                            header=None).values[:, idx]
            z = pd.read_csv(os.path.join(data_dir, f"{case}_{fields_map['z']}.csv"),
                            header=None).values[:, idx]

            order = np.argsort(x)
            ax.plot(x[order], z[order],
                    color=colors[ti],
                    linestyle=line_styles[case],
                    linewidth=2.0, alpha=0.8)

    # Time legend (color)
    time_handles = [Line2D([0],[0], color=colors[i], lw=2) for i in range(N)]
    time_labels  = [f"{times[idx]:.1e} s" for idx in time_indices]
    leg1 = ax.legend(time_handles, time_labels,
                     title="Time", loc="center right",
                     ncol=4, fontsize="small", frameon=True)
    ax.add_artist(leg1)

    # Case legend (line style)
    case_handles = [Line2D([0],[0], color='k', linestyle=line_styles[c], lw=2)
                    for c in cases]
    case_labels = [label_map[c] for c in cases]
    ax.legend(case_handles, case_labels,
              title="Case", loc="lower right",
              fontsize="small", frameon=True)

    ax.set_xlabel("x_coordinate (cm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("z_displacement (cm)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(length=4, labelsize=12)
    plt.tight_layout()

    fig.savefig(os.path.join(plot_dir, save_name), dpi=300)
    plt.close(fig)
    print(f"✔ Saved raw plot: {save_name}")

# -------------------------------------------------------------------
# 5b. Function to plot symmetric (mirrored) data
# -------------------------------------------------------------------
def plot_symmetric(cases: list[str], title: str, save_name: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    for ti, idx in enumerate(time_indices):
        for case in cases:
            x = pd.read_csv(os.path.join(data_dir, f"{case}_{fields_map['x']}.csv"),
                            header=None).values[:, idx]
            z = np.abs(pd.read_csv(os.path.join(data_dir, f"{case}_{fields_map['z']}.csv"),
                                   header=None).values[:, idx])

            order   = np.argsort(x)
            x_s, z_s = x[order], z[order]
            x_neg   = -x_s[1:][::-1]
            z_neg   =  z_s[1:][::-1]

            x_full = np.concatenate((x_neg, x_s))
            z_full = np.concatenate((z_neg, z_s))

            ax.plot(x_full, z_full,
                    color=colors[ti],
                    linestyle=line_styles[case],
                    linewidth=2.5, alpha=0.8)

    # Time legend
    time_handles = [Line2D([0],[0], color=colors[i], lw=2) for i in range(N)]
    time_labels  = [f"{times[idx]:.1e} s" for idx in time_indices]
    leg1 = ax.legend(time_handles, time_labels,
                     title="Time", loc="upper right",
                     ncol=4, fontsize="small", frameon=True)
    ax.add_artist(leg1)

    # Case legend
    case_handles = [Line2D([0],[0], color='k', linestyle=line_styles[c], lw=2)
                    for c in cases]
    case_labels = [label_map[c] for c in cases]
    ax.legend(case_handles, case_labels,
              title="Case", loc="lower right",
              fontsize="small", frameon=True)

    ax.set_xlabel("x_coordinate (cm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("|z_displacement| (cm)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(length=4, labelsize=12)
    plt.tight_layout()

    fig.savefig(os.path.join(plot_dir, save_name), dpi=300)
    plt.close(fig)
    print(f"Saved symmetric plot: {save_name}")

# -------------------------------------------------------------------
# 6. Generate and save all four plots
# -------------------------------------------------------------------
plot_raw(      cases_C,  "C Cases: Raw Backface Displacement",       "C_cases_raw.png")
plot_symmetric(cases_C,  "C Cases: Symmetric Backface Displacement", "C_cases_symmetric.png")
plot_raw(      cases_D4, "D4 Cases: Raw Backface Displacement",      "D4_cases_raw.png")
plot_symmetric(cases_D4, "D4 Cases: Symmetric Backface Displacement","D4_cases_symmetric.png")
