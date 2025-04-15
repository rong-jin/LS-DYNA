"""
Author: Rong Jin, University of Kentucky
Date: 04-15-2025
Description: This code reads experimental data from a MATLAB file ("Mg_PS_Shot9.mat") and simulation data from plain-text files ("Coordinates3" and "Displacements3"), processes both datasets by adjusting coordinates and constructing symmetric arrays, and then plots the experimental and simulation data on a single figure using a 256-color HSV colormap. Experimental data are plotted with dashed lines while simulation data are shown with solid lines, and the plot is annotated with axis labels, legends, and a title to compare the two datasets.
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

##############################
# 1. Read the Mg_PS_Shot9.mat file and plot experimental data
##############################

# Set the working directory to the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

plt.rcParams['font.family'] = 'Times New Roman'

# Load the MAT file and use squeeze_me to remove extra dimensions (cell data becomes a 1-D object array)
data_exp = sio.loadmat('Mg_PS_Shot9.mat', squeeze_me=True)
w_data = data_exp['w']  # e.g., a 1-D array where each element is a frame matrix
x_data = data_exp['x']

# In MATLAB, iStart = 41 and iEnd = 96, which correspond to Python indices 40 and 95, respectively.
iStart = 40  # Corresponds to the 41st frame in MATLAB
iEnd   = 95  # Corresponds to the 96th frame in MATLAB

# Check if the number of frames is sufficient
if np.ndim(w_data) == 0 or len(w_data) <= iEnd:
    raise ValueError(f"Insufficient number of frames in w_data. Current frame count: {len(w_data) if hasattr(w_data, '__len__') else 'unknown'}, at least {iEnd+1} frames required.")

# Find the reference position with the maximum displacement from the iEnd frame
w_end = w_data[iEnd]
max_disp = np.max(w_end)
row_inds, col_inds = np.where(w_end == max_disp)
# Take the first occurrence (reference point)
row = row_inds[0]
col = col_inds[0]

# Compute the reference value: the maximum displacement in the reference row of the iStart frame
w_ref = np.max(w_data[iStart][row, :])

# Extract the corresponding row data (displacement and coordinates) from frames iStart to iEnd,
# and adjust the coordinates relative to the reference point
Disp_exp = []
Coord_exp = []
for i in range(iStart, iEnd + 1):
    disp_i = w_data[i][row, :]
    # Coordinate adjustment: subtract the reference x value from the current frame's x data and offset by 0.3
    coord_i = x_data[i][row, :] - x_data[i][row, col] - 0.3
    Disp_exp.append(disp_i)
    Coord_exp.append(coord_i)
    
# Convert the lists to numpy arrays with dimensions: (number of frames, number of points)
Disp_exp = np.array(Disp_exp)
Coord_exp = np.array(Coord_exp)

# Set the time indices for plotting experimental data.
# In MATLAB, T_exp = [1 6 11 16 21] (1-based), which corresponds to T_exp = [0, 5, 10, 15, 20] in Python.
T_exp = [0, 5, 10, 15, 20]

##############################
# 2. Read the Coordinates1 and Displacements1 files and generate simulation data
##############################

# Load the simulation data files (make sure the file variables are stored as 'Coordinates1' and 'Displacements1')
Coordinates1 = np.genfromtxt('Coordinates3', dtype=float)  # Alternatively, use 'Coordinates1', depending on your file format
Displacements1 = np.genfromtxt('Displacements3', dtype=float)

# Set parameters according to the MATLAB code
N_time = 101
# Calculate the number of nodes: number of rows in Displacements1 divided by N_time
N_node = int(Displacements1.shape[0] // N_time)

# Extract the initial coordinate for each node (corresponding to MATLAB: Coordinates1(1:N_time:end,2) * 10.0)
# Note: MATLAB's 2nd column corresponds to index 1 in Python
Coord0 = Coordinates1[0::N_time, 1] * 10.0  
# Calculate time (not used later, just for consistency with MATLAB)
Time = Coordinates1[0:N_time, 0].T * 1e6  

# Loop to calculate each node's displacement and coordinate information
Disp1_list = []
Coord1_list = []
for i in range(N_node):
    start = i * N_time
    end = (i + 1) * N_time
    # In MATLAB: Disp1(i,:) = -Displacements1((i-1)*N_time+1:i*N_time,2)' * 10.0
    disp_i = - Displacements1[start:end, 1].T * 10.0
    # In MATLAB: Coord1(i,:) = Coordinates1((i-1)*N_time+1:i*N_time,2)' * 10.0
    coord_i = Coordinates1[start:end, 1].T * 10.0
    Disp1_list.append(disp_i)
    Coord1_list.append(coord_i)

# Convert the lists to NumPy arrays with dimensions: (N_node, N_time)
Disp1 = np.array(Disp1_list)
Coord1 = np.array(Coord1_list)

# Next, perform data concatenation and sorting as in MATLAB:
# For displacement data:
#   tmp = [Coord0 Disp1]; sortrows(tmp,1); Disp1 = tmp(:,2:end);
#   Disp2 = flip(Disp1(2:end,:)); Disp = [Disp2; Disp1];
tmp_disp = np.column_stack((Coord0, Disp1))  # Concatenate to form an array of shape (N_node, 1+N_time)
# Sort by the first column
tmp_disp_sorted = tmp_disp[np.argsort(tmp_disp[:, 0]), :]
Disp1_sorted = tmp_disp_sorted[:, 1:]  # Extract the sorted displacement part
# Flip the displacement data (excluding the first node) upside down
Disp2 = np.flipud(Disp1_sorted[1:, :])
# Concatenate to obtain the final displacement data matrix
Disp = np.vstack((Disp2, Disp1_sorted))

# For coordinate data:
#   tmp = [Coord0 Coord1]; sortrows(tmp,1); Coord1 = tmp(:,2:end);
#   Coord2 = flip(Coord1(2:end,:)); Coord = [-Coord2; Coord1];
tmp_coord = np.column_stack((Coord0, Coord1))
tmp_coord_sorted = tmp_coord[np.argsort(tmp_coord[:, 0]), :]
Coord1_sorted = tmp_coord_sorted[:, 1:]
Coord2 = np.flipud(Coord1_sorted[1:, :])
Coord = np.vstack((-Coord2, Coord1_sorted))

# According to MATLAB, set the reference time index, N_ref = 14 (1-based)
N_ref = 14
# In MATLAB, T_exp = [1 6 11 16 21] corresponds to T_sim = N_ref + 2*(T_exp - 1)
# Calculate T_sim (note: MATLAB indexing is 1-based; subtract 1 when converting to Python)
T_exp_matlab = np.array([1, 6, 11, 16, 21])
T_sim_matlab = N_ref + 2 * (T_exp_matlab - 1)    # Results in [11, 21, 31, 41, 51]
T_sim = (T_sim_matlab - 1).astype(int)             # Converted to Python indices: [10, 20, 30, 40, 50]

##############################
# 3. Plot the experimental and simulation data in one figure for comparison
##############################

# Set up the color map with 256 colors
cnum = 256
colors = plt.cm.hsv(np.linspace(0, 1, cnum))

plt.figure(figsize=(6, 4.5))

# Plot experimental data (dashed lines)
# For each selected frame in T_exp, plot the corresponding data and calculate the color index using the color map
exp_handles = []
for i, idx in enumerate(T_exp):
    # Calculate the color index using the provided formula:
    alpha = int(round(1 + i * (cnum - 1) / len(T_exp))) - 1  # Convert to 0-index
    h, = plt.plot(Coord_exp[idx, :], Disp_exp[idx, :], '--', color=colors[alpha],
                  linewidth=3, label=f'{i} μs')
    exp_handles.append(h)

# Plot simulation data (solid lines) using the column indices specified in T_sim
for i, idx in enumerate(T_sim):
    alpha = int(round(1 + i * (cnum - 1) / len(T_exp))) - 1
    tmp_x = Coord[:, idx]
    tmp_y = Disp[:, idx]
    # Remove points where displacement is 0
    nonzero = tmp_y != 0
    plt.plot(tmp_x[nonzero], tmp_y[nonzero], '-', color=colors[alpha],
             linewidth=3)

plt.xlabel('Coordinate (mm)', fontsize=16, fontweight='bold')
plt.ylabel('Backface deflection (mm)', fontsize=16, fontweight='bold')
plt.xlim([-20, 20])
plt.xticks(np.arange(-20, 21, 5), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(exp_handles, ['0 μs', '1 μs', '2 μs', '3 μs', '4 μs'], loc='upper right', fontsize=12, frameon=True)
plt.title('Experimental (dashed) & Simulation (solid) Data', fontsize=16, fontweight='bold')
# plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
