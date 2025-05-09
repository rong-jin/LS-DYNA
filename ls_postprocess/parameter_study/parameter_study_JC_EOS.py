"""
Author: Rong Jin, University of Kentucky
Date: 05-09-2025

Parameter perturbation study for AZ31B using LS‑DYNA.
For each of 13 material parameters the script creates two cases (+10 %, –10 %),
runs them in parallel, extracts the back‑face Z–displacement from the resulting
nodout files, and stores the data in <BASE_DIR>/data.
"""

import os
import numpy as np
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ────────────────────────────────────────────────────────────────
# 0. Global paths and LS‑DYNA settings
# ────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                        # e.g. interactive console
    BASE_DIR = os.getcwd()

MAT_FILE     = os.path.join(BASE_DIR, "MAT.txt")
ORIG_K_FILE  = os.path.join(BASE_DIR, "Run.k")
DATA_DIR     = os.path.join(BASE_DIR, "data")
PLOT_DIR     = os.path.join(BASE_DIR, "plot")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

LSDYNA_EXEC  = (
    r"C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsdyna_dp.exe"
)
NCPU         = 1
MEMORY       = "256m"

MAX_PARALLEL = 4                         # Tune according to licence / cores
results_lock = threading.Lock()          # Protects the results dict
results      = {}                        # Holds displacement arrays

# ────────────────────────────────────────────────────────────────
# 1. Parameter names (readable) and corresponding keywords in Run.k
# ────────────────────────────────────────────────────────────────
PARM_NAMES = [
    "RA",  "RB",  "Rn",  "RC",  "Rm",
    "RD1", "RD2", "RD3", "RD4", "RD5",
    "RCS", "RS1", "RG"
]

K_KEYWORDS = [
    "RA1", "RB1", "Rn1", "RC1", "Rm1",
    "RD11", "RD21", "RD31", "RD41", "RD51",
    "RCS",  "RS1", "RG"
]

assert len(PARM_NAMES) == len(K_KEYWORDS) == 13, "Exactly 13 parameters expected."
IDX_MAP = {name: i for i, name in enumerate(PARM_NAMES)}

# ────────────────────────────────────────────────────────────────
# 2. Read baseline values and define perturbation factor
# ────────────────────────────────────────────────────────────────
params_true = np.loadtxt(MAT_FILE)
assert params_true.size == 13, "MAT.txt must contain 13 numeric values."

ADJ_FACTOR = 0.10                       # ±10 %
SELECTED   = PARM_NAMES[:]              # Modify the full set

# ────────────────────────────────────────────────────────────────
# 3. Helper functions
# ────────────────────────────────────────────────────────────────
def cleanup_d3_files(folder: str) -> None:
    """Remove every file starting with 'd3' in *folder*."""
    if not os.path.isdir(folder):
        return
    for fname in os.listdir(folder):
        if fname.lower().startswith("d3"):
            try:
                os.remove(os.path.join(folder, fname))
            except Exception as exc:
                print(f"[cleanup] Cannot delete {fname}: {exc}")

def modify_k_file(base_k: str, out_k: str, idx_list, val_list) -> None:
    """
    Replace lines 8–20 in *base_k* (0‑based 7–19) and write to *out_k*.
    Only the indices in *idx_list* are updated.
    """
    with open(base_k, "r") as fh:
        lines = fh.readlines()

    for idx, val in zip(idx_list, val_list):
        line_no = 7 + idx                       # 0‑based
        keyword = K_KEYWORDS[idx]
        lines[line_no] = f"{keyword},{val:.3e}\n"

    with open(out_k, "w") as fh_out:
        fh_out.writelines(lines)

def run_lsdyna(k_file: str, work_dir: str, ncpu: int = NCPU,
               memory: str = MEMORY) -> int:
    """Launch LS‑DYNA and return its exit code."""
    cmd = [
        LSDYNA_EXEC,
        f"i={k_file}",
        f"ncpu={ncpu}",
        f"memory={memory}"
    ]
    try:
        res = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True
        )
        if res.returncode != 0:
            # Write stderr to file for troubleshooting
            with open(os.path.join(work_dir, "stderr.txt"), "w") as err_file:
                err_file.write(res.stderr)
            print(f"[LS‑DYNA] Error in {work_dir}, see stderr.txt.")
        return res.returncode
    except Exception as exc:
        print(f"[LS‑DYNA] Subprocess launch failed: {exc}")
        return -1

def extract_z_disp(nodout_path: str, num_nodes: int = 54,
                   line_start: int = 68, step: int = 60) -> np.ndarray:
    """
    Extract Z‑displacement blocks from *nodout*.

    Parameters
    ----------
    nodout_path : str
        Path to the nodout file.
    num_nodes   : int
        Nodes per block (default 54).
    line_start  : int
        1‑based first line containing node data.
    step        : int
        Line increment between successive timesteps.

    Returns
    -------
    ndarray of shape (n_time, num_nodes)
    """
    with open(nodout_path, "r") as fh:
        lines = fh.readlines()

    blocks, idx = [], line_start - 1
    while idx + num_nodes <= len(lines):
        blocks.append(
            [float(lines[idx + i][34:46]) for i in range(num_nodes)]
        )
        idx += step
    return np.asarray(blocks)

# ────────────────────────────────────────────────────────────────
# 4. Parallel execution: create ±10 % cases and run them
# ────────────────────────────────────────────────────────────────
def run_single_case(par_name: str, par_idx: int,
                    scale: float, tag: str):
    """
    Prepare, execute and post‑process one perturbation case.

    Parameters
    ----------
    par_name : str
        Human‑readable parameter name, e.g. 'RB'.
    par_idx  : int
        Index in the parameter list.
    scale    : float
        Multiplicative factor to apply (1 ± ADJ_FACTOR).
    tag      : str
        'max' or 'min' suffix for folder names.
    """
    run_dir = os.path.join(BASE_DIR, f"{par_name}_{tag}")
    os.makedirs(run_dir, exist_ok=True)

    # Generate modified k file
    out_k = os.path.join(run_dir, f"Run_{par_name}_{tag}.k")
    vector = params_true.copy()
    vector[par_idx] *= scale
    modify_k_file(ORIG_K_FILE, out_k, [par_idx], [vector[par_idx]])

    # Cleanup old d3* files before starting the job
    cleanup_d3_files(run_dir)

    # Execute LS‑DYNA
    exit_code = run_lsdyna(out_k, run_dir)
    if exit_code != 0:
        raise RuntimeError(f"LS‑DYNA failed in {run_dir}")

    # Extract nodout results
    nodout = os.path.join(run_dir, "nodout")
    if not os.path.isfile(nodout):
        raise FileNotFoundError(f"nodout not found in {run_dir}")

    z_disp = extract_z_disp(nodout)

    # Remove large d3* files to free disk space
    cleanup_d3_files(run_dir)

    return f"{par_name}_{tag}", z_disp

# Submit all cases to the thread pool
futures = []
with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
    for name in PARM_NAMES:
        idx = IDX_MAP[name]
        for factor, label in ((1 + ADJ_FACTOR, "max"),
                              (1 - ADJ_FACTOR, "min")):
            futures.append(pool.submit(run_single_case, name, idx,
                                       factor, label))

    # Gather results as they finish
    for fut in as_completed(futures):
        key, z = fut.result()
        with results_lock:
            results[key] = z
        # Write plain‑text and CSV outputs
        np.savetxt(os.path.join(DATA_DIR, f"{key}_z_disp.txt"),
                   z.flatten(), fmt="%.6e")
        pd.DataFrame(z.T).to_csv(
            os.path.join(DATA_DIR, f"{key}_z_disp.csv"),
            index=False, header=False
        )
        print(f"[done] {key} written to data/")

print("All LS‑DYNA perturbation cases finished successfully.")
