import os
import math

# ================= Configuration Section =================
output_filename = "geo.k"

# --- 1. SPH Box Configuration ---
box_cfg = {
    "enable": True,
    # Geometry (PMin, PMax)
    "x_min": -2.54, "x_max": 2.54,
    "y_min": -2.54, "y_max": 2.54,
    "z_min": -0.635, "z_max": 0.0,
    # Particle Count
    "num_x": 111, "num_y": 111, "num_z": 14,
    # Material & ID
    "density": 7.8,
    "part_id": 5000001,
    "start_nid": 5000001,  # High ID range for Box
    "start_eid": 5000001
}

# --- 2. SPH Sphere Configuration ---
sphere_cfg = {
    "enable": True,
    # Geometry
    "cx": 0.0, "cy": 0.0, "cz": 0.251,
    "radius": 0.25,
    # Grid Resolution (Divisions along 2*Radius)
    "num_x_grid": 11, "num_y_grid": 11, "num_z_grid": 11,
    # Material & ID
    "density": 7.8,
    "part_id": 1,
    "start_nid": 1,        # Low ID range for Sphere
    "start_eid": 1
}

# ================= Helper Functions =================

def chdir_to_script_dir(verbose: bool = True) -> str:
    """Sets the current working directory to the script's location."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    if script_dir and os.path.isdir(script_dir):
        os.chdir(script_dir)
        if verbose:
            print(f"[info] CWD set to script dir: {script_dir}", flush=True)
    return os.getcwd()

# ================= Generation Logic =================

def calculate_box_particles(cfg):
    """Calculates coordinates and mass for a Box shape."""
    if not cfg["enable"]: return [], 0.0
    
    # Unpack config
    x_min, x_max = cfg["x_min"], cfg["x_max"]
    y_min, y_max = cfg["y_min"], cfg["y_max"]
    z_min, z_max = cfg["z_min"], cfg["z_max"]
    nx, ny, nz = cfg["num_x"], cfg["num_y"], cfg["num_z"]
    
    # Spacing
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz
    
    # Mass Calculation
    total_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    total_particles = nx * ny * nz
    particle_mass = (total_volume * cfg["density"]) / total_particles
    
    coords = []
    # Column-First Order: X (Outer) -> Y (Middle) -> Z (Inner)
    for i in range(nx):
        x_pos = x_min + (i + 0.5) * dx
        for j in range(ny):
            y_pos = y_min + (j + 0.5) * dy
            for k in range(nz):
                z_pos = z_min + (k + 0.5) * dz
                coords.append((x_pos, y_pos, z_pos))
                
    return coords, particle_mass

def calculate_sphere_particles(cfg):
    """Calculates coordinates and mass for a Sphere shape."""
    if not cfg["enable"]: return [], 0.0
    
    cx, cy, cz = cfg["cx"], cfg["cy"], cfg["cz"]
    r = cfg["radius"]
    nx, ny, nz = cfg["num_x_grid"], cfg["num_y_grid"], cfg["num_z_grid"]
    
    # Bounding box dimensions
    box_len = 2.0 * r
    dx = box_len / nx
    dy = box_len / ny
    dz = box_len / nz
    
    # Mass Calculation (Density * Single Cell Volume)
    particle_mass = cfg["density"] * (dx * dy * dz)
    
    x_min, y_min, z_min = cx - r, cy - r, cz - r
    coords = []
    
    # Column-First Order: X (Outer) -> Y (Middle) -> Z (Inner)
    for i in range(nx):
        x_pos = x_min + (i + 0.5) * dx
        if abs(x_pos - cx) > r: continue
        
        for j in range(ny):
            y_pos = y_min + (j + 0.5) * dy
            if (x_pos - cx)**2 + (y_pos - cy)**2 > r**2: continue
            
            for k in range(nz):
                z_pos = z_min + (k + 0.5) * dz
                dist_sq = (x_pos - cx)**2 + (y_pos - cy)**2 + (z_pos - cz)**2
                
                if dist_sq <= r**2:
                    coords.append((x_pos, y_pos, z_pos))
                    
    return coords, particle_mass

# ================= Main Execution =================

def main():
    chdir_to_script_dir()
    
    print("--- Generating Combined SPH Geometry ---")
    
    # 1. Calculate Data in Memory
    box_coords, box_mass = calculate_box_particles(box_cfg)
    sph_coords, sph_mass = calculate_sphere_particles(sphere_cfg)
    
    print(f"Box Particles:    {len(box_coords)}")
    print(f"Sphere Particles: {len(sph_coords)}")
    
    # 2. Check for ID overlaps (Optional safety check)
    if box_cfg["enable"] and sphere_cfg["enable"]:
        box_end_nid = box_cfg["start_nid"] + len(box_coords)
        sph_end_nid = sphere_cfg["start_nid"] + len(sph_coords)
        # Check simple overlap logic
        if (box_cfg["start_nid"] < sph_end_nid) and (sphere_cfg["start_nid"] < box_end_nid):
            print("[WARN] Node IDs might overlap! Check 'start_nid' settings.")

    # 3. Write to File
    with open(output_filename, 'w') as f:
        f.write("*KEYWORD\n")
        
        # ---------------- Write Nodes ----------------
        f.write("*NODE\n")
        
        # Write Box Nodes
        if box_cfg["enable"]:
            nid = box_cfg["start_nid"]
            for x, y, z in box_coords:
                f.write(f"{nid:8d}{x:16.6f}{y:16.6f}{z:16.6f}\n")
                nid += 1
                
        # Write Sphere Nodes
        if sphere_cfg["enable"]:
            nid = sphere_cfg["start_nid"]
            for x, y, z in sph_coords:
                f.write(f"{nid:8d}{x:16.6f}{y:16.6f}{z:16.6f}\n")
                nid += 1
        
        # ---------------- Write Elements ----------------
        f.write("*ELEMENT_SPH\n")
        
        # Write Box Elements
        if box_cfg["enable"]:
            eid = box_cfg["start_eid"]
            nid = box_cfg["start_nid"]
            pid = box_cfg["part_id"]
            for _ in box_coords:
                f.write(f"{eid:8d}{pid:8d}{nid:8d}{box_mass:16.6e}\n")
                eid += 1
                nid += 1
                
        # Write Sphere Elements
        if sphere_cfg["enable"]:
            eid = sphere_cfg["start_eid"]
            nid = sphere_cfg["start_nid"]
            pid = sphere_cfg["part_id"]
            for _ in sph_coords:
                f.write(f"{eid:8d}{pid:8d}{nid:8d}{sph_mass:16.6e}\n")
                eid += 1
                nid += 1
        
        f.write("*END\n")
        
    print(f"Done! Combined geometry saved to: {output_filename}")

if __name__ == "__main__":
    main()
