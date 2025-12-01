import os

# ================= Configuration Section =================
output_filename = "sph_projectile.k"

# Geometry Parameters (Center and Radius)
cx, cy, cz = 0.0, 0.0, 0.251
radius = 0.25

# Grid Resolution (Matches NumX, NumY, NumZ in PrePost)
# Represents how many divisions exist along the bounding box edge length (2*Radius)
num_x_grid = 11
num_y_grid = 11
num_z_grid = 11

# Material Properties
density = 7.8  # Density

# ID Settings (Start NID, Start PID)
start_nid = 1
start_eid = 1
part_id = 1

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
    else:
        print("[warn] Could not determine script directory; using current CWD", flush=True)
    return os.getcwd()

# ================= Main Logic =================

def generate_sph_sphere():
    # 1. Calculate Bounding Box and Spacing
    box_len_x = 2.0 * radius
    box_len_y = 2.0 * radius
    box_len_z = 2.0 * radius
    
    dx = box_len_x / num_x_grid
    dy = box_len_y / num_y_grid
    dz = box_len_z / num_z_grid
    
    # Calculate Single Particle Mass
    # Mass = Density * Volume of one grid cell
    particle_volume = dx * dy * dz
    particle_mass = density * particle_volume
    
    # Determine Starting Point (Bottom-Left-Rear corner of the box)
    x_min = cx - radius
    y_min = cy - radius
    z_min = cz - radius
    
    print(f"--- SPH Sphere Generation (Column-First Order) ---")
    print(f"Spacing (dx, dy, dz): {dx:.5f}, {dy:.5f}, {dz:.5f}")
    print(f"Particle Mass:        {particle_mass:.6e}")
    
    valid_particles = []
    
    # 2. Loop Generation (Order: X -> Y -> Z)
    # This matches LS-PrePost logic: fix X and Y, then fill Z (height).
    for i in range(num_x_grid):
        x_pos = x_min + (i + 0.5) * dx
        
        # Optimization: Skip if the X-slice is completely outside the radius
        if abs(x_pos - cx) > radius: continue

        for j in range(num_y_grid):
            y_pos = y_min + (j + 0.5) * dy
            
            # Optimization: Skip if the XY-column is outside the radius
            if (x_pos - cx)**2 + (y_pos - cy)**2 > radius**2: continue

            for k in range(num_z_grid):
                z_pos = z_min + (k + 0.5) * dz
                
                # Core Distance Check: Keep particle only if inside sphere
                dist_sq = (x_pos - cx)**2 + (y_pos - cy)**2 + (z_pos - cz)**2
                
                if dist_sq <= radius**2:
                    valid_particles.append((x_pos, y_pos, z_pos))

    print(f"Generated {len(valid_particles)} particles inside the sphere.")
    
    # 3. Write to .k file
    with open(output_filename, 'w') as f:
        f.write("*KEYWORD\n")
        
        # Write Nodes
        f.write("*NODE\n")
        current_nid = start_nid
        for x, y, z in valid_particles:
            f.write(f"{current_nid:8d}{x:16.6f}{y:16.6f}{z:16.6f}\n")
            current_nid += 1
            
        # Write Elements
        f.write("*ELEMENT_SPH\n")
        current_eid = start_eid
        current_nid_ref = start_nid
        
        for _ in valid_particles:
            # Format: EID, PID, NID, MASS
            f.write(f"{current_eid:8d}{part_id:8d}{current_nid_ref:8d}{particle_mass:16.6e}\n")
            current_eid += 1
            current_nid_ref += 1
            
        f.write("*END\n")
    
    print(f"Done! File saved to: {output_filename}")

if __name__ == "__main__":
    chdir_to_script_dir()
    generate_sph_sphere()
