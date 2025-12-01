import os

# ================= Configuration Section =================
output_filename = "sph_box.k"

# Geometry Range (Matches PMin and PMax in PrePost)
x_min, x_max = -2.54, 2.54
y_min, y_max = -2.54, 2.54
z_min, z_max = -0.635, 0.0

# Particle Count Definition (Matches NumX, NumY, NumZ)
num_x = 111
num_y = 111
num_z = 14

# Material Properties
density = 7.8  # Density (e.g., g/cm^3)

# ID Settings (Matches Start NID, Start PID)
start_nid = 5000001
start_eid = 5000001
part_id = 5000001

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

def generate_sph_box():
    # 1. Calculate Spacing (dx, dy, dz)
    dx = (x_max - x_min) / num_x
    dy = (y_max - y_min) / num_y
    dz = (z_max - z_min) / num_z
    
    # 2. Calculate Single Particle Mass
    # Mass = Density * Volume of one grid cell
    particle_volume = dx * dy * dz
    particle_mass = density * particle_volume
    
    total_particles = num_x * num_y * num_z

    print(f"--- SPH Box Generation (Column-First Order) ---")
    print(f"Total Particles: {total_particles}")
    print(f"Particle Mass:   {particle_mass:.6e}")
    
    with open(output_filename, 'w') as f:
        f.write("*KEYWORD\n")
        
        # Write Nodes
        f.write("*NODE\n")
        
        current_nid = start_nid
        coords = []

        # Loop Order: X -> Y -> Z
        # This order ensures nodes are generated in vertical columns first,
        # matching the behavior of LS-PrePost.
        for i in range(num_x):
            x_pos = x_min + (i + 0.5) * dx
            for j in range(num_y):
                y_pos = y_min + (j + 0.5) * dy
                for k in range(num_z):
                    z_pos = z_min + (k + 0.5) * dz
                    
                    coords.append((current_nid, x_pos, y_pos, z_pos))
                    current_nid += 1
        
        # Flush coordinates to file
        for nid, x, y, z in coords:
            f.write(f"{nid:8d}{x:16.6f}{y:16.6f}{z:16.6f}\n")

        # Write Elements
        f.write("*ELEMENT_SPH\n")
        current_eid = start_eid
        current_nid_ref = start_nid
        
        # Loop through valid particles to create elements
        for _ in range(total_particles):
            # Format: EID, PID, NID, MASS
            f.write(f"{current_eid:8d}{part_id:8d}{current_nid_ref:8d}{particle_mass:16.6e}\n")
            current_eid += 1
            current_nid_ref += 1
            
        f.write("*END\n")

    print(f"Done! File saved to: {output_filename}")

if __name__ == "__main__":
    chdir_to_script_dir()
    generate_sph_box()
