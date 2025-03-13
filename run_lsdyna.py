"""
Author: Rong Jin, University of Kentucky
Date: 03-13-2025
Description: The function executes an LS-DYNA simulation using the specified input file and simulation parameters.
"""
import subprocess
import os

def run_lsdyna(input_file, ncpu, memory, dump_file=None, work_dir=None,
               dynasolver_path=r"C:\\Program Files\\ANSYS Inc\\v231\\ansys\\bin\\winx64\\lsdyna_dp.exe"):
    """
    Executes the LS-DYNA simulation using the specified parameters.

    Parameters:
        input_file (str): Full path to the input .k file.
        ncpu (int): Number of CPU cores to use.
        memory (str): Memory allocation size (e.g., "1024m").
        dump_file (str, optional): Dump file name if needed.
        work_dir (str, optional): Working directory to execute the command.
                                  If not provided, it defaults to the directory of input_file.
        dynasolver_path (str): Path to the LS-DYNA dynasolver.
                               Defaults to the specified path.
    
    Returns:
        None. The function prints the output or error messages.
    """
    try:
        # If work_dir is not specified, use the directory of the input file.
        if work_dir is None:
            work_dir = os.path.dirname(input_file)
        
        # Construct the LS-DYNA command with required parameters.
        command = [
            dynasolver_path,
            f"i={input_file}",
            f"ncpu={ncpu}",
            f"memory={memory}"
        ]
        
        # If a dump file is provided, add it to the command.
        if dump_file:
            command.append(f"R={dump_file}")
        
        # Execute the command in the specified working directory.
        result = subprocess.run(command, capture_output=True, text=True, cwd=work_dir)
        
        # Check if the command executed successfully.
        if result.returncode == 0:
            print(f"Command executed successfully: {input_file}")
            print(result.stdout)
        else:
            print(f"Command execution failed: {input_file}")
            print(result.stderr)
    
    except Exception as e:
        print(f"Error executing command: {e}")
