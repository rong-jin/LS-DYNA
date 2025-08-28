"""
Author: Rong Jin, University of Kentucky
Date: 04-30-2025
"""
import os
def delete_d3plot_files(root_dir):
    """
    Recursively traverse the specified directory and its subdirectories, removing all files that start with the prefix "d3plot".

    Parameters:
        root_dir (str): The path to the root directory to scan for d3plot files.

    The function walks through the directory tree, checks each file name,
    and attempts to delete any matching file. Successful deletions and errors
    are logged to the console.
    """
    # Walk through the directory tree from the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Iterate over each file in the current directory
        for filename in filenames:
            # Identify files whose names start with the prefix "d3plot"
            if filename.startswith("d3plot"):
                # Construct the full path to the file
                file_path = os.path.join(dirpath, filename)
                try:
                    # Attempt to remove the file from the filesystem
                    os.remove(file_path)
                    # Log successful deletion
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    # Log any errors encountered during file removal
                    print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    # Determine the directory where this script resides
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the script directory for consistency
    os.chdir(current_dir)
    # Invoke the deletion function on the current directory
    delete_d3plot_files(current_dir)  # Remove all d3plot files under this directory and its subdirectories


