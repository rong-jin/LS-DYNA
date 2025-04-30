# delete_d3plot_files.py

A simple Python utility to recursively locate and delete all files beginning with the prefix `d3plot` in a specified directory tree.

---

## Overview

This script scans a target directory (and all its subdirectories) for files whose names start with `d3plot`, then attempts to remove each matching file. It logs both successful deletions and any errors encountered.

## Prerequisites

- Python 3.6 or higher
- No external libraries required (uses only the standard library)

## Installation

1. Clone or download this repository.
2. Ensure you have Python 3 installed.
3. Place `delete_d3plot_files.py` in the directory from which you want to run the cleanup, or modify its path accordingly.

## Usage

```bash
python delete_d3plot_files.py
```

By default, the script uses its own location as the root directory:

1. Determines the directory containing `delete_d3plot_files.py`.
2. Changes the working directory to this location for consistent relative paths.
3. Recursively walks through every subdirectory.
4. Deletes any file whose name starts with `d3plot`.

### Custom Root Directory

To modify the root directory (other than the script’s own directory), you can:

1. Import and call the function directly in another Python script:

    ```python
    from delete_d3plot_files import delete_d3plot_files

    # Replace '/path/to/target' with your desired root path
    delete_d3plot_files('/path/to/target')
    ```

2. Alternatively, edit the `current_dir` assignment in the `if __name__ == "__main__"` block to point to your custom path.

## Function Details

- `delete_d3plot_files(root_dir: str)`
  - **Parameters**:
    - `root_dir`: Path to the directory tree to scan.
  - **Behavior**:
    1. Uses `os.walk` to iterate through `root_dir` and its subdirectories.
    2. Checks each filename for the prefix `d3plot`.
    3. Attempts to remove matching files via `os.remove()`.
    4. Prints a confirmation on success or an error message on failure.

## Logging and Error Handling

- **Success**: Prints `Deleted file: <full path>` for each removed file.
- **Failure**: Prints `Error deleting <full path>: <exception>` if an exception occurs (e.g., permission denied).

## Example

```bash
# Navigate to the project folder
cd path/to/project

# Run the cleanup
python delete_d3plot_files.py
```

This will remove all `d3plot*` files under `path/to/project` recursively.

## License

Released under the MIT License.

---

*For questions or feedback, please contact the repository maintainer.*
