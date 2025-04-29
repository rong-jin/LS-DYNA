import os

def delete_d3plot_files(root_dir):
    """
    遍历 root_dir 及其所有子目录，删除所有以 "d3plot" 开头的文件。
    
    Args:
        root_dir (str): 要遍历的根目录路径。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("d3"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前工作目录
    os.chdir(current_dir)
    delete_d3plot_files(current_dir)
