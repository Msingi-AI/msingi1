import os
from pathlib import Path

def undo_file_copy():
    """
    Remove the files that were copied to the data folder.
    This undoes the operation performed by move_datasets_to_data.py
    """
    # List of files that were copied
    files_to_remove = [
        "kiswahili.txt",
        "utafiti.txt", 
        "parliament_clean.txt",
        "mwongozo_clean.txt",
        "katiba_clean.txt",
        "adero_clean.txt",
        "wiki_sw_combined_clean.txt"
    ]
    
    data_dir = Path("data")
    
    # Remove each file
    for filename in files_to_remove:
        file_path = data_dir / filename
        
        # Check if file exists
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"Removed {filename} from data folder")
            except Exception as e:
                print(f"Error removing {filename}: {e}")
        else:
            print(f"File {filename} not found in data folder")

if __name__ == "__main__":
    undo_file_copy()
