import os
import sys
import random
import shutil
from pathlib import Path

def split_file(input_file, train_file, valid_file, train_ratio=0.9, seed=42):
    """Split a text file into training and validation sets."""
    print(f"Splitting {input_file} into training and validation sets...")
    print(f"Train ratio: {train_ratio:.2f}, Validation ratio: {1-train_ratio:.2f}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all lines from the input file
    print("Reading input file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines: {total_lines:,}")
    
    # Shuffle the lines
    print("Shuffling lines...")
    random.shuffle(lines)
    
    # Calculate split point
    train_size = int(total_lines * train_ratio)
    
    # Split the lines
    train_lines = lines[:train_size]
    valid_lines = lines[train_size:]
    
    print(f"Training set: {len(train_lines):,} lines ({len(train_lines)/total_lines:.1%})")
    print(f"Validation set: {len(valid_lines):,} lines ({len(valid_lines)/total_lines:.1%})")
    
    # Write training set
    print(f"Writing training set to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Write validation set
    print(f"Writing validation set to {valid_file}...")
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    
    # Calculate file sizes
    train_size_mb = os.path.getsize(train_file) / (1024 * 1024)
    valid_size_mb = os.path.getsize(valid_file) / (1024 * 1024)
    
    print(f"Training set size: {train_size_mb:.2f} MB")
    print(f"Validation set size: {valid_size_mb:.2f} MB")
    
    return len(train_lines), len(valid_lines)

def delete_files_in_directory(directory, pattern="*.txt"):
    """Delete files matching pattern in the specified directory."""
    print(f"Deleting {pattern} files in {directory}...")
    
    path = Path(directory)
    if not path.exists():
        print(f"Directory {directory} does not exist.")
        return
    
    files = list(path.glob(pattern))
    print(f"Found {len(files)} files matching pattern.")
    
    for file in files:
        print(f"Deleting {file}...")
        file.unlink()
    
    print(f"Deleted {len(files)} files.")

if __name__ == "__main__":
    # Input and output files
    input_file = "datasets/combined_swahili_corpus.txt"
    data_dir = "data"
    train_file = os.path.join(data_dir, "train.txt")
    valid_file = os.path.join(data_dir, "valid.txt")
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Delete existing files in data directory
    delete_files_in_directory(data_dir)
    
    # Split the corpus
    split_file(input_file, train_file, valid_file)
    
    print("Done!")
