import os
import sys
import time
import shutil

def append_files(file_paths, output_path):
    """Append multiple text files to an existing file."""
    print(f"Appending files to {output_path}...")
    
    # Check if the output file exists
    if not os.path.exists(output_path):
        print(f"Error: Output file {output_path} does not exist.")
        return
    
    original_size = os.path.getsize(output_path)
    print(f"Original file size: {original_size / (1024 * 1024):.2f} MB")
    
    total_input_size = sum(os.path.getsize(file_path) for file_path in file_paths)
    print(f"Total input size to append: {total_input_size / (1024 * 1024):.2f} MB")
    
    with open(output_path, 'a', encoding='utf-8') as outfile:
        for file_path in file_paths:
            print(f"Appending {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as infile:
                # Copy in chunks to handle large files efficiently
                shutil.copyfileobj(infile, outfile)
                # Add a newline between files to ensure separation
                outfile.write('\n')
    
    new_size = os.path.getsize(output_path)
    print(f"Updated file size: {new_size / (1024 * 1024):.2f} MB")
    print(f"Added {(new_size - original_size) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    # Files to append
    files_to_append = [
        "datasets/adero_clean.txt",
        "datasets/bible_clean.txt",
        "datasets/katiba_clean.txt",
        "datasets/kiswahili.txt",
        "datasets/koran.txt",
        "datasets/mwongozo_clean.txt",
        "datasets/utafiti.txt",
        "datasets/parliament_clean.txt"
    ]
    
    # Output combined file
    output_file = "datasets/combined_swahili_corpus.txt"
    
    # Append files
    append_files(files_to_append, output_file)
