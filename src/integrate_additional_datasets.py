import os
from pathlib import Path
import time
import random

def integrate_additional_datasets(input_files, output_file, shuffle=True):
    """
    Add additional datasets to the existing combined corpus.
    
    Args:
        input_files: List of input file paths to add
        output_file: Path to the existing corpus file
        shuffle: Whether to shuffle the lines (default: True)
    
    Returns:
        tuple: (total_files, total_lines, total_words)
    """
    output_path = Path(output_file)
    
    # Create a backup of the original file
    backup_file = str(output_path) + ".backup"
    print(f"Creating backup of original corpus at {backup_file}")
    
    # Read the existing corpus
    print(f"Reading existing corpus from {output_path}...")
    existing_lines = []
    existing_words = 0
    
    try:
        with open(output_path, 'r', encoding='utf-8') as infile:
            chunk_size = 10 * 1024 * 1024  # 10 MB chunks
            while True:
                chunk = infile.readlines(chunk_size)
                if not chunk:
                    break
                existing_lines.extend(chunk)
                print(f"  Read chunk: {len(chunk):,} lines")
        
        # Create backup
        with open(backup_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(existing_lines)
        
        # Count words in existing corpus
        existing_words = sum(len(line.split()) for line in existing_lines)
        print(f"Existing corpus has {len(existing_lines):,} lines and {existing_words:,} words")
    except Exception as e:
        print(f"Error reading existing corpus: {e}")
        return 0, 0, 0
    
    # Read all additional files
    total_files = 0
    total_lines = len(existing_lines)
    total_words = existing_words
    additional_lines = []
    
    start_time = time.time()
    
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: {input_path} does not exist, skipping...")
            continue
        
        print(f"Reading {input_path}...")
        print(f"File size: {input_path.stat().st_size / (1024*1024):.2f} MB")
        
        try:
            # Read the file in chunks to handle very large files
            file_lines = []
            with open(input_path, 'r', encoding='utf-8') as infile:
                chunk_size = 10 * 1024 * 1024  # 10 MB chunks
                while True:
                    chunk = infile.readlines(chunk_size)
                    if not chunk:
                        break
                    file_lines.extend(chunk)
                    print(f"  Read chunk: {len(chunk):,} lines")
            
            # Count words in the file
            file_words = sum(len(line.split()) for line in file_lines)
            
            # Add to the additional lines list
            additional_lines.extend(file_lines)
            
            # Update counters
            total_files += 1
            file_lines_count = len(file_lines)
            total_lines += file_lines_count
            total_words += file_words
            
            print(f"  Added {file_lines_count:,} lines, {file_words:,} words")
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    
    # Combine existing and additional lines
    all_lines = existing_lines + additional_lines
    
    # Shuffle the lines if requested
    if shuffle:
        print(f"Shuffling {len(all_lines):,} lines...")
        random.shuffle(all_lines)
    
    # Write the combined content to the output file
    print(f"Writing {len(all_lines):,} lines to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(all_lines)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nIntegration complete in {elapsed_time:.2f} seconds!")
    print(f"Total additional files processed: {total_files:,}")
    print(f"Total lines in updated corpus: {total_lines:,}")
    print(f"Total words in updated corpus: {total_words:,}")
    
    return total_files, total_lines, total_words

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Input files to add
    input_files = [
        os.path.join("datasets", "kiswahili.txt"),
        os.path.join("datasets", "utafiti.txt"),
        os.path.join("datasets", "parliament_clean.txt"),
        os.path.join("datasets", "mwongozo_clean.txt"),
        os.path.join("datasets", "katiba_clean.txt"),
        os.path.join("datasets", "adero_clean.txt"),
        os.path.join("datasets", "wiki_sw_combined_clean.txt")  # Corrected from wiki_sw_combined.txt based on actual filename
    ]
    
    # Output file (existing corpus)
    output_file = os.path.join("data", "combined_corpus.txt")
    
    # Integrate additional files and get statistics
    total_files, total_lines, total_words = integrate_additional_datasets(input_files, output_file)
    
    print(f"\nFinal word count in {output_file}: {total_words:,}")
