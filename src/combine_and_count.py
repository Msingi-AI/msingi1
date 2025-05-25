import os
import sys
import time
import shutil

def combine_files(file_paths, output_path):
    """Combine multiple text files into one."""
    print(f"Combining files into {output_path}...")
    
    total_size = sum(os.path.getsize(file_path) for file_path in file_paths)
    print(f"Total input size: {total_size / (1024 * 1024):.2f} MB")
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file_path in file_paths:
            print(f"Adding {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as infile:
                # Copy in chunks to handle large files efficiently
                shutil.copyfileobj(infile, outfile)
                # Add a newline between files to ensure separation
                outfile.write('\n')
    
    print(f"Combined file created: {output_path}")
    print(f"Combined file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

def count_words(file_path):
    """Count words in a text file with progress updates."""
    print(f"Counting words in {file_path}...")
    start_time = time.time()
    
    word_count = 0
    line_count = 0
    char_count = 0
    
    chunk_size = 1024 * 1024 * 10  # 10MB chunks
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = f.read(chunk_size)
            chunk_num = 0
            
            while chunk:
                chunk_num += 1
                # Count lines in this chunk
                lines = chunk.count('\n')
                line_count += lines
                
                # Count characters
                char_count += len(chunk)
                
                # Count words
                words = len(chunk.split())
                word_count += words
                
                # Print progress
                elapsed = time.time() - start_time
                print(f"Processed chunk {chunk_num}, {word_count:,} words so far... ({word_count/elapsed:.2f} words/sec)")
                
                # Read next chunk
                chunk = f.read(chunk_size)
        
        elapsed_time = time.time() - start_time
        print(f"\nResults for {os.path.basename(file_path)}:")
        print(f"  File size:       {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
        print(f"  Lines:           {line_count:,}")
        print(f"  Words:           {word_count:,}")
        print(f"  Characters:      {char_count:,}")
        print(f"  Processing time: {elapsed_time:.2f} seconds")
        print(f"  Processing rate: {word_count/elapsed_time:.2f} words/second")
        
        return word_count, line_count, char_count
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return 0, 0, 0

if __name__ == "__main__":
    # Input files
    file1 = "datasets/Swahili Corpus/Swahili_Corpus_combined.txt"
    file2 = "datasets/hcs-na-v2/combined.txt"
    
    # Output combined file
    output_file = "datasets/combined_swahili_corpus.txt"
    
    # Combine files
    combine_files([file1, file2], output_file)
    
    # Count words in combined file
    count_words(output_file)
