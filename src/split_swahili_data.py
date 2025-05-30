import os
import random

def count_lines(filename):
    """Count lines in file without loading into memory"""
    print("Counting lines in file...")
    count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 100000 == 0:  # Progress update
                print(f"  Counted {count:,} lines...")
    return count

def split_file_efficiently(input_file, output_train, output_valid, train_ratio=0.95):
    """Split file without loading everything into memory"""
    
    # Count total lines first
    total_lines = count_lines(input_file)
    train_size = int(total_lines * train_ratio)
    
    print(f"Total lines: {total_lines:,}")
    print(f"Train set size (95%): {train_size:,} lines")
    print(f"Validation set size (5%): {total_lines - train_size:,} lines")
    
    # Generate random indices for validation set (5%)
    print("Generating random validation indices...")
    random.seed(42)
    valid_indices = set(random.sample(range(total_lines), total_lines - train_size))
    
    print("Splitting file...")
    
    # Open output files
    train_count = 0
    valid_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as input_f, \
         open(output_train, 'w', encoding='utf-8') as train_f, \
         open(output_valid, 'w', encoding='utf-8') as valid_f:
        
        for line_idx, line in enumerate(input_f):
            if line_idx in valid_indices:
                valid_f.write(line)
                valid_count += 1
            else:
                train_f.write(line)
                train_count += 1
    
            # Progress update
            if (line_idx + 1) % 50000 == 0:
                print(f"  Processed {line_idx + 1:,} lines...")
    
    return train_count, valid_count

# Define file paths
input_file = 'swahili_safi_clean.txt'
output_train = os.path.join('data', 'train.txt')
output_valid = os.path.join('data', 'valid.txt')

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found!")
    print("Make sure you've downloaded the dataset first.")
    exit(1)

print(f"Input file: {input_file}")
print(f"Output train file: {output_train}")
print(f"Output valid file: {output_valid}")
print()

try:
    # Split the file efficiently
    train_count, valid_count = split_file_efficiently(input_file, output_train, output_valid)
    
    print(f"\n✅ Data split complete!")
    print(f"📄 Train set: {train_count:,} lines → {output_train}")
    print(f"📄 Validation set: {valid_count:,} lines → {output_valid}")
    
    # Check file sizes
    train_size_mb = os.path.getsize(output_train) / (1024 * 1024)
    valid_size_mb = os.path.getsize(output_valid) / (1024 * 1024)
    
    print(f"💾 Train file size: {train_size_mb:.1f} MB")
    print(f"💾 Valid file size: {valid_size_mb:.1f} MB")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure you have enough disk space and the input file exists.")