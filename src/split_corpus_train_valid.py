import os
import random
from pathlib import Path
import time
import argparse
import numpy as np

def split_corpus(input_file, train_file, valid_file, train_ratio=0.9, valid_ratio=0.1, shuffle=True):
    """
    Split the corpus into training and validation sets.
    
    Args:
        input_file: Path to the input corpus file
        train_file: Path to output training file
        valid_file: Path to output validation file
        train_ratio: Proportion for training set (default: 0.9)
        valid_ratio: Proportion for validation set (default: 0.1)
        shuffle: Whether to shuffle the lines (default: True)
    
    Returns:
        tuple: Statistics about the split
    """
    # Validate ratios
    if abs(train_ratio + valid_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    input_path = Path(input_file)
    train_path = Path(train_file)
    valid_path = Path(valid_file)
    
    # Ensure output directories exist
    train_path.parent.mkdir(exist_ok=True)
    valid_path.parent.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    print(f"Reading corpus from {input_path}...")
    
    # Read all lines from the input file
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    total_lines = len(lines)
    print(f"Read {total_lines:,} lines from corpus")
    
    # Shuffle the lines if requested
    if shuffle:
        print(f"Shuffling {total_lines:,} lines...")
        random.shuffle(lines)
    
    # Calculate split indices
    train_end = int(total_lines * train_ratio)
    
    # Split the lines
    train_lines = lines[:train_end]
    valid_lines = lines[train_end:]
    
    # Write the splits to files
    print(f"Writing {len(train_lines):,} lines to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(train_lines)
    
    print(f"Writing {len(valid_lines):,} lines to {valid_path}...")
    with open(valid_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(valid_lines)
    
    elapsed_time = time.time() - start_time
    
    # Count words in each split
    train_words = sum(len(line.split()) for line in train_lines)
    valid_words = sum(len(line.split()) for line in valid_lines)
    
    print(f"\nSplit complete in {elapsed_time:.2f} seconds!")
    print(f"Training set: {len(train_lines):,} lines, {train_words:,} words ({train_ratio*100:.1f}%)")
    print(f"Validation set: {len(valid_lines):,} lines, {valid_words:,} words ({valid_ratio*100:.1f}%)")
    
    return {
        'train_lines': len(train_lines),
        'train_words': train_words,
        'valid_lines': len(valid_lines),
        'valid_words': valid_words
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split a corpus into training and validation sets')
    parser.add_argument('--input', default=os.path.join("data", "combined_corpus.txt"), help='Input corpus file')
    parser.add_argument('--train', default=os.path.join("data", "train.txt"), help='Output training file')
    parser.add_argument('--valid', default=os.path.join("data", "valid.txt"), help='Output validation file')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Proportion for training set')
    parser.add_argument('--valid-ratio', type=float, default=0.1, help='Proportion for validation set')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle the lines')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.valid_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Split the corpus
    split_corpus(
        input_file=args.input,
        train_file=args.train,
        valid_file=args.valid,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        shuffle=not args.no_shuffle
    )
