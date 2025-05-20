import os
import random
from pathlib import Path
import time
import argparse
import numpy as np
from collections import Counter
import re

def analyze_text_distribution(lines):
    """
    Analyze the distribution of text in the corpus to help ensure representative splits.
    
    Args:
        lines: List of text lines
    
    Returns:
        dict: Statistics about the corpus
    """
    # Sample a subset of lines for faster analysis if the corpus is very large
    sample_size = min(len(lines), 100000)
    sample_lines = random.sample(lines, sample_size)
    
    # Basic statistics
    line_lengths = [len(line) for line in sample_lines]
    word_counts = [len(line.split()) for line in sample_lines]
    
    # Analyze language features (simplified)
    swahili_markers = ['ni', 'na', 'wa', 'ya', 'kwa', 'katika', 'hii', 'huo', 'huyo']
    marker_counts = {marker: sum(1 for line in sample_lines if f" {marker} " in f" {line} ") for marker in swahili_markers}
    
    # Check for different content types (simplified detection)
    religious_terms = ['mungu', 'yesu', 'biblia', 'allah', 'kurani', 'mtume']
    legal_terms = ['sheria', 'katiba', 'mahakama', 'bunge', 'serikali']
    news_terms = ['habari', 'gazeti', 'ripoti', 'taarifa', 'leo']
    
    content_type_counts = {
        'religious': sum(1 for line in sample_lines if any(term in line.lower() for term in religious_terms)),
        'legal': sum(1 for line in sample_lines if any(term in line.lower() for term in legal_terms)),
        'news': sum(1 for line in sample_lines if any(term in line.lower() for term in news_terms))
    }
    
    return {
        'line_length': {
            'mean': np.mean(line_lengths),
            'median': np.median(line_lengths),
            'min': min(line_lengths),
            'max': max(line_lengths)
        },
        'word_count': {
            'mean': np.mean(word_counts),
            'median': np.median(word_counts),
            'min': min(word_counts),
            'max': max(word_counts)
        },
        'marker_counts': marker_counts,
        'content_type_counts': content_type_counts
    }

def split_corpus(input_file, train_file, valid_file, test_file=None, 
                train_ratio=0.9, valid_ratio=0.1, test_ratio=0.0, 
                shuffle=True, stratify=True):
    """
    Split the corpus into training, validation, and optionally test sets.
    
    Args:
        input_file: Path to the input corpus file
        train_file: Path to output training file
        valid_file: Path to output validation file
        test_file: Path to output test file (optional)
        train_ratio: Proportion for training set (default: 0.9)
        valid_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.0)
        shuffle: Whether to shuffle the lines (default: True)
        stratify: Whether to try to maintain similar distributions (default: True)
    
    Returns:
        tuple: Statistics about the split
    """
    # Validate ratios
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    input_path = Path(input_file)
    train_path = Path(train_file)
    valid_path = Path(valid_file)
    test_path = Path(test_file) if test_file else None
    
    # Ensure output directories exist
    train_path.parent.mkdir(exist_ok=True)
    valid_path.parent.mkdir(exist_ok=True)
    if test_path:
        test_path.parent.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    print(f"Reading corpus from {input_path}...")
    
    # Read all lines from the input file
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    total_lines = len(lines)
    print(f"Read {total_lines:,} lines from corpus")
    
    # Analyze corpus for stratification
    if stratify:
        print("Analyzing corpus distribution...")
        corpus_stats = analyze_text_distribution(lines)
        print(f"Average line length: {corpus_stats['line_length']['mean']:.2f} characters")
        print(f"Average word count: {corpus_stats['word_count']['mean']:.2f} words per line")
        
        # Simple content type detection
        for content_type, count in corpus_stats['content_type_counts'].items():
            percentage = (count / len(lines)) * 100
            print(f"Estimated {content_type} content: {percentage:.2f}%")
    
    # Shuffle the lines if requested
    if shuffle:
        print(f"Shuffling {total_lines:,} lines...")
        random.shuffle(lines)
    
    # Calculate split indices
    train_end = int(total_lines * train_ratio)
    valid_end = train_end + int(total_lines * valid_ratio)
    
    # Split the lines
    train_lines = lines[:train_end]
    valid_lines = lines[train_end:valid_end]
    test_lines = lines[valid_end:] if test_ratio > 0 else []
    
    # Verify stratification if requested
    if stratify:
        print("\nVerifying split distribution...")
        train_stats = analyze_text_distribution(train_lines)
        valid_stats = analyze_text_distribution(valid_lines)
        
        print("Training set:")
        print(f"Average word count: {train_stats['word_count']['mean']:.2f} words per line")
        print("Validation set:")
        print(f"Average word count: {valid_stats['word_count']['mean']:.2f} words per line")
    
    # Write the splits to files
    print(f"Writing {len(train_lines):,} lines to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(train_lines)
    
    print(f"Writing {len(valid_lines):,} lines to {valid_path}...")
    with open(valid_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(valid_lines)
    
    if test_ratio > 0:
        print(f"Writing {len(test_lines):,} lines to {test_path}...")
        with open(test_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(test_lines)
    
    elapsed_time = time.time() - start_time
    
    # Count words in each split
    train_words = sum(len(line.split()) for line in train_lines)
    valid_words = sum(len(line.split()) for line in valid_lines)
    test_words = sum(len(line.split()) for line in test_lines) if test_lines else 0
    
    print(f"\nSplit complete in {elapsed_time:.2f} seconds!")
    print(f"Training set: {len(train_lines):,} lines, {train_words:,} words ({train_ratio*100:.1f}%)")
    print(f"Validation set: {len(valid_lines):,} lines, {valid_words:,} words ({valid_ratio*100:.1f}%)")
    if test_ratio > 0:
        print(f"Test set: {len(test_lines):,} lines, {test_words:,} words ({test_ratio*100:.1f}%)")
    
    return {
        'train_lines': len(train_lines),
        'train_words': train_words,
        'valid_lines': len(valid_lines),
        'valid_words': valid_words,
        'test_lines': len(test_lines) if test_lines else 0,
        'test_words': test_words
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split a corpus into training, validation, and test sets')
    parser.add_argument('--input', default=os.path.join("data", "combined_corpus.txt"), help='Input corpus file')
    parser.add_argument('--train', default=os.path.join("data", "train.txt"), help='Output training file')
    parser.add_argument('--valid', default=os.path.join("data", "valid.txt"), help='Output validation file')
    parser.add_argument('--test', default=os.path.join("data", "test.txt"), help='Output test file')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Proportion for training set')
    parser.add_argument('--valid-ratio', type=float, default=0.1, help='Proportion for validation set')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Proportion for test set')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle the lines')
    parser.add_argument('--no-stratify', action='store_true', help='Do not attempt to stratify the splits')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Split the corpus
    split_corpus(
        input_file=args.input,
        train_file=args.train,
        valid_file=args.valid,
        test_file=args.test,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        shuffle=not args.no_shuffle,
        stratify=not args.no_stratify
    )
