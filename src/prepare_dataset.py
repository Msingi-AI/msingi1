import os
import random
from typing import List, Tuple

def read_category_file(filepath: str) -> List[str]:
    """Read and return non-empty lines from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_category_sample(lines: List[str], sample_size: int) -> List[str]:
    """Get a stratified sample from a category, preserving order within chunks."""
    if len(lines) <= sample_size:
        return lines
    
    # Calculate chunk size to maintain some locality in the text
    chunk_size = 10
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    # Calculate how many chunks we need
    chunks_needed = (sample_size + chunk_size - 1) // chunk_size
    selected_chunks = random.sample(chunks, chunks_needed)
    
    # Flatten and trim to exact size
    selected_lines = [line for chunk in selected_chunks for line in chunk]
    return selected_lines[:sample_size]

def prepare_dataset(corpus_dir: str, output_dir: str, train_ratio=0.8, test_ratio=0.1):
    """Prepare the dataset by combining and splitting files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all category files
    category_files = [f for f in os.listdir(corpus_dir) if f.endswith('_Cleaned.txt')]
    
    # Calculate total lines and words for progress tracking
    total_lines = []
    category_sizes = {}
    
    print("Reading files and calculating sizes...")
    for file in category_files:
        filepath = os.path.join(corpus_dir, file)
        lines = read_category_file(filepath)
        category = file.replace('_Cleaned.txt', '')
        category_sizes[category] = len(lines)
        total_lines.extend((line, category) for line in lines)
    
    # Shuffle while keeping track of categories
    print(f"Shuffling {len(total_lines):,} lines...")
    random.shuffle(total_lines)
    
    # Calculate split sizes
    total_size = len(total_lines)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    valid_size = total_size - train_size - test_size
    
    # Split the data
    train_data = total_lines[:train_size]
    test_data = total_lines[train_size:train_size + test_size]
    valid_data = total_lines[train_size + test_size:]
    
    # Write splits to files
    splits = [
        ('train.txt', train_data),
        ('test.txt', test_data),
        ('valid.txt', valid_data)
    ]
    
    category_counts = {split: {} for split, _ in splits}
    
    print("\nWriting splits...")
    for filename, data in splits:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for line, category in data:
                f.write(line + '\n')
                category_counts[filename][category] = category_counts[filename].get(category, 0) + 1
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"{'='*60}")
    print(f"Total lines: {total_size:,}")
    
    for filename, data in splits:
        print(f"\n{filename.upper()}:")
        print(f"Total lines: {len(data):,}")
        print("Category distribution:")
        for category, count in sorted(category_counts[filename].items()):
            percentage = (count / len(data)) * 100
            print(f"- {category:<12}: {count:,} lines ({percentage:.1f}%)")
    
    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"Files saved in: {output_dir}")

if __name__ == "__main__":
    corpus_dir = "Swahili Corpus"
    output_dir = "data/Swahili data/Swahili data"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    prepare_dataset(corpus_dir, output_dir)
