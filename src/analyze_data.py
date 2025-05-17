"""
Analyze the existing Swahili text data in the data folder.
This script provides statistics and information about the available text data.
"""
import os
import re
import sys
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

def count_words(text):
    """Count the number of words in a text."""
    return len(re.findall(r'\S+', text))

def count_chars(text):
    """Count the number of characters in a text."""
    return len(text)

def analyze_file(file_path):
    """Analyze a single text file and return statistics."""
    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # For very large files, use a more efficient approach
        num_chars = 0
        num_words = 0
        num_lines = 0
        word_counter = Counter()
        total_word_length = 0
        total_word_count = 0
        
        # Process the file in chunks to avoid memory issues
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        
        print(f"  Size: {file_size_mb:.2f} MB - ", end="")
        if file_size_mb > 100:
            print(f"Large file, processing in chunks...")
        else:
            print(f"Processing...")
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Count lines, words, and characters in chunks
            for chunk in iter(lambda: f.read(chunk_size), ''):
                num_chars += len(chunk)
                num_lines += chunk.count('\n')
                
                # Count words
                words_in_chunk = re.findall(r'\b\w+\b', chunk.lower())
                num_words += len(words_in_chunk)
                
                # Update word counter (for large files, only sample words)
                if file_size_mb <= 100 or len(word_counter) < 1000:
                    word_counter.update(words_in_chunk[:10000])  # Limit words per chunk for efficiency
                
                # Update average word length calculation
                total_word_length += sum(len(word) for word in words_in_chunk[:10000])
                total_word_count += min(len(words_in_chunk), 10000)
        
        # Calculate average word length
        avg_word_length = total_word_length / total_word_count if total_word_count > 0 else 0
        
        # Get top 20 words
        word_freq = word_counter.most_common(20)
        
        return {
            'path': file_path,
            'size_bytes': file_size,
            'size_mb': file_size_mb,
            'num_chars': num_chars,
            'num_words': num_words,
            'num_lines': num_lines + 1,  # Add 1 for files that don't end with newline
            'avg_word_length': avg_word_length,
            'word_freq': word_freq
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def analyze_data_folder(data_dir):
    """Analyze all text files in the data directory."""
    data_dir = Path(data_dir)
    results = []
    
    # Find all .txt files
    txt_files = list(data_dir.glob('*.txt'))
    
    if not txt_files:
        print(f"No .txt files found in {data_dir}")
        return []
    
    print(f"Found {len(txt_files)} .txt files in {data_dir}")
    
    # Analyze each file
    for file_path in txt_files:
        print(f"Analyzing {file_path}...")
        result = analyze_file(file_path)
        if result:
            results.append(result)
    
    return results

def print_summary(results):
    """Print a summary of the analysis results."""
    if not results:
        print("No results to summarize.")
        return
    
    total_size_mb = sum(r['size_mb'] for r in results)
    total_words = sum(r['num_words'] for r in results)
    total_chars = sum(r['num_chars'] for r in results)
    total_lines = sum(r['num_lines'] for r in results)
    
    print("\n" + "="*80)
    print("SUMMARY OF SWAHILI TEXT DATA")
    print("="*80)
    
    print(f"\nTotal files analyzed: {len(results)}")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Total words: {total_words:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total lines: {total_lines:,}")
    
    print("\nIndividual file statistics:")
    print("-"*80)
    print(f"{'Filename':<30} {'Size (MB)':<10} {'Words':<12} {'Lines':<10}")
    print("-"*80)
    
    for r in sorted(results, key=lambda x: x['num_words'], reverse=True):
        filename = os.path.basename(r['path'])
        print(f"{filename:<30} {r['size_mb']:<10.2f} {r['num_words']:<12,} {r['num_lines']:<10,}")
    
    # Most common words across all files
    all_words = []
    for r in results:
        all_words.extend([word for word, _ in r['word_freq']])
    
    word_counter = Counter(all_words)
    print("\nMost common words across all files:")
    for word, count in word_counter.most_common(20):
        print(f"{word}: {count}")

def main():
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    print(f"Analyzing Swahili text data in: {data_dir}")
    
    # Analyze the data folder
    results = analyze_data_folder(data_dir)
    
    # Print summary
    print_summary(results)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
