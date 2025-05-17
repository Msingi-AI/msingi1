def count_words(file_path):
    """Count the number of words in a text file."""
    word_count = 0
    line_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                words = line.split()
                word_count += len(words)
                line_count += 1
    
    return word_count, line_count

if __name__ == "__main__":
    import os
    
    data_dir = os.path.join(os.getcwd(), "data")
    train_file = os.path.join(data_dir, "train.txt")
    valid_file = os.path.join(data_dir, "valid.txt")
    
    print("Counting words in train.txt...")
    train_words, train_lines = count_words(train_file)
    
    print("Counting words in valid.txt...")
    valid_words, valid_lines = count_words(valid_file)
    
    total_words = train_words + valid_words
    total_lines = train_lines + valid_lines
    
    print("\nWord Count Summary:")
    print(f"Train set: {train_words:,} words in {train_lines:,} lines")
    print(f"Validation set: {valid_words:,} words in {valid_lines:,} lines")
    print(f"Total: {total_words:,} words in {total_lines:,} lines")
    
    # Calculate average words per line
    avg_words_per_line = total_words / total_lines if total_lines > 0 else 0
    print(f"Average words per line: {avg_words_per_line:.2f}")
