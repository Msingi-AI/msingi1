def remove_duplicates(input_file, output_file=None):
    """Remove duplicate lines while preserving order."""
    if output_file is None:
        output_file = input_file
        
    # Read all lines and keep track of seen lines
    seen = set()
    unique_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)
    
    # Write back unique lines
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')
    
    print(f"Original lines: {len(seen)}")
    print(f"Unique lines: {len(unique_lines)}")
    print(f"Removed {len(seen) - len(unique_lines)} duplicates")

if __name__ == "__main__":
    synthetic_file = "data/Swahili data/Swahili data/synthetic_swahili.txt"
    remove_duplicates(synthetic_file)
