import os

def add_synthetic_data(mode='append'):
    """
    Add synthetic data to the training set.
    
    Args:
        mode (str): One of:
            - 'append': Directly append to train.txt
            - 'separate': Create synthetic_swahili.txt
            - 'combine': Create a new combined dataset
    """
    data_dir = "data/Swahili data/Swahili data"
    train_file = os.path.join(data_dir, "train.txt")
    synthetic_file = os.path.join(data_dir, "synthetic_swahili.txt")
    
    # Create backup if not exists
    backup_file = train_file + ".backup"
    if not os.path.exists(backup_file):
        print("Creating backup of original train.txt")
        with open(train_file, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    
    if mode == 'separate':
        # Just create the synthetic file
        print(f"Please paste your synthetic data into: {synthetic_file}")
        if not os.path.exists(synthetic_file):
            open(synthetic_file, 'w', encoding='utf-8').close()
            
    elif mode == 'append':
        # Create a temporary file for pasting
        temp_file = os.path.join(data_dir, "temp_synthetic.txt")
        print(f"Please paste your synthetic data into: {temp_file}")
        print("After pasting, the data will be appended to train.txt")
        open(temp_file, 'w', encoding='utf-8').close()
        
        input("Press Enter after you've pasted the synthetic data...")
        
        # Append to train.txt
        with open(temp_file, 'r', encoding='utf-8') as src:
            synthetic_data = src.read()
            with open(train_file, 'a', encoding='utf-8') as dst:
                dst.write('\n' + synthetic_data)
        
        # Remove temp file
        os.remove(temp_file)
        print("Synthetic data appended to train.txt")
        
    elif mode == 'combine':
        # Create synthetic file first
        if not os.path.exists(synthetic_file):
            print(f"Please paste your synthetic data into: {synthetic_file}")
            open(synthetic_file, 'w', encoding='utf-8').close()
            input("Press Enter after you've pasted the synthetic data...")
        
        # Create combined dataset
        combined_file = os.path.join(data_dir, "combined_train.txt")
        with open(combined_file, 'w', encoding='utf-8') as dst:
            # Copy original training data
            with open(train_file, 'r', encoding='utf-8') as src:
                dst.write(src.read())
            
            # Add synthetic data
            with open(synthetic_file, 'r', encoding='utf-8') as src:
                dst.write('\n' + src.read())
        
        print(f"Created combined dataset at: {combined_file}")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Append directly to train.txt")
    print("2. Create separate synthetic_swahili.txt")
    print("3. Create new combined dataset")
    
    choice = input("Enter choice (1-3): ")
    mode_map = {'1': 'append', '2': 'separate', '3': 'combine'}
    
    if choice in mode_map:
        add_synthetic_data(mode_map[choice])
    else:
        print("Invalid choice")
