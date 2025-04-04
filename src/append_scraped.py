import json
import os

def append_scraped_to_train():
    # Load scraped data
    with open('cleaned_swahili.json', 'r', encoding='utf-8') as f:
        scraped_data = json.load(f)
    
    # Extract and clean texts
    texts = []
    for item in scraped_data:
        if 'text' in item:
            # Clean the text - remove multiple spaces and empty lines
            text = ' '.join(item['text'].split())
            if text:
                texts.append(text)
    
    print(f"Found {len(texts)} texts to append")
    
    # Backup original train.txt
    train_file = "data/Swahili data/Swahili data/train.txt"
    backup_file = train_file + ".backup"
    
    if not os.path.exists(backup_file):
        print("Creating backup of original train.txt")
        with open(train_file, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    
    # Append texts to train.txt
    print("Appending scraped texts to train.txt")
    with open(train_file, 'a', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    print("Done! Added scraped texts to train.txt")
    print("Original file backed up as train.txt.backup")

if __name__ == "__main__":
    append_scraped_to_train()
