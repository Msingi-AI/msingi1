from datasets import load_dataset
import re
import random
import os

print("Downloading Swahili-SAFI dataset...")
print("This is a clean dataset (~3.5GB) specifically for language modeling")
print("Will split into train.txt (95%) and valid.txt (5%)")

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download the dataset (with trust_remote_code=True)
ds = load_dataset("flax-community/swahili-safi", trust_remote_code=True)

# Output files in data folder
train_file = "data/train.txt"
valid_file = "data/valid.txt"

print(f"Processing {len(ds['train'])} documents...")
print("Cleaning and splitting data...")

# Collect all valid texts first
valid_texts = []
for example in ds['train']:
    text = example['text']
    
    # Basic filtering and cleaning
    if text and len(text.strip()) > 50:  # Skip very short texts
        # Clean whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        valid_texts.append(text)

total_docs = len(valid_texts)
print(f"Found {total_docs:,} valid documents")

# Shuffle for random split
random.seed(42)  # For reproducibility
random.shuffle(valid_texts)

# Calculate split point (95% train, 5% valid)
split_point = int(total_docs * 0.95)
train_texts = valid_texts[:split_point]
valid_texts = valid_texts[split_point:]

print(f"Train set: {len(train_texts):,} documents ({len(train_texts)/total_docs*100:.1f}%)")
print(f"Valid set: {len(valid_texts):,} documents ({len(valid_texts)/total_docs*100:.1f}%)")

# Write training set
print("Writing train.txt...")
train_chars = 0
with open(train_file, 'w', encoding='utf-8') as f:
    for i, text in enumerate(train_texts):
        f.write(text + '\n\n')
        train_chars += len(text)
        
        if (i + 1) % 10000 == 0:
            print(f"  Written {i+1:,} training documents...")

# Write validation set
print("Writing valid.txt...")
valid_chars = 0
with open(valid_file, 'w', encoding='utf-8') as f:
    for text in valid_texts:
        f.write(text + '\n\n')
        valid_chars += len(text)

print(f"\n✅ Download and split complete!")
print(f"📄 Training file: {train_file}")
print(f"   📊 Documents: {len(train_texts):,}")
print(f"   📝 Characters: {train_chars/1e6:.1f}M")
print(f"   💾 Size: ~{train_chars/1e6:.1f} MB")
print(f"📄 Validation file: {valid_file}")
print(f"   📊 Documents: {len(valid_texts):,}")
print(f"   📝 Characters: {valid_chars/1e6:.1f}M")
print(f"   💾 Size: ~{valid_chars/1e6:.1f} MB")
print(f"\nTotal: {train_chars + valid_chars/1e6:.1f}M characters")
print(f"You can now use these files for training!")