import numpy as np
import os
import json
from tqdm import tqdm
from pathlib import Path
from transformers import PreTrainedTokenizerFast
import random

# --- Config ---
TOKENIZER_PATH = "tokenizer/swahili_unigram_32000/tokenizer.json"
TRAIN_FILE = "data/train.txt"
VALID_FILE = "data/valid.txt"
OUTPUT_DIR = "msingi_tokens"
SHARD_SIZE = 10_000_000  # tokens per shard
SEED = 42

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

def create_token_shards(
    tokenizer_path, 
    train_file, 
    valid_file, 
    output_dir, 
    shard_size=10_000_000
):
    """
    Tokenize and shard the Swahili corpus for efficient training.
    
    Args:
        tokenizer_path: Path to the Unigram tokenizer
        train_file: Path to the training text file
        valid_file: Path to the validation text file
        output_dir: Directory to save the token shards
        shard_size: Number of tokens per shard
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Get EOT token ID
    eot_id = tokenizer.eos_token_id
    print(f"EOT token ID: {eot_id}")
    
    # Process validation set
    print(f"\nProcessing validation set: {valid_file}")
    val_tokens = []
    with open(valid_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if line:
                # Add EOT token after each document
                tokens = tokenizer.encode(line) + [eot_id]
                val_tokens.extend(tokens)
    
    val_tokens = np.array(val_tokens, dtype=np.uint16)
    val_path = os.path.join(output_dir, "msingi_val_000000.npy")
    np.save(val_path, val_tokens)
    print(f"Saved validation set with {len(val_tokens):,} tokens to {val_path}")
    
    # Process training set
    print(f"\nProcessing training set: {train_file}")
    train_tokens = []
    doc_count = 0
    
    with open(train_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if line:
                # Add EOT token after each document
                tokens = tokenizer.encode(line) + [eot_id]
                train_tokens.extend(tokens)
                doc_count += 1
                
                # Create a shard when we reach the shard size
                if len(train_tokens) >= shard_size:
                    shard_id = len(os.listdir(output_dir)) - 1  # Account for val shard
                    shard_path = os.path.join(output_dir, f"msingi_train_{shard_id:06d}.npy")
                    np.save(shard_path, np.array(train_tokens[:shard_size], dtype=np.uint16))
                    print(f"Saved shard {shard_id} with {shard_size:,} tokens to {shard_path}")
                    
                    # Keep remaining tokens for next shard
                    train_tokens = train_tokens[shard_size:]
    
    # Save final shard with remaining tokens
    if train_tokens:
        shard_id = len(os.listdir(output_dir)) - 1  # Account for val shard
        shard_path = os.path.join(output_dir, f"msingi_train_{shard_id:06d}.npy")
        np.save(shard_path, np.array(train_tokens, dtype=np.uint16))
        print(f"Saved final shard {shard_id} with {len(train_tokens):,} tokens to {shard_path}")
    
    # Save metadata
    metadata = {
        "tokenizer": tokenizer_path,
        "vocab_size": tokenizer.vocab_size,
        "train_file": train_file,
        "valid_file": valid_file,
        "train_documents": doc_count,
        "shard_size": shard_size,
        "num_shards": len(os.listdir(output_dir)) - 1,  # Exclude val shard
        "val_tokens": len(val_tokens),
        "total_tokens_estimate": (len(os.listdir(output_dir)) - 1) * shard_size + len(train_tokens),
        "eot_token_id": eot_id,
        "created_at": Path(output_dir).stat().st_mtime
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nSharding complete!")
    print(f"Created {len(os.listdir(output_dir)) - 1} training shards + 1 validation shard")
    print(f"Metadata saved to {os.path.join(output_dir, 'metadata.json')}")
    
    return metadata

if __name__ == "__main__":
    print("Creating token shards for Msingi1 training...")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Valid file: {VALID_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Shard size: {SHARD_SIZE:,} tokens")
    
    metadata = create_token_shards(
        tokenizer_path=TOKENIZER_PATH,
        train_file=TRAIN_FILE,
        valid_file=VALID_FILE,
        output_dir=OUTPUT_DIR,
        shard_size=SHARD_SIZE
    )
    
    # Print summary
    print("\nSummary:")
    print(f"Total tokens: ~{metadata['total_tokens_estimate']:,}")
    print(f"Validation tokens: {metadata['val_tokens']:,}")
    print(f"Training tokens: ~{metadata['total_tokens_estimate'] - metadata['val_tokens']:,}")
    print(f"Number of shards: {metadata['num_shards']}")
    print(f"Average tokens per shard: ~{(metadata['total_tokens_estimate'] - metadata['val_tokens']) / metadata['num_shards']:,.0f}")
