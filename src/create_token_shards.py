import numpy as np
import os
import json
import gc
import psutil
from tqdm import tqdm
from pathlib import Path
from transformers import PreTrainedTokenizerFast
import random

# --- Config ---
TOKENIZER_PATH = "tokenizer/swahili_unigram_32000/tokenizer.json"
TRAIN_FILE = "data/train.txt"
VALID_FILE = "data/valid.txt"
OUTPUT_DIR = "msingi_tokens"
SHARD_SIZE = 10_000_000  # tokens per shard (optimized for 13GB+ RAM)
VAL_CHUNK_SIZE = 3_000_000  # Process validation in larger chunks (for 13GB+ RAM)
BUFFER_SIZE = 3_000_000  # Maximum tokens to accumulate before writing
SEED = 42

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

def monitor_memory():
    """Monitor current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def create_token_shards(
    tokenizer_path, 
    train_file, 
    valid_file, 
    output_dir, 
    shard_size=5_000_000,
    val_chunk_size=1_000_000,
    buffer_size=1_000_000
):
    """
    Tokenize and shard the Swahili corpus for efficient training.
    
    Args:
        tokenizer_path: Path to the Unigram tokenizer
        train_file: Path to the training text file
        valid_file: Path to the validation text file
        output_dir: Directory to save the token shards
        shard_size: Number of tokens per shard
        val_chunk_size: Number of tokens to process at once for validation
        buffer_size: Maximum tokens to accumulate in memory before writing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Get or add EOT token ID
    if tokenizer.eos_token_id is None:
        print("EOT token not found in tokenizer. Adding <eot> token...")
        # Add <eot> token to the tokenizer
        tokenizer.add_special_tokens({'eos_token': '<eot>'})
        # Save the updated tokenizer
        tokenizer.save_pretrained(os.path.join(os.path.dirname(tokenizer_path), "updated"))
        print(f"Updated tokenizer saved with <eot> token")
    
    eot_id = tokenizer.eos_token_id
    print(f"EOT token ID: {eot_id}")
    
    if eot_id is None:
        raise ValueError("Failed to add EOT token to tokenizer")
    
    # Process validation set in chunks to save memory
    print(f"\nProcessing validation set: {valid_file}")
    print(f"Memory before validation processing: {monitor_memory():.1f} MB")
    
    val_tokens = []
    val_token_count = 0
    val_chunk_id = 0
    val_chunks = []
    
    with open(valid_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if line:
                # Add EOT token after each document
                tokens = tokenizer.encode(line) + [eot_id]
                val_tokens.extend(tokens)
                
                # If we've accumulated enough tokens, save as a chunk
                if len(val_tokens) >= val_chunk_size:
                    val_chunk_path = os.path.join(output_dir, f"msingi_val_{val_chunk_id:06d}.npy")
                    np.save(val_chunk_path, np.array(val_tokens, dtype=np.uint16))
                    val_chunks.append(val_chunk_path)
                    val_token_count += len(val_tokens)
                    print(f"Saved validation chunk {val_chunk_id} with {len(val_tokens):,} tokens")
                    val_chunk_id += 1
                    val_tokens = []
                    gc.collect()  # Force garbage collection
    
    # Save any remaining validation tokens
    if val_tokens:
        val_chunk_path = os.path.join(output_dir, f"msingi_val_{val_chunk_id:06d}.npy")
        np.save(val_chunk_path, np.array(val_tokens, dtype=np.uint16))
        val_chunks.append(val_chunk_path)
        val_token_count += len(val_tokens)
        print(f"Saved final validation chunk {val_chunk_id} with {len(val_tokens):,} tokens")
    
    print(f"Saved validation set with {val_token_count:,} tokens in {len(val_chunks)} chunks")
    print(f"Memory after validation processing: {monitor_memory():.1f} MB")
    gc.collect()  # Force garbage collection
    
    # Process training set with improved memory management
    print(f"\nProcessing training set: {train_file}")
    print(f"Memory before training processing: {monitor_memory():.1f} MB")
    
    train_tokens = []
    doc_count = 0
    shard_id = len(val_chunks)  # Start after validation chunks
    total_train_tokens = 0
    
    with open(train_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f)):
            line = line.strip()
            if line:
                # Add EOT token after each document
                tokens = tokenizer.encode(line) + [eot_id]
                train_tokens.extend(tokens)
                doc_count += 1
                
                # Create a shard when we reach the shard size or buffer limit
                if len(train_tokens) >= shard_size:
                    shard_path = os.path.join(output_dir, f"msingi_train_{shard_id:06d}.npy")
                    tokens_to_save = train_tokens[:shard_size]
                    np.save(shard_path, np.array(tokens_to_save, dtype=np.uint16))
                    total_train_tokens += len(tokens_to_save)
                    print(f"Saved shard {shard_id} with {len(tokens_to_save):,} tokens to {shard_path}")
                    shard_id += 1
                    
                    # Keep remaining tokens for next shard
                    train_tokens = train_tokens[shard_size:]
                    
                    # Force garbage collection to free memory
                    del tokens_to_save
                    gc.collect()
                    
                # Periodically report memory usage
                if line_num % 100000 == 0:
                    print(f"Memory usage at line {line_num:,}: {monitor_memory():.1f} MB")
    
    # Save final shard with remaining tokens
    if train_tokens:
        shard_path = os.path.join(output_dir, f"msingi_train_{shard_id:06d}.npy")
        np.save(shard_path, np.array(train_tokens, dtype=np.uint16))
        total_train_tokens += len(train_tokens)
        print(f"Saved final shard {shard_id} with {len(train_tokens):,} tokens to {shard_path}")
    
    print(f"Memory after training processing: {monitor_memory():.1f} MB")
    
    # Save metadata
    train_shards = [f for f in os.listdir(output_dir) if f.startswith("msingi_train_")]
    val_shards = [f for f in os.listdir(output_dir) if f.startswith("msingi_val_")]
    
    metadata = {
        "tokenizer": tokenizer_path,
        "vocab_size": tokenizer.vocab_size,
        "train_file": train_file,
        "valid_file": valid_file,
        "train_documents": doc_count,
        "shard_size": shard_size,
        "num_train_shards": len(train_shards),
        "num_val_shards": len(val_shards),
        "val_tokens": val_token_count,
        "train_tokens": total_train_tokens,
        "total_tokens": val_token_count + total_train_tokens,
        "eot_token_id": eot_id,
        "created_at": Path(output_dir).stat().st_mtime
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nSharding complete!")
    print(f"Created {len(train_shards)} training shards + {len(val_shards)} validation shards")
    print(f"Metadata saved to {os.path.join(output_dir, 'metadata.json')}")
    print(f"Final memory usage: {monitor_memory():.1f} MB")
    
    return metadata

if __name__ == "__main__":
    print("Creating token shards for Msingi1 training...")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Valid file: {VALID_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Shard size: {SHARD_SIZE:,} tokens")
    print(f"Validation chunk size: {VAL_CHUNK_SIZE:,} tokens")
    print(f"Buffer size: {BUFFER_SIZE:,} tokens")
    print(f"Initial memory usage: {monitor_memory():.1f} MB")
    
    metadata = create_token_shards(
        tokenizer_path=TOKENIZER_PATH,
        train_file=TRAIN_FILE,
        valid_file=VALID_FILE,
        output_dir=OUTPUT_DIR,
        shard_size=SHARD_SIZE,
        val_chunk_size=VAL_CHUNK_SIZE,
        buffer_size=BUFFER_SIZE
    )
    
    # Print summary
    print("\nSummary:")
    print(f"Total tokens: {metadata['total_tokens']:,}")
    print(f"Validation tokens: {metadata['val_tokens']:,}")
    print(f"Training tokens: {metadata['train_tokens']:,}")
    print(f"Number of training shards: {metadata['num_train_shards']}")
    print(f"Number of validation shards: {metadata['num_val_shards']}")
    if metadata['num_train_shards'] > 0:
        print(f"Average tokens per training shard: ~{metadata['train_tokens'] / metadata['num_train_shards']:,.0f}")
