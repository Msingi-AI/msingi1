"""
Process the downloaded mC4 Swahili corpus for training Msingi models.

This script:
1. Tokenizes the raw text corpus
2. Adds <eot> tokens between documents
3. Creates memory-mapped numpy token shards
4. Generates metadata for training
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

# Set console to UTF-8 mode for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Process mC4 Swahili corpus for training")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the raw text corpus file")
    parser.add_argument("--output-dir", type=str, default="msingi_tokens",
                        help="Directory to save token shards")
    parser.add_argument("--tokenizer-path", type=str, 
                        default="tokenizer/swahili_unigram_32000/tokenizer.json",
                        help="Path to the tokenizer file")
    parser.add_argument("--shard-size", type=int, default=10_000_000,
                        help="Target number of tokens per shard")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation"],
                        help="Dataset split being processed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    
    # Add special tokens if not already present
    special_tokens = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '<mask>'
    }
    
    # Check and add special tokens if needed
    for token_type, token in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            print(f"Adding {token_type}: {token}")
    
    tokenizer.add_special_tokens({
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '<mask>'
    })
    
    # Add <eot> token if not present
    if '<eot>' not in tokenizer.get_vocab():
        print("Adding <eot> token to tokenizer")
        tokenizer.add_special_tokens({'additional_special_tokens': ['<eot>']})
    
    eot_token_id = tokenizer.convert_tokens_to_ids('<eot>')
    print(f"<eot> token ID: {eot_token_id}")
    
    # Process the corpus
    print(f"Processing {args.input_file}...")
    
    # Read documents from the corpus file
    documents = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        current_doc = []
        for line in f:
            line = line.strip()
            if not line and current_doc:
                # Empty line marks document boundary
                documents.append('\n'.join(current_doc))
                current_doc = []
            elif line:
                current_doc.append(line)
        
        # Add the last document if not empty
        if current_doc:
            documents.append('\n'.join(current_doc))
    
    print(f"Found {len(documents)} documents in the corpus")
    
    # Tokenize documents and create shards
    all_tokens = []
    token_count = 0
    doc_token_counts = []
    
    print("Tokenizing documents...")
    for doc in tqdm(documents):
        # Tokenize document
        doc_tokens = tokenizer.encode(doc)
        doc_token_counts.append(len(doc_tokens))
        
        # Add to all tokens
        all_tokens.extend(doc_tokens)
        
        # Add <eot> token after each document
        all_tokens.append(eot_token_id)
        
        token_count += len(doc_tokens) + 1  # +1 for <eot>
    
    print(f"Total tokens: {token_count}")
    
    # Create shards
    num_shards = (token_count + args.shard_size - 1) // args.shard_size  # Ceiling division
    print(f"Creating {num_shards} shards of approximately {args.shard_size} tokens each")
    
    shard_metadata = []
    
    for shard_idx in tqdm(range(num_shards), desc="Creating shards"):
        start_idx = shard_idx * args.shard_size
        end_idx = min((shard_idx + 1) * args.shard_size, token_count)
        
        # Get tokens for this shard
        shard_tokens = all_tokens[start_idx:end_idx]
        
        # Convert to numpy array
        shard_array = np.array(shard_tokens, dtype=np.int32)
        
        # Save as memory-mapped numpy file
        shard_filename = os.path.join(args.output_dir, f"{args.split}_shard_{shard_idx:03d}.npy")
        np.save(shard_filename, shard_array)
        
        # Add to metadata
        shard_metadata.append({
            "filename": os.path.basename(shard_filename),
            "num_tokens": len(shard_tokens),
            "start_idx": start_idx,
            "end_idx": end_idx
        })
    
    # Create metadata file
    metadata = {
        "dataset": f"mC4 Swahili ({args.split})",
        "num_documents": len(documents),
        "num_tokens": token_count,
        "num_shards": num_shards,
        "shard_size": args.shard_size,
        "avg_doc_length": sum(doc_token_counts) / len(doc_token_counts) if doc_token_counts else 0,
        "max_doc_length": max(doc_token_counts) if doc_token_counts else 0,
        "min_doc_length": min(doc_token_counts) if doc_token_counts else 0,
        "tokenizer_path": args.tokenizer_path,
        "eot_token_id": eot_token_id,
        "shards": shard_metadata
    }
    
    metadata_file = os.path.join(args.output_dir, f"{args.split}_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processing complete! Metadata saved to {metadata_file}")
    print(f"Token shards saved to {args.output_dir}")
    print(f"You can now use these shards for training with train_with_shards.py")

if __name__ == "__main__":
    main()
