import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def sample_text_for_training(input_file, output_file, sample_size=500000):
    """Sample lines from a large text file for tokenizer training."""
    random.seed(42)
    
    # Count total lines
    print(f"Counting lines in {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total lines: {total_lines:,}")
    
    # Determine sampling rate
    sample_size = min(sample_size, total_lines)
    sampling_rate = sample_size / total_lines
    
    print(f"Sampling {sample_size:,} lines (rate: {sampling_rate:.4f})...")
    
    # Sample lines
    sampled_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if random.random() < sampling_rate and line.strip():
                sampled_lines.append(line)
                if len(sampled_lines) >= sample_size:
                    break
    
    # Write sampled lines
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
    
    print(f"Wrote {len(sampled_lines):,} lines to {output_file}")
    return output_file

def train_byte_level_bpe_tokenizer(
    input_files, 
    vocab_size=32000, 
    min_frequency=2, 
    save_dir="tokenizer",
    special_tokens=None
):
    """Train a ByteLevelBPE tokenizer optimized for Swahili."""
    if special_tokens is None:
        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<sw>"]
    
    print(f"Training ByteLevelBPE tokenizer...")
    print(f"Input files: {input_files}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Minimum frequency: {min_frequency}")
    print(f"Special tokens: {special_tokens}")
    
    # Initialize tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Train
    start_time = time.time()
    tokenizer.train(
        files=input_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    elapsed_time = time.time() - start_time
    
    # Save the tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_model(save_dir)
    tokenizer.save(f"{save_dir}/tokenizer.json")
    
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Tokenizer saved to {save_dir}")
    
    return tokenizer

def train_wordpiece_tokenizer(
    input_files, 
    vocab_size=32000, 
    min_frequency=2, 
    save_dir="tokenizer",
    special_tokens=None
):
    """Train a WordPiece tokenizer optimized for Swahili."""
    if special_tokens is None:
        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<sw>"]
    
    print(f"Training WordPiece tokenizer...")
    print(f"Input files: {input_files}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Minimum frequency: {min_frequency}")
    print(f"Special tokens: {special_tokens}")
    
    # Initialize tokenizer with WordPiece model
    tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
    
    # Add normalizers
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    
    # Add pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Punctuation()
    ])
    
    # Add decoder
    tokenizer.decoder = decoders.WordPiece()
    
    # Initialize trainer
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Train
    start_time = time.time()
    tokenizer.train(input_files, trainer)
    elapsed_time = time.time() - start_time
    
    # Save the tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(f"{save_dir}/tokenizer.json")
    
    # Save vocabulary separately
    with open(f"{save_dir}/vocab.json", 'w', encoding='utf-8') as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)
    
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Tokenizer saved to {save_dir}")
    
    return tokenizer

def train_unigram_tokenizer(
    input_files, 
    vocab_size=32000, 
    save_dir="tokenizer",
    special_tokens=None
):
    """Train a Unigram tokenizer optimized for Swahili."""
    if special_tokens is None:
        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<sw>"]
    
    print(f"Training Unigram tokenizer...")
    print(f"Input files: {input_files}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    
    # Initialize tokenizer with Unigram model
    tokenizer = Tokenizer(models.Unigram())
    
    # Add normalizers
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Replace(Regex=r"\s+", Replacement=" ")
    ])
    
    # Add pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Whitespace()
    ])
    
    # Add post-processor
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>") if "<s>" in tokenizer.get_vocab() else 0),
            ("</s>", tokenizer.token_to_id("</s>") if "</s>" in tokenizer.get_vocab() else 1),
        ],
    )
    
    # Initialize trainer
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Train
    start_time = time.time()
    tokenizer.train(input_files, trainer)
    elapsed_time = time.time() - start_time
    
    # Save the tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(f"{save_dir}/tokenizer.json")
    
    # Save vocabulary separately
    with open(f"{save_dir}/vocab.json", 'w', encoding='utf-8') as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)
    
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Tokenizer saved to {save_dir}")
    
    return tokenizer

def convert_to_transformers(tokenizer_path, save_dir):
    """Convert tokenizer to Transformers format."""
    print("Converting to Transformers format...")
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<sw>"]
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"Transformers tokenizer saved to {save_dir}")

def test_tokenizer(tokenizer_path):
    """Test the trained tokenizer on sample sentences."""
    test_sentences = [
        "Jambo! Habari yako?",
        "Nina umri wa miaka 25 na ninaishi Nairobi.",
        "Elimu ni muhimu sana kwa maendeleo ya jamii.",
        "Kompyuta yangu mpya ina programu za kisasa.",
        "Haba na haba hujaza kibaba.",
        "Ninapenda kusoma vitabu vya Kiswahili na kusikiliza muziki.",
        "Serikali ya Kenya imeahidi kuboresha miundombinu ya nchi.",
        "Wanafunzi wanasoma kwa bidii ili kufaulu mtihani wao."
    ]
    
    # Load the tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print("\nTokenizer tests:")
    for test in test_sentences:
        # Encode
        encoded = tokenizer.encode(test)
        
        # Print results
        print(f"\nOriginal: {test}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Decoded: {tokenizer.decode(encoded.ids)}")
        print(f"Number of tokens: {len(encoded.tokens)}")

def main():
    parser = argparse.ArgumentParser(description="Train a tokenizer for Swahili using Hugging Face Tokenizers")
    parser.add_argument("--input", default="data/train.txt", help="Input text file")
    parser.add_argument("--sample", action="store_true", help="Sample input for faster training")
    parser.add_argument("--sample-size", type=int, default=500000, help="Number of lines to sample")
    parser.add_argument("--model-type", choices=["bpe", "wordpiece", "unigram"], default="bpe", 
                        help="Tokenizer model type")
    parser.add_argument("--model-dir", default="tokenizer", help="Directory to save model")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency")
    
    args = parser.parse_args()
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Sample input if requested
    input_files = [args.input]
    if args.sample:
        sample_file = os.path.join(args.model_dir, "sampled_input.txt")
        input_files = [sample_text_for_training(args.input, sample_file, args.sample_size)]
    
    # Set model directory
    model_dir = os.path.join(args.model_dir, f"swahili_{args.model_type}_{args.vocab_size}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Train tokenizer based on model type
    if args.model_type == "bpe":
        tokenizer = train_byte_level_bpe_tokenizer(
            input_files=input_files,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            save_dir=model_dir
        )
    elif args.model_type == "wordpiece":
        tokenizer = train_wordpiece_tokenizer(
            input_files=input_files,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            save_dir=model_dir
        )
    elif args.model_type == "unigram":
        tokenizer = train_unigram_tokenizer(
            input_files=input_files,
            vocab_size=args.vocab_size,
            save_dir=model_dir
        )
    
    # Convert to Transformers format
    convert_to_transformers(
        tokenizer_path=f"{model_dir}/tokenizer.json",
        save_dir=os.path.join(model_dir, "transformers")
    )
    
    # Test tokenizer
    test_tokenizer(tokenizer_path=f"{model_dir}/tokenizer.json")

if __name__ == "__main__":
    main()
