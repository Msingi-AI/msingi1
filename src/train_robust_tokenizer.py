import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
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

def train_tokenizer(input_files, vocab_size=32000, min_frequency=2, save_dir="tokenizer"):
    """Train a ByteLevelBPE tokenizer optimized for Swahili."""
    # Define special tokens
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
        special_tokens=special_tokens
    )
    elapsed_time = time.time() - start_time
    
    # Save the tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_model(save_dir)
    tokenizer.save(f"{save_dir}/tokenizer.json")
    
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Tokenizer saved to {save_dir}")
    
    return tokenizer

def convert_to_transformers(model_dir):
    """Convert tokenizer to Transformers format."""
    print("Converting to Transformers format...")
    
    # Create directory
    transformers_dir = os.path.join(model_dir, "transformers")
    os.makedirs(transformers_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(model_dir, "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<sw>"]
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(transformers_dir)
    print(f"Transformers tokenizer saved to {transformers_dir}")

def test_tokenizer_with_transformers(model_dir):
    """Test the trained tokenizer using Transformers."""
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
    transformers_dir = os.path.join(model_dir, "transformers")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(transformers_dir)
    
    print("\nTokenizer tests (using Transformers):")
    for test in test_sentences:
        # Encode
        encoded = tokenizer.encode(test)
        tokens = tokenizer.tokenize(test)
        
        # Print results
        print(f"\nOriginal: {test}")
        print(f"Tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Decoded: {tokenizer.decode(encoded)}")

def analyze_vocab(model_dir):
    """Analyze the tokenizer vocabulary."""
    vocab_file = os.path.join(model_dir, "vocab.json")
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    print("\nVocabulary Analysis:")
    print(f"Total vocabulary size: {len(vocab)}")
    
    # Count token types
    word_tokens = 0
    subword_tokens = 0
    special_tokens = 0
    
    for token in vocab.keys():
        if token.startswith('Ġ'):  # Word-initial piece in BPE
            word_tokens += 1
        elif token.startswith('<') and token.endswith('>'):  # Special token
            special_tokens += 1
        else:
            subword_tokens += 1
    
    print(f"Word-initial tokens: {word_tokens} ({word_tokens/len(vocab)*100:.1f}%)")
    print(f"Subword tokens: {subword_tokens} ({subword_tokens/len(vocab)*100:.1f}%)")
    print(f"Special tokens: {special_tokens} ({special_tokens/len(vocab)*100:.1f}%)")
    
    # Print some example tokens
    print("\nExample tokens:")
    word_examples = [token for token in vocab.keys() if token.startswith('Ġ')][:10]
    subword_examples = [token for token in vocab.keys() if not token.startswith('Ġ') and not (token.startswith('<') and token.endswith('>'))][:10]
    
    print(f"Word-initial examples: {word_examples}")
    print(f"Subword examples: {subword_examples}")

def main():
    parser = argparse.ArgumentParser(description="Train a ByteLevelBPE tokenizer for Swahili")
    parser.add_argument("--input", default="data/train.txt", help="Input text file")
    parser.add_argument("--sample", action="store_true", help="Sample input for faster training")
    parser.add_argument("--sample-size", type=int, default=500000, help="Number of lines to sample")
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
    model_dir = os.path.join(args.model_dir, f"swahili_bpe_{args.vocab_size}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Train tokenizer
    train_tokenizer(
        input_files=input_files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_dir=model_dir
    )
    
    # Convert to Transformers format
    convert_to_transformers(model_dir)
    
    # Test tokenizer
    test_tokenizer_with_transformers(model_dir)
    
    # Analyze vocabulary
    analyze_vocab(model_dir)

if __name__ == "__main__":
    main()
