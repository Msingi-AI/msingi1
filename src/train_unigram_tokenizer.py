import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from tokenizers.models import Unigram
from transformers import PreTrainedTokenizerFast

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def train_unigram_tokenizer(
    input_files: List[str], 
    vocab_size: int = 32000, 
    save_dir: str = "tokenizer/swahili_unigram_32000",
    special_tokens: Optional[List[str]] = None
):
    """Train a Unigram tokenizer optimized for Swahili."""
    if special_tokens is None:
        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<sw>", "<eot>"]
    
    print(f"Training Unigram tokenizer...")
    print(f"Input files: {input_files}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    
    # Initialize tokenizer with Unigram model
    tokenizer = Tokenizer(Unigram())
    
    # Add pre-tokenizer and normalizer
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    
    # Use ByteLevel pre-tokenizer for better handling of character boundaries
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Configure the trainer
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unk_token="<unk>",
        show_progress=True
    )
    
    # Train
    start_time = time.time()
    tokenizer.train(files=input_files, trainer=trainer)
    elapsed_time = time.time() - start_time
    
    # Add post-processor for handling special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Add decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Save the tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(f"{save_dir}/tokenizer.json")
    
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Tokenizer saved to {save_dir}")
    
    # Convert to Transformers format
    transformers_dir = os.path.join(save_dir, "transformers")
    os.makedirs(transformers_dir, exist_ok=True)
    
    # Create transformers tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{save_dir}/tokenizer.json",
        bos_token="<s>",
        eos_token="<eot>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<sw>"]
    )
    
    # Save transformers tokenizer
    hf_tokenizer.save_pretrained(transformers_dir)
    print(f"Transformers tokenizer saved to {transformers_dir}")
    
    # Test the tokenizer
    test_tokenizer(tokenizer)
    
    return tokenizer

def test_tokenizer(tokenizer):
    """Test the trained tokenizer on sample Swahili sentences."""
    test_sentences = [
        "Jambo! Habari yako?",
        "Nina umri wa miaka 25 na ninaishi Nairobi.",
        "Elimu ni muhimu sana kwa maendeleo ya jamii.",
        "Kompyuta yangu mpya ina programu za kisasa.",
        "Haba na haba hujaza kibaba.",
        "Ninapenda kusoma vitabu vya Kiswahili na kusikiliza muziki.",
        "Mtu ni watu.<eot>Mchana mwema."
    ]
    
    print("\nTesting tokenizer on sample sentences:")
    for test in test_sentences:
        # Encode
        encoded = tokenizer.encode(test)
        
        # Print results
        print(f"\nOriginal: {test}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Number of tokens: {len(encoded.tokens)}")
        
        # Check if <eot> is properly tokenized (for the last test case)
        if "<eot>" in test:
            eot_positions = [i for i, token in enumerate(encoded.tokens) if token == "<eot>"]
            if eot_positions:
                print(f"<eot> token found at positions: {eot_positions}")
            else:
                print("<eot> token not found in encoded sequence!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Unigram tokenizer for Swahili")
    parser.add_argument("--train-file", default="data/train.txt", help="Training data file")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--save-dir", default="tokenizer/swahili_unigram_32000", help="Directory to save tokenizer")
    
    args = parser.parse_args()
    
    # Check if training file exists
    if not os.path.exists(args.train_file):
        print(f"Error: Training file {args.train_file} does not exist.")
        sys.exit(1)
    
    # Train tokenizer
    train_unigram_tokenizer(
        input_files=[args.train_file],
        vocab_size=args.vocab_size,
        save_dir=args.save_dir
    )
