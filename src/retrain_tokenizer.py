import os
import sys
import time
import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def retrain_tokenizer(
    input_files,
    vocab_size=32000,
    min_frequency=3,
    save_dir="tokenizer/swahili_bpe_32000",
    special_tokens=None
):
    """Retrain the ByteLevelBPE tokenizer on the expanded Swahili corpus."""
    if special_tokens is None:
        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<sw>"]
    
    print(f"Retraining ByteLevelBPE tokenizer...")
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
    
    # Convert to Transformers format
    transformers_dir = os.path.join(save_dir, "transformers")
    os.makedirs(transformers_dir, exist_ok=True)
    
    # Create transformers tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{save_dir}/tokenizer.json",
        bos_token="<s>",
        eos_token="</s>",
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
        "Ninapenda kusoma vitabu vya Kiswahili na kusikiliza muziki."
    ]
    
    print("\nTesting tokenizer on sample sentences:")
    for test in test_sentences:
        # Encode
        encoded = tokenizer.encode(test)
        
        # Print results
        print(f"\nOriginal: {test}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Number of tokens: {len(encoded.tokens)}")

def check_files_exist(files):
    """Check if all files in the list exist."""
    for file in files:
        if not os.path.exists(file):
            print(f"Error: File {file} does not exist.")
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain the ByteLevelBPE tokenizer on the expanded Swahili corpus")
    parser.add_argument("--train-file", default="data/train.txt", help="Training data file")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=3, help="Minimum token frequency")
    parser.add_argument("--save-dir", default="tokenizer/swahili_bpe_32000", help="Directory to save tokenizer")
    
    args = parser.parse_args()
    
    # Check if training file exists
    if not check_files_exist([args.train_file]):
        sys.exit(1)
    
    # Retrain tokenizer
    retrain_tokenizer(
        input_files=[args.train_file],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_dir=args.save_dir
    )
