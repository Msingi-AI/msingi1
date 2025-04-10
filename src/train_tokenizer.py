import os
import sys
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from typing import List, Dict, Union

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def train_tokenizer(
    vocab_size: int = 50000,
    min_frequency: int = 2,
    save_dir: str = "tokenizer"
) -> ByteLevelBPETokenizer:
    """Train a ByteLevelBPE tokenizer on Swahili text data."""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize a new tokenizer
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    
    # Get paths to training files
    train_path = "data/Swahili data/Swahili data/train.txt"
    valid_path = "data/Swahili data/Swahili data/valid.txt"
    
    print("Training tokenizer...")
    # Train the tokenizer
    tokenizer.train(
        files=[train_path, valid_path],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
    )
    
    # Save the tokenizer
    tokenizer.save_model(save_dir)
    tokenizer.save(f"{save_dir}/tokenizer.json")
    print(f"Tokenizer saved to {save_dir}")
    
    # Print some statistics
    vocab = tokenizer.get_vocab()
    print(f"\nVocabulary Statistics:")
    print(f"Total vocab size: {len(vocab)}")
    print(f"Special tokens: ['<s>', '</s>', '<unk>', '<pad>']")
    
    # Run some tests
    test_texts = [
        "Jambo! Habari yako?",
        "Nina umri wa miaka 25 na ninaishi Nairobi.",
        "Elimu ni muhimu sana kwa maendeleo ya jamii.",
        "Kompyuta yangu mpya ina programu za kisasa.",
        "Haba na haba hujaza kibaba."
    ]
    
    print("\nTokenizer tests:")
    for test in test_texts:
        encoded = tokenizer.encode(test)
        print(f"\nOriginal: {test}")
        try:
            print(f"Tokens: {' '.join(encoded.tokens)}")
        except UnicodeEncodeError:
            print("Tokens: <tokens contain special characters>")
        print(f"Decoded: {tokenizer.decode(encoded.ids)}")
        print(f"Number of tokens: {len(encoded.tokens)}")
    
    return tokenizer

if __name__ == "__main__":
    train_tokenizer()
