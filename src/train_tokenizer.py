import os
import sys
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, Tokenizer

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processor import extract_dataset, get_dataset_stats

def train_tokenizer(
    texts=None,
    vocab_size=50000,
    min_frequency=2,
    save_dir="tokenizer",
    special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
):
    """
    Train a ByteLevelBPE tokenizer on Swahili text data.
    
    Args:
        texts: List of text samples. If None, will load from archive.zip
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
        save_dir: Directory to save the tokenizer
        special_tokens: List of special tokens to add
    """
    if texts is None:
        print("Loading dataset...")
        texts = extract_dataset("archive.zip")
        stats = get_dataset_stats(texts)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    print(f"\nTraining tokenizer with vocab size {vocab_size}...")
    
    # Initialize tokenizer
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    
    # Train tokenizer
    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the tokenizer files
    tokenizer.save(f"{save_dir}/vocab.json")
    tokenizer.save(f"{save_dir}/merges.txt")
    tokenizer.save(f"{save_dir}/tokenizer.json")
    
    print(f"Tokenizer saved to {save_dir}")
    
    # Test the tokenizer
    test_text = "Jambo! Habari yako? Leo ni siku nzuri."
    encoded = tokenizer.encode(test_text)
    
    print("\nTokenizer test:")
    print(f"Original text: {test_text}")
    print(f"Encoded tokens: {' '.join(encoded.tokens)}")
    print(f"Decoded: {tokenizer.decode(encoded.ids)}")
    
    return tokenizer

if __name__ == "__main__":
    train_tokenizer()
