import os
import sys
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processor import extract_dataset, get_dataset_stats

def train_tokenizer(
    texts=None,
    vocab_size=32000,  # Reduced from 50k to 32k for better efficiency
    min_frequency=3,   # Increased from 2 to 3 to reduce noise
    save_dir="tokenizer",
    special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<sw>"]  # Added Swahili marker and mask token
):
    """
    Train a ByteLevelBPE tokenizer optimized for Swahili text data.
    
    Args:
        texts: List of text samples. If None, will load from archive.zip
        vocab_size: Size of the vocabulary (32k is optimal for most languages)
        min_frequency: Minimum frequency for a token to be included
        save_dir: Directory to save the tokenizer
        special_tokens: List of special tokens including:
            - <s>, </s>: Start/end of sequence
            - <unk>: Unknown token
            - <pad>: Padding token
            - <mask>: Masked language modeling
            - <sw>: Swahili language marker
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
    tokenizer = ByteLevelBPETokenizer()
    
    # Train tokenizer
    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the tokenizer
    tokenizer.save_model(save_dir)
    print(f"Tokenizer saved to {save_dir}")
    
    # Test the tokenizer
    test_text = texts[0][:100]  # Take first 100 chars of first text
    encoded = tokenizer.encode(test_text)
    
    print("\nTokenizer test:")
    print(f"Original text: {test_text}")
    print(f"Encoded: {encoded.tokens}")
    print(f"Decoded: {tokenizer.decode(encoded.ids)}")
    
    return tokenizer

if __name__ == "__main__":
    train_tokenizer()
