import os
from tokenizers import ByteLevelBPETokenizer
from data_processor import extract_dataset, get_dataset_stats

def train_tokenizer(
    texts,
    vocab_size=50000,
    min_frequency=2,
    save_dir="tokenizer",
    special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
):
    """
    Train a ByteLevelBPE tokenizer on Swahili text data.
    
    Args:
        texts: List of text samples
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
        save_dir: Directory to save the tokenizer
        special_tokens: List of special tokens to add
    """
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

def main():
    # Load and process dataset
    print("Loading dataset...")
    texts = extract_dataset("archive.zip")
    
    # Print dataset statistics
    stats = get_dataset_stats(texts)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Train tokenizer
    tokenizer = train_tokenizer(texts)

if __name__ == "__main__":
    main()
