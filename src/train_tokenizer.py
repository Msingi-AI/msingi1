import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Union
from tokenizers import ByteLevelBPETokenizer

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processor import load_swahili_dataset

def train_tokenizer(
    texts: Union[List[str], List[Dict[str, str]]] = None,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    save_dir: str = "tokenizer",
    special_tokens: List[str] = ["<s>", "</s>", "<unk>", "<pad>"]
) -> ByteLevelBPETokenizer:
    """
    Train a ByteLevelBPE tokenizer on Swahili text data.
    
    Args:
        texts: List of text samples or dictionaries with 'text' key
        vocab_size: Size of the vocabulary (reduced for small model)
        min_frequency: Minimum frequency for a token to be included
        save_dir: Directory to save the tokenizer
        special_tokens: List of special tokens to add
    """
    if texts is None:
        print("Loading dataset...")
        texts = load_swahili_dataset(max_samples=2000)  # Limit samples for focused training
    
    # Extract text from dictionaries if needed
    if isinstance(texts[0], dict):
        texts = [item['text'] for item in texts]
    
    print(f"\nTraining tokenizer with vocab size {vocab_size}...")
    
    # Initialize tokenizer with Swahili-specific settings
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    
    # Save texts to temporary file for training
    temp_file = "temp_training_data.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    # Train tokenizer
    tokenizer.train(
        files=[temp_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )
    
    # Remove temporary file
    os.remove(temp_file)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the tokenizer files in the requested format
    tokenizer.save(f"{save_dir}/tokenizer.model")  # Save the model
    
    # Save vocabulary in a custom format
    vocab = tokenizer.get_vocab()
    with open(f"{save_dir}/tokenizer.vocab", "w", encoding="utf-8") as f:
        for token, id in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\t{id}\n")
    
    print(f"Tokenizer saved to {save_dir}")
    
    # Test the tokenizer on different types of text
    test_texts = [
        "Jambo! Habari yako?",  # Greetings
        "Elimu ni muhimu sana kwa maendeleo.",  # Education
        "Utamaduni wetu ni hazina kubwa.",  # Culture
        "Serikali imetangaza mpango mpya."  # Politics
    ]
    
    print("\nTokenizer tests:")
    for test in test_texts:
        encoded = tokenizer.encode(test)
        print(f"\nOriginal: {test}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Decoded: {tokenizer.decode(encoded.ids)}")
    
    # Print vocabulary statistics
    print(f"\nVocabulary Statistics:")
    print(f"Total vocab size: {len(vocab)}")
    print(f"Special tokens: {special_tokens}")
    
    return tokenizer

if __name__ == "__main__":
    train_tokenizer()
