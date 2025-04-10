import sys
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import ByteLevel as ByteLevelProcessor

def convert_tokenizer():
    """Convert the existing tokenizer to HuggingFace format."""
    # Load the existing tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "tokenizer/tokenizer.model",
        "tokenizer/tokenizer.vocab"
    )
    
    # Add special tokens
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
    tokenizer.add_special_tokens(special_tokens)
    
    # Save in HuggingFace format
    tokenizer.save("tokenizer/tokenizer.json")
    print("Tokenizer converted and saved as tokenizer.json")

if __name__ == "__main__":
    convert_tokenizer()
