import os
import zipfile
import json
from typing import List, Dict
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

def extract_dataset(archive_path: str, extract_dir: str = "data") -> List[str]:
    """
    Extract the dataset from zip file and return list of text content.
    
    Args:
        archive_path: Path to the archive.zip file
        extract_dir: Directory to extract files to
    
    Returns:
        List of text content from extracted files
    """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    text_files = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    text_files.append(f.read())
    
    return text_files

class SwahiliDataset(Dataset):
    """Dataset class for Swahili text data."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: ByteLevelBPETokenizer,
        max_length: int = 1024,
        stride: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            tokenizer: ByteLevelBPE tokenizer
            max_length: Maximum sequence length
            stride: Stride for sliding window tokenization
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []
        
        # Process all texts into overlapping chunks
        for text in texts:
            # Tokenize the entire text
            encoded = self.tokenizer.encode(text)
            
            # Create chunks with overlap
            for i in range(0, len(encoded.ids), stride):
                chunk_ids = encoded.ids[i:i + max_length]
                if len(chunk_ids) < max_length:  # Pad if needed
                    chunk_ids = chunk_ids + [self.tokenizer.token_to_id("<pad>")] * (max_length - len(chunk_ids))
                elif len(chunk_ids) > max_length:  # Truncate if needed
                    chunk_ids = chunk_ids[:max_length]
                
                # Create input_ids and labels
                self.examples.append({
                    "input_ids": chunk_ids[:-1],  # All tokens except last
                    "labels": chunk_ids[1:]  # All tokens except first
                })
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example
        
        Returns:
            Dictionary with input_ids and labels as torch tensors
        """
        example = self.examples[idx]
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long)
        }

def get_dataset_stats(texts: List[str]) -> Dict:
    """
    Get basic statistics about the dataset.
    """
    total_chars = sum(len(text) for text in texts)
    total_words = sum(len(text.split()) for text in texts)
    
    return {
        "num_samples": len(texts),
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_sample_length": total_chars / len(texts) if texts else 0,
        "avg_words_per_sample": total_words / len(texts) if texts else 0
    }

if __name__ == "__main__":
    # Example usage
    texts = extract_dataset("archive.zip")
    stats = get_dataset_stats(texts)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
