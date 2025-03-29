import os
import zipfile
import json
from typing import List, Dict
from pathlib import Path

def extract_dataset(archive_path: str, extract_dir: str = "data") -> List[str]:
    """
    Extract the dataset from zip file and return list of text content.
    
    Args:
        archive_path: Path to the archive.zip file
        extract_dir: Directory to extract files to
    
    Returns:
        List of text content from the dataset
    """
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    texts = []
    
    # Extract the zip file
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
        # Process each file in the archive
        for file_name in zip_ref.namelist():
            if file_name.endswith('.txt'):
                # Read text files directly from zip
                with zip_ref.open(file_name) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    texts.append(content)
            elif file_name.endswith('.json'):
                # Parse JSON files
                with zip_ref.open(file_name) as f:
                    content = json.loads(f.read().decode('utf-8', errors='ignore'))
                    # If the JSON contains a 'text' field, extract it
                    if isinstance(content, dict) and 'text' in content:
                        texts.append(content['text'])
                    elif isinstance(content, list):
                        # If it's a list of documents, extract text from each
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                texts.append(item['text'])
    
    # Remove empty strings and strip whitespace
    texts = [text.strip() for text in texts if text.strip()]
    
    print(f"Loaded {len(texts)} text samples from the dataset")
    return texts

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
