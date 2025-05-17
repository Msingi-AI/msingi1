import os
import sys
import requests
import json
import zipfile
import io
import pandas as pd
import time
from pathlib import Path

def download_from_huggingface_api(repo_id, output_dir):
    """
    Download dataset files directly from the Hugging Face API
    """
    print(f"Attempting to download dataset {repo_id} using Hugging Face API...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the list of files in the repository
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
    print(f"Fetching file list from: {api_url}")
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        files = response.json()
        
        if not files:
            print(f"No files found in the repository {repo_id}")
            return False
        
        print(f"Found {len(files)} files in the repository")
        
        # Download each file
        for file_info in files:
            if 'path' in file_info and 'type' in file_info and file_info['type'] == 'file':
                file_path = file_info['path']
                download_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
                
                print(f"Downloading {file_path} from {download_url}")
                file_response = requests.get(download_url, stream=True)
                file_response.raise_for_status()
                
                output_file = os.path.join(output_dir, os.path.basename(file_path))
                with open(output_file, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded {file_path} to {output_file}")
        
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        return False

def create_sample_swahili_dataset(output_dir):
    """
    Create a sample Swahili dataset from existing data
    """
    print("Creating a sample Swahili dataset from existing data...")
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all text files in the data directory
    text_files = [f for f in os.listdir(data_dir) if f.endswith('.txt') and f not in ['train.txt', 'valid.txt']]
    
    if not text_files:
        print("No text files found in the data directory")
        return False
    
    # Create a dataset from the text files
    dataset = []
    for i, file_name in enumerate(text_files):
        file_path = os.path.join(data_dir, file_name)
        try:
            # Read the first 1000 lines of each file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines()[:1000] if line.strip()]
            
            # Add each line as a separate example
            for j, line in enumerate(lines):
                dataset.append({
                    'id': f"{i}_{j}",
                    'text': line
                })
            
            print(f"Added {len(lines)} lines from {file_name}")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
    
    # Save the dataset as a JSON file
    output_file = os.path.join(output_dir, 'swahili_sample.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Created sample dataset with {len(dataset)} examples")
    print(f"Saved to {output_file}")
    
    # Also create a CSV version
    df = pd.DataFrame(dataset)
    csv_file = os.path.join(output_dir, 'swahili_sample.csv')
    df.to_csv(csv_file, index=False)
    print(f"Also saved as CSV to {csv_file}")
    
    return True

def main():
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset')
    
    # Try to download the CiviVox Swahili dataset
    datasets_to_try = [
        "Adeptschneider/CiviVox-Swahili-text-corpus-v2.0",
        "Adeptschneider/CiviVox-Swahili-text-corpus",
        "masakhane/masakhanews",  # Alternative Swahili dataset
        "swahili_news"            # Another alternative
    ]
    
    success = False
    for dataset_name in datasets_to_try:
        print(f"\nAttempting to download {dataset_name}...")
        success = download_from_huggingface_api(dataset_name, dataset_dir)
        if success:
            print(f"Successfully downloaded {dataset_name}")
            break
        else:
            print(f"Failed to download {dataset_name}, trying next dataset...")
            time.sleep(1)  # Wait a bit before trying the next dataset
    
    if not success:
        print("\nAll download attempts failed. Creating a sample dataset from existing data...")
        create_sample_swahili_dataset(dataset_dir)
    
    print("\nProcess completed!")
    return success

if __name__ == "__main__":
    main()
