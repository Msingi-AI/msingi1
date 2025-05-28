#!/usr/bin/env python3
"""
Download Msingi1 data from Google Drive
This script downloads the Msingi1 dataset and tokenizer files from Google Drive.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def install_gdown():
    """Install gdown if not already installed"""
    try:
        import gdown
        print("gdown is already installed.")
        return True
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        return True

def download_folder(folder_id, output_dir, quiet=False):
    """
    Download an entire folder from Google Drive
    
    Args:
        folder_id: The Google Drive folder ID
        output_dir: The directory to save the downloaded files
        quiet: Whether to suppress output
    """
    import gdown
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the folder
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url=url, output=output_dir, quiet=quiet)
    
    print(f"Downloaded folder contents to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download Msingi1 data from Google Drive")
    parser.add_argument("--folder-id", type=str, default="16Yzgk3WOIknO4rS7XZvUlzx0GnZdUkCV",
                        help="Google Drive folder ID")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save the downloaded files")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")
    
    args = parser.parse_args()
    
    # Install gdown if needed
    install_gdown()
    
    # Download the folder
    download_folder(args.folder_id, args.output_dir, args.quiet)
    
    # List downloaded files
    if not args.quiet:
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(args.output_dir):
            level = root.replace(args.output_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"{sub_indent}{file} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
