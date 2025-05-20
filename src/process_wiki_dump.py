import os
import re
import bz2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import requests
import argparse

def download_wiki_dump(output_dir="data/wiki_dumps"):
    """
    Download the latest Swahili Wikipedia dump
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for the latest Swahili Wikipedia dump
    dump_url = "https://dumps.wikimedia.org/swwiki/latest/swwiki-latest-pages-articles.xml.bz2"
    
    # Output file path
    output_file = os.path.join(output_dir, "swwiki-latest-pages-articles.xml.bz2")
    
    # Check if file already exists
    if os.path.exists(output_file):
        print(f"Dump file already exists at {output_file}")
        return output_file
    
    print(f"Downloading Swahili Wikipedia dump from {dump_url}...")
    
    # Download the file with progress reporting
    response = requests.get(dump_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_file, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc="Downloading"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    
    print(f"Download complete. File saved to {output_file}")
    return output_file

def clean_wikitext(text):
    """
    Clean Wikipedia markup from text
    """
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    
    # Remove templates
    text = re.sub(r'{{[^}]*}}', '', text, flags=re.DOTALL)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove categories and files
    text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)
    text = re.sub(r'\[\[File:[^\]]*\]\]', '', text)
    text = re.sub(r'\[\[Image:[^\]]*\]\]', '', text)
    
    # Convert internal links to just their text
    text = re.sub(r'\[\[([^|]*)\|([^\]]*)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def process_wiki_dump(dump_file, output_dir="data/wiki_sw_processed"):
    """
    Process the Wikipedia dump file and extract clean text
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file for all articles
    all_articles_file = os.path.join(output_dir, "all_articles.txt")
    
    print(f"Processing Wikipedia dump file: {dump_file}")
    
    # Open the dump file
    with bz2.open(dump_file, 'rt', encoding='utf-8') as f:
        # Initialize variables
        in_page = False
        in_title = False
        in_text = False
        current_title = ""
        current_text = ""
        page_count = 0
        
        # Process the file line by line
        for line in tqdm(f, desc="Processing dump"):
            line = line.strip()
            
            # Check for page start/end
            if "<page>" in line:
                in_page = True
                current_title = ""
                current_text = ""
            elif "</page>" in line:
                in_page = False
                
                # Process the page if it has content
                if current_title and current_text:
                    # Clean the text
                    clean_text = clean_wikitext(current_text)
                    
                    if clean_text:
                        # Save individual article
                        # More thorough filename sanitization
                        safe_title = re.sub(r'[\\/*?:"<>|]', "_", current_title)
                        safe_title = safe_title[:100]  # Limit filename length
                        article_file = os.path.join(output_dir, f"{safe_title}.txt")
                        
                        with open(article_file, "w", encoding="utf-8") as af:
                            af.write(clean_text)
                        
                        # Append to all articles file
                        with open(all_articles_file, "a", encoding="utf-8") as af:
                            af.write(f"# {current_title}\n\n")
                            af.write(f"{clean_text}\n\n")
                            af.write("=" * 80 + "\n\n")
                        
                        page_count += 1
                        
                        # Print progress every 100 pages
                        if page_count % 100 == 0:
                            print(f"Processed {page_count} pages")
            
            # Check for title start/end
            elif in_page and "<title>" in line:
                in_title = True
                current_title = line.replace("<title>", "").replace("</title>", "")
            elif in_page and "</title>" in line:
                in_title = False
            
            # Check for text start/end
            elif in_page and "<text" in line:
                in_text = True
                # Extract text content from the line
                text_start = line.find(">") + 1
                if text_start > 0:
                    current_text = line[text_start:]
                    if "</text>" in line:
                        current_text = current_text.replace("</text>", "")
                        in_text = False
            elif in_page and "</text>" in line:
                in_text = False
                current_text += " " + line.replace("</text>", "")
            
            # Append text content
            elif in_page and in_text:
                current_text += " " + line
    
    print(f"Processing complete. Extracted {page_count} articles to {output_dir}/")
    return all_articles_file

def main():
    parser = argparse.ArgumentParser(description="Download and process Swahili Wikipedia dump")
    parser.add_argument("--download-only", action="store_true", help="Only download the dump without processing")
    parser.add_argument("--process-only", action="store_true", help="Only process an existing dump file")
    parser.add_argument("--dump-file", type=str, help="Path to an existing dump file to process")
    args = parser.parse_args()
    
    # Set up directories
    dump_dir = "data/wiki_dumps"
    output_dir = "data/wiki_sw_processed"
    
    if args.process_only and args.dump_file:
        # Process an existing dump file
        process_wiki_dump(args.dump_file, output_dir)
    elif args.download_only:
        # Only download the dump
        download_wiki_dump(dump_dir)
    else:
        # Download and process
        dump_file = download_wiki_dump(dump_dir)
        process_wiki_dump(dump_file, output_dir)

if __name__ == "__main__":
    main()
