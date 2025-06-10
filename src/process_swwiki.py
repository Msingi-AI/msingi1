#!/usr/bin/env python3
"""
Swahili Wikipedia Dump Processor
Extracts and cleans text from swwiki-latest-pages-articles-multistream.xml.bz2
"""

import bz2
import re
from bs4 import BeautifulSoup
import html
from tqdm import tqdm

# Swahili-specific cleaning patterns
SWAHILI_CLEAN_PATTERNS = [
    (r'\{\{.*?\}\}', ''),          # Templates
    (r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r'\2'),  # Linked phrases
    (r'\[\[([^\]]+)\]\]', r'\1'),           # Simple links
    (r'\b[0-9]+\b', ''),            # Isolated numbers
    (r'\={2,}(.*?)\={2,}', r'\1'), # Headers
    (r'\*+\s?', ''),                # List items
    (r'\#+\s?', ''),                # Numbered items
    (r'\;.*?:', ''),                 # Definition lists
    (r'\&lt;ref.*?\&gt;.*?\&lt;/ref\&gt;', ''), # References
    (r'\&lt;br\s?/\&gt;', '\n'),       # Line breaks
]

def clean_swahili_text(text):
    """Clean Swahili wiki text while preserving language structure"""
    # Basic HTML unescape
    text = html.unescape(text)
    
    # Apply Swahili-specific cleaning
    for pattern, repl in SWAHILI_CLEAN_PATTERNS:
        text = re.sub(pattern, repl, text, flags=re.DOTALL)
    
    # Normalize whitespace and quotes
    text = ' '.join(text.split())
    text = text.replace('"', "'")
    
    return text

def process_dump(input_path, output_path):
    """Process the compressed XML dump"""
    with bz2.open(input_path, 'rb') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        article_count = 0
        text_buffer = ""
        in_text = False
        
        # Process with progress bar
        for line in tqdm(f_in, desc='Processing dump'):
            line = line.decode('utf-8').strip()
            
            if '<text ' in line:
                in_text = True
                text_buffer = line[line.find('>')+1:]
            elif in_text:
                if '</text>' in line:
                    text_buffer += line[:line.find('</text>')]
                    in_text = False
                    
                    # Process completed article
                    try:
                        # Basic extraction
                        text = BeautifulSoup(text_buffer, 'html.parser').get_text()
                        text = clean_swahili_text(text)
                        
                        if text.strip() and len(text.split()) > 20:  # Minimum 20 words
                            f_out.write(text + '\n')
                            article_count += 1
                    except:
                        continue
                    
                    text_buffer = ""
                else:
                    text_buffer += line
    
    print(f"Processed {article_count} articles")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Swahili Wikipedia dump')
    parser.add_argument('input', help='Input .xml.bz2 dump file')
    parser.add_argument('output', help='Output .txt file')
    args = parser.parse_args()
    
    process_dump(args.input, args.output)
