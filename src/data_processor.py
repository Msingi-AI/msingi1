import os
import zipfile
import json
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import re
import time

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\.,!?-]', '', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def scrape_swahili_news(max_articles: int = 1000, force_fresh: bool = False) -> List[str]:
    """
    Scrape Swahili news articles from various sources.
    Args:
        max_articles: Maximum number of articles to scrape per source
        force_fresh: If True, ignore cached data and scrape fresh content
    """
    print("Scraping Swahili news articles...")
    all_texts = []  # Keep track of all texts
    
    # List of Swahili news sources
    sources = [
        # Tanzania Government Sites
        {
            'url': 'https://www.tanzania.go.tz/home/pages/3357',
            'article_selector': '.article-content, .content-area',
            'text_selector': 'p, .text-content'
        },
        {
            'url': 'https://www.ikulu.go.tz',
            'article_selector': '.news-item, .article',
            'text_selector': '.news-content, p'
        },
        # Tanzania News Sites
        {
            'url': 'https://www.mwananchi.co.tz',
            'article_selector': 'article, .article-item',
            'text_selector': '.article-summary, .article-content, p'
        },
        {
            'url': 'https://www.mwananchi.co.tz/mw/habari',
            'article_selector': 'article, .article-item',
            'text_selector': '.article-summary, .article-content, p'
        },
        {
            'url': 'https://www.mwananchi.co.tz/mw/michezo',
            'article_selector': 'article, .article-item',
            'text_selector': '.article-summary, .article-content, p'
        },
        {
            'url': 'https://habarileo.co.tz',
            'article_selector': '.article, .post',
            'text_selector': '.article-content, .entry-content'
        },
        {
            'url': 'https://www.majira.co.tz',
            'article_selector': '.post, .article',
            'text_selector': '.entry-content, .article-content'
        },
        # Educational Resources
        {
            'url': 'https://www.elimu.go.tz',
            'article_selector': '.content-area, .page-content',
            'text_selector': 'p, .text-content'
        },
        {
            'url': 'https://www.out.ac.tz',
            'article_selector': '.content, .article',
            'text_selector': 'p, .text-content'
        },
        # Wikipedia Main Articles
        {
            'url': 'https://sw.wikipedia.org/wiki/Tanzania',
            'article_selector': '.mw-parser-output',
            'text_selector': 'p'
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Kenya',
            'article_selector': '.mw-parser-output',
            'text_selector': 'p'
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Afrika_Mashariki',
            'article_selector': '.mw-parser-output',
            'text_selector': 'p'
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Kiswahili',
            'article_selector': '.mw-parser-output',
            'text_selector': 'p'
        },
        # Wikipedia Categories
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Historia_ya_Tanzania',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Utamaduni_wa_Tanzania',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Elimu_Tanzania',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Mawasiliano_Tanzania',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Uchumi_wa_Tanzania',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Lugha_za_Tanzania',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Dini_Tanzania',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        # Additional Wikipedia Categories
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Fasihi',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Sanaa',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Muziki',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        },
        {
            'url': 'https://sw.wikipedia.org/wiki/Jamii:Michezo',
            'article_selector': '.mw-category-group li a',
            'text_selector': None,
            'is_category': True
        }
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'sw,en-US;q=0.7,en;q=0.3',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }

    def extract_text_from_url(url: str) -> List[str]:
        """Helper function to extract text from a URL."""
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get all paragraphs
            paragraphs = soup.find_all('p', recursive=True)
            texts = []
            
            for p in paragraphs:
                # Remove references and other unwanted elements
                [s.extract() for s in p.select('.reference, .mw-editsection')]
                text = p.get_text().strip()
                if text:
                    text = clean_text(text)
                    if len(text.split()) >= 20:  # Only keep substantial paragraphs
                        texts.append(text)
            
            return texts
        except Exception as e:
            print(f"Error extracting text from {url}: {str(e)}")
            return []
    
    for source in sources:
        source_texts = []  # Keep track of texts from this source
        try:
            print(f"\nScraping from {source['url']}...")
            response = requests.get(source['url'], headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Handle category pages differently
            if source.get('is_category', False):
                # Get all article links from category
                links = soup.select(source['article_selector'])
                if links:
                    print(f"Found {len(links)} articles in category")
                    
                    # Visit each article
                    for link in links[:max_articles // len(sources)]:
                        try:
                            href = link.get('href')
                            if href:
                                if not href.startswith('http'):
                                    article_url = 'https://sw.wikipedia.org' + href
                                else:
                                    article_url = href
                                print(f"Visiting article: {article_url}")
                                article_texts = extract_text_from_url(article_url)
                                source_texts.extend(article_texts)
                                time.sleep(1)  # Be nice to Wikipedia
                        except Exception as e:
                            print(f"Error processing category link: {str(e)}")
                            continue
                            
            else:
                # Try different selectors
                selectors = source['article_selector'].split(', ')
                articles = []
                for selector in selectors:
                    articles.extend(soup.select(selector))
                    
                print(f"Found {len(articles)} potential articles")
                
                for article in articles[:max_articles // len(sources)]:
                    # Try different text selectors
                    text_selectors = source['text_selector'].split(', ')
                    paragraphs = []
                    for selector in text_selectors:
                        paragraphs.extend(article.select(selector))
                    
                    if not paragraphs and 'wiki' in source['url']:
                        # Special handling for Wikipedia: get all paragraphs
                        paragraphs = article.find_all('p', recursive=True)
                    
                    texts = []
                    for p in paragraphs:
                        # Remove references and other unwanted elements
                        [s.extract() for s in p.select('.reference, .mw-editsection')]
                        text = p.get_text().strip()
                        if text:
                            texts.append(text)
                    
                    if texts:
                        text = ' '.join(texts)
                        text = clean_text(text)
                        
                        if len(text.split()) >= 20:  # Only keep substantial paragraphs
                            source_texts.append(text)
            
            # Add source texts to all texts
            all_texts.extend(source_texts)
            print(f"Successfully extracted {len(source_texts)} articles with content")
            
        except Exception as e:
            print(f"Error scraping {source['url']}: {str(e)}")
            continue
        
        # Be nice to servers
        time.sleep(2)
    
    if all_texts:
        # Remove duplicates while preserving order
        all_texts = list(dict.fromkeys(all_texts))
        
        # Save scraped texts to a file
        output_file = "scraped_swahili.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_texts, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved {len(all_texts)} unique scraped articles to {output_file}")
    else:
        print("\nNo articles were successfully scraped")
    
    return all_texts

def load_swahili_dataset() -> List[str]:
    """
    Load Swahili text from scraped content.
    """
    texts = []
    
    # Force a fresh scrape
    print("Performing fresh scrape of Swahili content...")
    texts.extend(scrape_swahili_news(force_fresh=True))
    
    return texts

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

def combine_datasets(swahili_texts: List[str], archive_texts: List[str]) -> List[str]:
    """
    Combine texts from multiple sources and remove duplicates.
    """
    all_texts = swahili_texts + archive_texts
    # Remove duplicates while preserving order
    unique_texts = list(dict.fromkeys(all_texts))
    print(f"Combined dataset size: {len(unique_texts)} documents")
    return unique_texts

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
    # Load both datasets
    swahili_texts = load_swahili_dataset()
    archive_texts = extract_dataset("archive.zip")
    
    # Combine datasets
    all_texts = combine_datasets(swahili_texts, archive_texts)
    
    # Get and print statistics
    stats = get_dataset_stats(all_texts)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
    # Print sample texts
    print("\nSample texts:")
    for i, text in enumerate(all_texts[:3]):
        print(f"\nSample {i+1}:")
        print(text[:200] + "...")
