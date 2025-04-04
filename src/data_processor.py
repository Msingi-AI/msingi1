import os
import re
import json
import random
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import re
import time
import zipfile

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
    # Remove phone numbers
    text = re.sub(r'\+?\d{10,}', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\.,!?-]', '', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def process_scraped_data(force_clean: bool = False, max_samples: int = 5000) -> List[Dict[str, str]]:
    """
    Process and clean the scraped data.
    Args:
        force_clean: If True, reprocess even if clean data exists
    Returns:
        List of cleaned text samples
    """
    print("Processing scraped data...")
    
    input_file = "scraped_swahili.json"
    output_file = "cleaned_swahili.json"
    
    if not force_clean and os.path.exists(output_file):
        print(f"Loading existing cleaned data from {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    
    print(f"Processing {len(texts)} text samples...")
    cleaned_texts = []
    
    # Define categories for better domain coverage
    categories = {
        'news': [],
        'education': [],
        'culture': [],
        'general': []
    }
    
    for text in texts:
        # Basic cleaning
        text = clean_text(text)
        
        # Additional Swahili-specific cleaning
        # Remove common noise patterns
        text = re.sub(r'ADVERTISEMENT', '', text, flags=re.IGNORECASE)
        text = re.sub(r'MATANGAZO', '', text, flags=re.IGNORECASE)
        text = re.sub(r'ILANI', '', text, flags=re.IGNORECASE)
        text = re.sub(r'HABARI ZAIDI', '', text, flags=re.IGNORECASE)
        text = re.sub(r'SOMA ZAIDI', '', text, flags=re.IGNORECASE)
        text = re.sub(r'ENDELEA KUSOMA', '', text, flags=re.IGNORECASE)
        text = re.sub(r'JIUNGE NASI', '', text, flags=re.IGNORECASE)
        
        # Remove date patterns
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
        text = re.sub(r'\d{1,2}-\d{1,2}-\d{2,4}', '', text)
        
        # Remove timestamps
        text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?', '', text)
        
        # Remove common metadata markers
        text = re.sub(r'Updated:|Imehaririwa:|Published:|Edited by:|Na:', '', text)
        
        # Remove social media handles
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Fix common typos and normalize spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)  # Add space after punctuation
        text = re.sub(r'\.{2,}', '.', text)  # Replace multiple dots with single dot
        
        # Remove lines that are too short or likely headers
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines or very short ones
            if len(line) < 20:
                continue
            # Skip lines that are likely headers (all caps)
            if line.isupper():
                continue
            filtered_lines.append(line)
        
        text = ' '.join(filtered_lines)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace again
        text = text.strip()
        
        # Only keep high-quality paragraphs
        words = text.split()
        if len(words) >= 20 and len(words) <= 150:  # Adjusted length for small model training
            # Basic quality checks
            if not any(word.isdigit() for word in words[:5]):  # Avoid texts starting with numbers
                if sum(1 for w in words if len(w) > 2) / len(words) > 0.8:  # Ensure meaningful words
                    # Categorize the text
                    # More comprehensive Swahili keywords for better categorization
                    if any(word in text.lower() for word in ['habari', 'taarifa', 'ripoti', 'gazeti', 'vyombo', 'magazeti']):
                        categories['news'].append({'text': text, 'category': 'news'})
                    elif any(word in text.lower() for word in ['elimu', 'masomo', 'chuo', 'shule', 'wanafunzi', 'mwalimu', 'darasa']):
                        categories['education'].append({'text': text, 'category': 'education'})
                    elif any(word in text.lower() for word in ['utamaduni', 'mila', 'desturi', 'sanaa', 'sherehe', 'ngoma', 'tamaduni']):
                        categories['culture'].append({'text': text, 'category': 'culture'})
                    else:
                        # Check for other common domains
                        if any(word in text.lower() for word in ['siasa', 'serikali', 'bunge', 'uchaguzi']):
                            categories['general'].append({'text': text, 'category': 'politics'})
                        elif any(word in text.lower() for word in ['biashara', 'uchumi', 'fedha', 'masoko']):
                            categories['general'].append({'text': text, 'category': 'business'})
                        else:
                            categories['general'].append({'text': text, 'category': 'general'})
    
    # Balance categories and combine
    samples_per_category = max(1, min(max_samples // len(categories), 
                                     min(len(cat) for cat in categories.values() if cat)))
    
    for category, texts in categories.items():
        if texts:  # If category has any texts
            # Prioritize diverse samples by taking from different parts of the list
            step = max(1, len(texts) // samples_per_category)
            selected = texts[::step][:samples_per_category]
            cleaned_texts.extend(selected)
    
    # Shuffle to mix categories
    random.shuffle(cleaned_texts)
    
    # Save cleaned texts
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_texts, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(cleaned_texts)} cleaned articles to {output_file}")
    
    # Print some statistics
    total_words = sum(len(item['text'].split()) for item in cleaned_texts)
    total_chars = sum(len(item['text']) for item in cleaned_texts)
    categories_count = dict([(cat, sum(1 for x in cleaned_texts if x['category'] == cat)) 
                            for cat in set(x['category'] for x in cleaned_texts)])
    print(f"\nDataset Statistics:")
    print(f"num_samples: {len(cleaned_texts)}")
    print(f"total_words: {total_words:,}")
    print(f"total_characters: {total_chars:,}")
    print(f"avg_words_per_sample: {total_words / len(cleaned_texts):.2f}")
    print(f"avg_chars_per_sample: {total_chars / len(cleaned_texts):.2f}")
    print(f"\nCategory Distribution:")
    for cat, count in categories_count.items():
        print(f"{cat}: {count} samples ({count/len(cleaned_texts)*100:.1f}%)")
    
    # Print a few sample texts
    print("\nSample texts:\n")
    for i, item in enumerate(cleaned_texts[:3], 1):
        preview = item['text'][:200] + "..."
        print(f"Sample {i} ({item['category']}):\n{preview}\n")
    
    return cleaned_texts

def load_training_data(file_path: str) -> List[Dict[str, str]]:
    """Load and process training data from a file."""
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    # Basic quality check - just remove empty lines
                    words = text.split()
                    if len(words) > 0:
                            # Categorize the text
                            if any(word in text.lower() for word in ['habari', 'taarifa', 'ripoti', 'gazeti']):
                                category = 'news'
                            elif any(word in text.lower() for word in ['elimu', 'masomo', 'shule']):
                                category = 'education'
                            elif any(word in text.lower() for word in ['utamaduni', 'mila', 'desturi']):
                                category = 'culture'
                            elif any(word in text.lower() for word in ['siasa', 'serikali', 'bunge']):
                                category = 'politics'
                            elif any(word in text.lower() for word in ['biashara', 'uchumi', 'fedha']):
                                category = 'business'
                            else:
                                category = 'general'
                            
                            texts.append({
                                'text': text,
                                'category': category,
                                'source': 'dataset'
                            })
                            
                            # No sample limit, collect all data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
    
    return texts

def load_swahili_dataset(max_samples: int = 5000) -> List[Dict[str, str]]:
    """
    Load the combined Swahili dataset.
    Returns:
        List of text samples
    """
    all_texts = []
    
    # Load from original dataset files
    data_dir = "data/Swahili data/Swahili data"
    train_file = os.path.join(data_dir, "train.txt")
    valid_file = os.path.join(data_dir, "valid.txt")
    
    # Load ALL training data
    train_texts = load_training_data(train_file)
    print(f"Loaded {len(train_texts)} samples from training set")
    train_stats = get_word_stats(train_texts)
    print(f"Training set stats:")
    print(f"- Total words: {format_number(train_stats['total_words'])}")
    print(f"- Unique words: {format_number(train_stats['unique_words'])}")
    print(f"- Average words per sample: {train_stats['words_per_sample']:.1f}")
    all_texts.extend(train_texts)
    
    # Load ALL validation data
    valid_texts = load_training_data(valid_file)
    print(f"\nLoaded {len(valid_texts)} samples from validation set")
    valid_stats = get_word_stats(valid_texts)
    print(f"Validation set stats:")
    print(f"- Total words: {format_number(valid_stats['total_words'])}")
    print(f"- Unique words: {format_number(valid_stats['unique_words'])}")
    print(f"- Average words per sample: {valid_stats['words_per_sample']:.1f}")
    all_texts.extend(valid_texts)
    
    # Process scraped data
    print("\nProcessing scraped data...")
    scraped_texts = process_scraped_data(force_clean=True)
    print(f"Loaded {len(scraped_texts)} samples from scraped data")
    scraped_stats = get_word_stats(scraped_texts)
    print(f"Scraped data stats:")
    print(f"- Total words: {format_number(scraped_stats['total_words'])}")
    print(f"- Unique words: {format_number(scraped_stats['unique_words'])}")
    print(f"- Average words per sample: {scraped_stats['words_per_sample']:.1f}")
    all_texts.extend(scraped_texts)
    
    # Shuffle the combined dataset
    random.shuffle(all_texts)
    
    # Get combined statistics
    combined_stats = get_word_stats(all_texts)
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"COMBINED DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"Total samples: {format_number(len(all_texts))}")
    print(f"Total words: {format_number(combined_stats['total_words'])}")
    print(f"Unique words: {format_number(combined_stats['unique_words'])}")
    print(f"Average words per sample: {combined_stats['words_per_sample']:.1f}")
    print(f"Average word length: {combined_stats['avg_word_length']:.1f} characters")
    
    # Category distribution
    categories = {}
    sources = {'dataset': 0, 'scraped': 0}
    
    for item in all_texts:
        categories[item['category']] = categories.get(item['category'], 0) + 1
        sources[item.get('source', 'scraped')] = sources.get(item.get('source', 'scraped'), 0) + 1
    
    print("\nCategory Distribution:")
    for cat, count in categories.items():
        print(f"{cat}: {format_number(count)} samples ({count/len(all_texts)*100:.1f}%)")
    
    print("\nSource Distribution:")
    for source, count in sources.items():
        print(f"{source}: {format_number(count)} samples ({count/len(all_texts)*100:.1f}%)")
    
    # Print sample texts
    print("\nSample texts:\n")
    for i, item in enumerate(all_texts[:3], 1):
        preview = item['text'][:200] + "..."
        print(f"Sample {i} ({item['category']}, {item.get('source', 'scraped')}):\n{preview}\n")
    
    return all_texts

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

def format_number(num):
    """Format large numbers with commas and abbreviate if over a million."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    return f"{num:,}"

def get_word_stats(texts):
    """Get detailed word statistics from texts."""
    total_words = sum(len(item['text'].split()) for item in texts)
    unique_words = set()
    word_lengths = []
    
    for item in texts:
        words = item['text'].split()
        unique_words.update(words)
        word_lengths.extend(len(word) for word in words)
    
    return {
        'total_words': total_words,
        'unique_words': len(unique_words),
        'avg_word_length': sum(word_lengths) / len(word_lengths) if word_lengths else 0,
        'words_per_sample': total_words / len(texts) if texts else 0
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
