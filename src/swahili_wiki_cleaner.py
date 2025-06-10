import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import unicodedata

class SwahiliWikiCleaner:
    """Comprehensive cleaner for Swahili Wikipedia dumps"""
    
    def __init__(self):
        self._patterns = {
            'images': re.compile(r'thumb\|\d+px\|.*?(?=\w)'),
            'galleries': re.compile(r'<gallery>.*?</gallery>', re.DOTALL),
            'links': re.compile(r'\[\[([^\|\]]*\|)?([^\]]*)\]\]'),
            'templates': re.compile(r'\{\{.*?\}\}'),
            'headers': re.compile(r'={2,}(.*?)={2,}'),
            'citations': re.compile(r'<ref>.*?</ref>', re.DOTALL),
            'lists': re.compile(r'^\*+.*?$', re.MULTILINE),
            'html_tags': re.compile(r'<[^>]+>'),
            'residual_thumbs': re.compile(r'^thumb\|.*$', re.MULTILINE),
            'numeric_bullets': re.compile(r'^\d+\.\s*'),
            'image_alt': re.compile(r'alt=.*?\|'),
            'cite_web': re.compile(r'cite web.*?url=[^\]]+'),
            'special_chars': re.compile(r'[â€œâ€â€˜â€™]'),
            'residual_wiki': re.compile(r'\b(thumb|image|file)\s*\|.*?(?=\w)'),
            'cite_templates': re.compile(r'cite [a-z]+\|.*?(?=\w)'),
            'urls': re.compile(r'https?:\/\/[^\s]+'),
            'short_lines': re.compile(r'^.{1,30}$', re.MULTILINE)
        }
        
        self._replacements = {
            "''": '"',
            "'''": '"',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'"
        }
        
        self._swahili_validation = [
            ' ya ', ' wa ', ' na ', ' katika ', ' kwa ', ' la ',
            ' ya', ' wa', ' na', ' katika', ' kwa', ' la',
            ' na ', ' ya ', ' wa ', ' katika ', ' kwa ',  # Common particles
            'ali', 'ana', 'ata', 'ili', 'ama',  # Verb prefixes
            'na', 'ya', 'wa', 'katika', 'kwa',  # Particles
            'ali', 'ana', 'ata', 'ili', 'ama'   # Verb prefixes
        ]

    def clean_text(self, text):
        """Main cleaning pipeline"""
        if not text.strip():
            return ""
            
        # Phase 1: Structural cleaning
        text = self._patterns['images'].sub('', text)
        text = self._patterns['galleries'].sub('', text)
        text = self._patterns['links'].sub(r'\2', text)
        text = self._patterns['templates'].sub('', text)
        text = self._patterns['residual_thumbs'].sub('', text)
        text = self._patterns['numeric_bullets'].sub('', text)
        text = self._patterns['image_alt'].sub('', text)
        text = self._patterns['cite_web'].sub('', text)
        text = self._patterns['special_chars'].sub('', text)
        text = self._patterns['residual_wiki'].sub('', text)
        text = self._patterns['cite_templates'].sub('', text)
        text = self._patterns['urls'].sub('', text)
        text = self._patterns['short_lines'].sub('', text)
        
        # Phase 2: Format normalization
        for old, new in self._replacements.items():
            text = text.replace(old, new)
            
        text = self._patterns['headers'].sub(r'\1:', text)
        text = self._patterns['citations'].sub('', text)
        text = self._patterns['lists'].sub('', text)
        text = self._patterns['html_tags'].sub('', text)
        
        # Enhanced sentence segmentation
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        
        # Validate Swahili content
        if not self._is_valid_swahili(text):
            return ""
            
        return self._finalize_text(text)
    
    def _finalize_text(self, text):
        """Final text normalization"""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text + '\n'

    def _is_valid_swahili(self, text):
        """More rigorous validation"""
        if len(text) < 150:  # Increased minimum length
            return False
            
        # Check for Swahili grammatical markers
        swa_markers = sum(
            text.lower().count(marker) 
            for marker in self._swahili_validation
        )
        return swa_markers >= 5 and len(text.split()) > 30

    def process_file(self, input_path, output_path, workers=4):
        """Process entire file with parallel cleaning"""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        with (input_path.open('r', encoding='utf-8') as f_in,
              output_path.open('w', encoding='utf-8') as f_out,
              ThreadPoolExecutor(max_workers=workers) as executor):
            
            cleaned = executor.map(self.clean_text, f_in)
            f_out.writelines(line for line in cleaned if line)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input text file path")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("-j", "--workers", type=int, default=4,
                       help="Number of parallel workers")
    args = parser.parse_args()
    
    cleaner = SwahiliWikiCleaner()
    cleaner.process_file(args.input, args.output, args.workers)
