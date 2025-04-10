import sys
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_tokenizer(text: str):
    """Test the trained tokenizer with sample Swahili text."""
    # Load the trained tokenizer
    model_path = Path("tokenizer/tokenizer.model")
    vocab_path = Path("tokenizer/tokenizer.vocab")
    
    if not model_path.exists() or not vocab_path.exists():
        raise FileNotFoundError("Tokenizer files not found! Please ensure both tokenizer.model and tokenizer.vocab exist.")
    
    tokenizer = ByteLevelBPETokenizer(
        str(model_path),
        str(vocab_path)
    )
    
    # Add special tokens that were used during training
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
    tokenizer.add_special_tokens(special_tokens)
    
    # Encode the text
    encoded = tokenizer.encode(text)
    
    # Print results
    print("\nTokenizer Test Results:")
    print("-" * 50)
    print(f"Original text: {text}")
    print("-" * 50)
    print("Tokens:", " ".join(encoded.tokens))  # Join tokens with spaces for better display
    print("-" * 50)
    print("Token IDs:", encoded.ids)
    print("-" * 50)
    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded text: {decoded}")
    print("-" * 50)
    print(f"Number of tokens: {len(encoded.tokens)}")
    print(f"Average tokens per word: {len(encoded.tokens) / len(text.split()):.2f}")

if __name__ == "__main__":
    # Test with various Swahili text samples
    test_cases = [
        # Basic greeting and simple sentence
        "Jambo! Habari yako?",
        
        # Common phrase with numbers
        "Nina umri wa miaka 25 na ninaishi Nairobi.",
        
        # Complex sentence with punctuation
        "Elimu ni muhimu sana kwa maendeleo ya jamii; tunahitaji kufanya kazi kwa bidii!",
        
        # Technical terms and loan words
        "Kompyuta yangu mpya ina programu za kisasa za teknolojia.",
        
        # Traditional proverb
        "Haba na haba hujaza kibaba.",
        
        # Long compound sentence
        """Tulienda sokoni asubuhi na mapema, tukanunua matunda mbalimbali kama vile machungwa, 
        maembe, na mapapai, kisha tukarudi nyumbani kupika chakula cha mchana."""
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        test_tokenizer(test_text.strip())
