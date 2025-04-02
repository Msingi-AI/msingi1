from tokenizers import Tokenizer
from pathlib import Path

def test_tokenizer(text: str):
    """Test the trained tokenizer with sample Swahili text."""
    # Load the trained tokenizer
    tokenizer_path = Path("tokenizer/tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError("Tokenizer not found! Please train the tokenizer first.")
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Encode the text
    encoded = tokenizer.encode(text)
    
    # Print results
    print("\nTokenizer Test Results:")
    print("-" * 50)
    print(f"Original text: {text}")
    print("-" * 50)
    print("Tokens:", encoded.tokens)
    print("-" * 50)
    print("Token IDs:", encoded.ids)
    print("-" * 50)
    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded text: {decoded}")
    print("-" * 50)
    print(f"Number of tokens: {len(encoded.tokens)}")

if __name__ == "__main__":
    # Test with some Swahili sentences
    test_text = """
    Jambo! Leo ni siku nzuri. Tunaenda sokoni kununua matunda na mboga. 
    Maisha ni safari ndefu, na kila siku tunajifunza mambo mapya.
    """
    test_tokenizer(test_text.strip())
