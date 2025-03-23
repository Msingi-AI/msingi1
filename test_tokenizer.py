from tokenizers import ByteLevelBPETokenizer
import json
import sys

def test_tokenizer():
    # Set console to UTF-8 mode
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    # Load the tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "tokenizer/vocab.json",
        "tokenizer/merges.txt"
    )

    # Test sentences in Swahili
    test_sentences = [
        "Habari ya leo?",  # Hello, how are you today?
        "Ninafurahi kukutana nawe.",  # I'm happy to meet you
        "Karibu Tanzania, nchi nzuri.",  # Welcome to Tanzania, beautiful country
    ]

    print("Testing tokenizer with sample Swahili sentences:\n")
    
    for sentence in test_sentences:
        # Encode the sentence
        encoded = tokenizer.encode(sentence)
        
        print(f"\nOriginal text: {sentence}")
        print(f"Tokens: {encoded.tokens}")
        print(f"IDs: {encoded.ids}")
        
        # Decode back to text
        decoded = tokenizer.decode(encoded.ids)
        print(f"Decoded text: {decoded}")
        print("-" * 50)

if __name__ == "__main__":
    test_tokenizer()
