import os
import sys
from transformers import PreTrainedTokenizerFast

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def add_eot_token(tokenizer_path, save_path=None):
    """Add <eot> token to the tokenizer and save it."""
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Print current special tokens
    print(f"Current special tokens: {tokenizer.all_special_tokens}")
    print(f"Current vocabulary size: {tokenizer.vocab_size}")
    
    # Check if <eot> token already exists
    if "<eot>" in tokenizer.all_special_tokens:
        print("<eot> token already exists in the tokenizer.")
    else:
        # Add <eot> as the eos_token
        print("Adding <eot> token as eos_token...")
        tokenizer.add_special_tokens({'eos_token': '<eot>'})
        print(f"Updated special tokens: {tokenizer.all_special_tokens}")
        print(f"Updated vocabulary size: {tokenizer.vocab_size}")
        
        # Save the updated tokenizer
        if save_path is None:
            save_path = os.path.dirname(tokenizer_path)
        
        tokenizer.save_pretrained(save_path)
        print(f"Saved updated tokenizer to {save_path}")
    
    # Test the tokenizer
    test_sentences = [
        "This is a test sentence.<eot>Another sentence.",
        "Habari ya leo.<eot>Habari nzuri sana."
    ]
    
    print("\nTesting tokenizer with <eot> token:")
    for test in test_sentences:
        # Encode
        encoded = tokenizer.encode(test)
        
        # Print results
        print(f"\nOriginal: {test}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded)}")
        print(f"Number of tokens: {len(encoded)}")
        
        # Check if <eot> is properly tokenized
        if tokenizer.eos_token_id in encoded:
            eot_positions = [i for i, token_id in enumerate(encoded) if token_id == tokenizer.eos_token_id]
            print(f"<eot> token found at positions: {eot_positions}")
        else:
            print("<eot> token not found in encoded sequence!")
    
    return tokenizer

if __name__ == "__main__":
    # Path to the tokenizer file
    tokenizer_path = "tokenizer/swahili_bpe_32000/tokenizer.json"
    
    # Add <eot> token
    add_eot_token(tokenizer_path)
    
    # Also update the transformers version
    transformers_path = "tokenizer/swahili_bpe_32000/transformers"
    add_eot_token(tokenizer_path, transformers_path)
