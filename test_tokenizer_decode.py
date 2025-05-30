import argparse
from transformers import PreTrainedTokenizerFast

def test_tokenizer_encoding_decoding(tokenizer_path):
    """Test how a tokenizer encodes and decodes Swahili text."""
    print(f"Loading tokenizer from {tokenizer_path}")
    
    # Load the tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Sample Swahili sentences to test
    test_sentences = [
        "Jambo! Habari yako?",
        "Nina umri wa miaka 25 na ninaishi Nairobi.",
        "Elimu ni muhimu sana kwa maendeleo ya jamii.",
        "Kompyuta yangu mpya ina programu za kisasa.",
        "Haba na haba hujaza kibaba.",
        "Ninapenda kusoma vitabu vya Kiswahili na kusikiliza muziki.",
        "Mtu ni watu.<eot>Mchana mwema."
    ]
    
    # Test encoding and decoding for each sentence
    for i, sentence in enumerate(test_sentences):
        print(f"\n--- Test Sentence {i+1} ---")
        print(f"Original: {sentence}")
        
        # Encode the sentence
        encoded = tokenizer.encode(sentence)
        tokens = tokenizer.convert_ids_to_tokens(encoded)
        
        print(f"Token IDs: {encoded}")
        print(f"Tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        
        # Decode back to text
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")
        
        # Check if decoding matches the original
        if decoded == sentence:
            print("✓ Perfect reconstruction")
        else:
            print("⚠ Reconstruction differs from original")
            # Show differences character by character
            print("Differences (original vs decoded):")
            for j, (orig_char, dec_char) in enumerate(zip(sentence, decoded)):
                if orig_char != dec_char:
                    print(f"  Position {j}: '{orig_char}' vs '{dec_char}'")
            
            # Check if lengths differ
            if len(sentence) != len(decoded):
                print(f"  Length difference: original={len(sentence)}, decoded={len(decoded)}")
                if len(sentence) < len(decoded):
                    print(f"  Extra characters in decoded: '{decoded[len(sentence):]}'")
                else:
                    print(f"  Missing characters in decoded: '{sentence[len(decoded):]}'")
    
    # Test handling of special tokens
    print("\n--- Special Token Handling ---")
    
    # Test with multiple <eot> tokens
    multi_doc = "Document one.<eot>Document two.<eot>Document three."
    encoded_multi = tokenizer.encode(multi_doc)
    tokens_multi = tokenizer.convert_ids_to_tokens(encoded_multi)
    
    print(f"Multi-document text: {multi_doc}")
    print(f"Tokens: {tokens_multi}")
    
    # Find positions of <eot> tokens
    eot_positions = [i for i, token in enumerate(tokens_multi) if token == "<eot>"]
    print(f"<eot> token positions: {eot_positions}")
    
    # Decode back
    decoded_multi = tokenizer.decode(encoded_multi)
    print(f"Decoded multi-document: {decoded_multi}")
    
    # Test batch encoding/decoding
    print("\n--- Batch Processing ---")
    batch = test_sentences[:3]  # Take first 3 sentences
    
    # Encode batch
    encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")
    print(f"Batch encoding shape: {encoded_batch.input_ids.shape}")
    
    # Decode each item in batch
    for j, ids in enumerate(encoded_batch.input_ids):
        decoded_text = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"Batch item {j} decoded: {decoded_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tokenizer encoding and decoding")
    parser.add_argument("--tokenizer-path", default="tokenizer/new_swahili_unigram_32000/transformers", 
                        help="Path to the tokenizer directory")
    
    args = parser.parse_args()
    test_tokenizer_encoding_decoding(args.tokenizer_path)
