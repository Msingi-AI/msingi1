import os
import sys
import torch
import argparse
import json
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from model_v2 import Msingi2, Msingi2Config

# Set console to UTF-8 mode for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Generate text with Msingi2 model")
    parser.add_argument("--model-path", type=str, default="msingi2_best_model", 
                        help="Path to the model directory")
    parser.add_argument("--tokenizer-path", type=str, 
                        default="tokenizer/swahili_unigram_32000/tokenizer.json",
                        help="Path to the tokenizer file")
    parser.add_argument("--prompt", type=str, default="Habari ya leo,", 
                        help="Prompt to start generation with")
    parser.add_argument("--max-tokens", type=int, default=100, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, 
                        help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top-k", type=int, default=50, 
                        help="Top-k sampling (0 = disable)")
    parser.add_argument("--top-p", type=float, default=0.95, 
                        help="Top-p (nucleus) sampling (0 = disable)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, 
                        help="Repetition penalty (1.0 = no penalty)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of text samples to generate")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    
    # Add special tokens if not already present
    special_tokens = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '<mask>'
    }
    
    # Check if special tokens are already added
    for token_type, token in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            print(f"Adding {token_type}: {token}")
    
    # Add special tokens if needed
    tokenizer.add_special_tokens({
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '<mask>'
    })
    
    # Load model configuration
    model_config_path = os.path.join(args.model_path, "model_config.json")
    print(f"Loading model configuration from {model_config_path}")
    
    with open(model_config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create model config
    config = Msingi2Config(
        vocab_size=config_dict.get('vocab_size', 32000),
        block_size=config_dict.get('block_size', 1024),
        n_layer=config_dict.get('n_layer', 24),
        n_head=config_dict.get('n_head', 16),
        n_embd=config_dict.get('n_embd', 1024),
        dropout=config_dict.get('dropout', 0.1),
        bias=config_dict.get('bias', True),
        gradient_checkpointing=False  # Disable for inference
    )
    
    # Initialize model
    print(f"Initializing model with {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding size")
    model = Msingi2(config)
    
    # Load model weights
    model_path = os.path.join(args.model_path, "pytorch_model.bin")
    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    print(f"Model loaded successfully with {model.get_num_params():,} parameters")
    
    # Generate text
    print("\n===== GENERATING TEXT =====")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Parameters: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, repetition_penalty={args.repetition_penalty}")
    print(f"Generating {args.num_samples} samples, each with max {args.max_tokens} tokens\n")
    
    for i in range(args.num_samples):
        print(f"\n----- Sample {i+1} -----")
        
        # Tokenize prompt
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=args.top_p if args.top_p > 0 else None,
                repetition_penalty=args.repetition_penalty
            )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(generated_text)
    
    print("\n===== GENERATION COMPLETE =====")

if __name__ == "__main__":
    main()
