import os
import sys
import torch
import argparse
import json
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from model import MsingiConfig, Msingi1

def load_model(model_path):
    """Load the Msingi1 model from checkpoint directory"""
    # Load model config
    config_path = os.path.join(model_path, "model_config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Model config not found at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    
    # Create model config
    model_config = MsingiConfig(
        vocab_size=config_dict.get("vocab_size", 32000),
        max_position_embeddings=config_dict.get("max_position_embeddings", 1024),
        n_layer=config_dict.get("n_layer", 12),
        n_head=config_dict.get("n_head", 12),
        n_embd=config_dict.get("n_embd", 768),
        intermediate_size=config_dict.get("intermediate_size", 3072),
        dropout=config_dict.get("dropout", 0.1),
        rotary_emb=config_dict.get("rotary_emb", True),
        gradient_checkpointing=False  # Disable for inference
    )
    
    # Print model info
    print(f"Loading model with {model_config.n_layer} layers, {model_config.n_head} heads, {model_config.n_embd} embedding size")
    print(f"Approximate parameter count: {(model_config.n_layer * (12 * model_config.n_embd * model_config.n_embd) + model_config.vocab_size * model_config.n_embd) / 1_000_000:.1f}M")
    
    # Create model
    model = Msingi1(model_config)
    
    # Load weights
    weights_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise ValueError(f"Model weights not found at {weights_path}")
    
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()  # Set to evaluation mode
    
    return model, model_config

def load_tokenizer(tokenizer_path):
    """Load the tokenizer"""
    if not os.path.exists(tokenizer_path):
        raise ValueError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    return tokenizer

def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    device="cpu"
):
    """Generate text using the Msingi1 model"""
    model = model.to(device)
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.size(1)
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits for next token
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for token_id in input_ids[0].tolist():
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=top_k)[0][:, -1, None]
                next_token_logits[next_token_logits < indices_to_remove] = -float("Inf")
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = -float("Inf")
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check if we've generated an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(input_ids[0][input_length:], skip_special_tokens=True)
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": full_text
    }

def main():
    parser = argparse.ArgumentParser(description="Generate text with Msingi1 model")
    parser.add_argument("--model-path", type=str, default="msingi1_best_model", help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/swahili_unigram_32000/tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default="Habari ya leo. Jina langu ni", help="Text prompt to start generation")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--examples", type=int, default=3, help="Number of examples to generate")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, model_config = load_model(args.model_path)
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    print(f"Using device: {args.device}")
    print(f"Generating {args.examples} examples with prompt: '{args.prompt}'")
    
    # Generate multiple examples
    for i in range(args.examples):
        print(f"\n--- Example {i+1} ---")
        result = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device
        )
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"Full text: {result['full_text']}")

if __name__ == "__main__":
    main()
