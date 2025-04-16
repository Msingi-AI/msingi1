import torch
from model import Msingi1, MsingiConfig
from tokenizers import Tokenizer
import argparse
from train import TrainingConfig
import os
from collections import defaultdict

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.9, top_k=40, top_p=0.92, repetition_penalty=1.2, no_repeat_ngram_size=3):
    """Generate text from a prompt with advanced sampling techniques."""
    # Encode the prompt
    input_ids = tokenizer.encode(prompt).ids
    input_ids = [tokenizer.token_to_id("<s>")] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    
    # Move to the same device as model
    input_ids = input_ids.to(next(model.parameters()).device)
    
    # Track generated n-grams for no_repeat_ngram_size
    generated_ngrams = {}
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if token_id in range(len(next_token_logits)):
                        next_token_logits[token_id] /= repetition_penalty
            
            # Apply n-gram blocking
            if no_repeat_ngram_size > 0 and len(input_ids[0]) >= no_repeat_ngram_size:
                # Extract the last n-1 tokens
                prev_tokens = input_ids[0][-(no_repeat_ngram_size-1):].tolist()
                for banned_token in get_banned_tokens(prev_tokens, generated_ngrams, no_repeat_ngram_size):
                    next_token_logits[banned_token] = float('-inf')
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if we predict the end token
            if next_token.item() == tokenizer.token_to_id("</s>"):
                break
                
            # Add the predicted token to input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Update n-gram tracking
            if no_repeat_ngram_size > 0:
                update_ngrams(input_ids[0].tolist(), generated_ngrams, no_repeat_ngram_size)
    
    # Decode the generated tokens
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    # Clean up the text (remove special tokens)
    generated_text = generated_text.replace("<s>", "").replace("</s>", "").strip()
    
    return generated_text

def load_model(checkpoint_path, device='cpu'):
    """Load the model from a checkpoint file."""
    print(f"Loading model from {checkpoint_path} on {device}")
    
    try:
        # Load checkpoint (with weights_only=False for PyTorch 2.6 compatibility)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("Checkpoint loaded successfully")
        
        # Get model config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Using config from checkpoint: {config.n_layer} layers, {config.n_embd} hidden size")
        else:
            # Fallback to default config for our small model
            config = MsingiConfig(
                vocab_size=32000,
                max_position_embeddings=1024,
                n_layer=6,
                n_head=6,
                n_embd=384,
                intermediate_size=1536,
                dropout=0.1,
                rotary_emb=True,
                gradient_checkpointing=False  # Turn off for inference
            )
            print("Using default config (checkpoint didn't contain config)")
        
        # Initialize model
        model = Msingi1(config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        print("Model loaded successfully")
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def get_banned_tokens(prev_tokens, generated_ngrams, no_repeat_ngram_size):
    """Get tokens that would form a banned n-gram."""
    banned_tokens = []
    # Check if the last n-1 tokens can form a banned n-gram
    if tuple(prev_tokens) in generated_ngrams:
        # Get tokens that would create a banned n-gram
        banned_tokens = generated_ngrams[tuple(prev_tokens)]
    return banned_tokens

def update_ngrams(input_ids, generated_ngrams, no_repeat_ngram_size):
    """Update the n-gram tracking dictionary."""
    # Only update if we have enough tokens
    if len(input_ids) < no_repeat_ngram_size:
        return
    
    # Get the last n-gram
    current_ngram = tuple(input_ids[-(no_repeat_ngram_size):])
    
    # Store the last token of this n-gram as banned for the prefix (n-1 tokens)
    prefix = current_ngram[:-1]
    last_token = current_ngram[-1]
    
    if prefix not in generated_ngrams:
        generated_ngrams[prefix] = []
    
    generated_ngrams[prefix].append(last_token)

def main():
    parser = argparse.ArgumentParser(description='Test the Msingi1 model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt', 
                        help='Path to the model checkpoint (default: checkpoints/best.pt)')
    parser.add_argument('--tokenizer', type=str, default='tokenizer/tokenizer.json', 
                        help='Path to the tokenizer (default: tokenizer/tokenizer.json)')
    parser.add_argument('--prompt', type=str, default='Habari ya leo ni', 
                        help='Prompt to generate from')
    parser.add_argument('--max_length', type=int, default=50, 
                        help='Maximum length to generate (default: 50)')
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=40, 
                        help='Top-k filtering parameter (default: 40)')
    parser.add_argument('--top_p', type=float, default=0.95, 
                        help='Nucleus sampling parameter (default: 0.95)')
    parser.add_argument('--repetition_penalty', type=float, default=1.3, 
                        help='Penalty for repeating tokens (default: 1.3)')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3, 
                        help='Size of n-grams to prevent repetition (default: 3)')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to run on (default: cpu)')
    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer file not found at {args.tokenizer}")
        return

    # Load the tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer loaded with vocabulary size: {tokenizer.get_vocab_size()}")

    # Load the model
    model = load_model(args.checkpoint, device=args.device)
    
    # Generate text
    print(f"\nGenerating text with:\n - Temperature: {args.temperature}\n - Top-k: {args.top_k}\n - Top-p: {args.top_p}\n - Repetition penalty: {args.repetition_penalty}\n - No-repeat n-gram size: {args.no_repeat_ngram_size}\n - Max length: {args.max_length}")
    generated_text = generate_text(
        model, 
        tokenizer, 
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )
    
    print(f"\nPrompt: {args.prompt}")
    print(f"\nGenerated text:\n{'-' * 40}\n{generated_text}\n{'-' * 40}")

if __name__ == '__main__':
    main()
