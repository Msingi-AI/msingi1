import os
import sys
import torch
import argparse
import json
import time
import numpy as np
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from model_v2 import Msingi2, Msingi2Config

# Set console to UTF-8 mode for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Generate text with Msingi2 model")
    parser.add_argument("--model-path", type=str, default="best_model", 
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
    parser.add_argument("--presence-penalty", type=float, default=0.0,
                        help="Presence penalty (0.0 = no penalty)")
    parser.add_argument("--frequency-penalty", type=float, default=0.0,
                        help="Frequency penalty (0.0 = no penalty)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of text samples to generate")
    parser.add_argument("--guided-generation", action="store_true",
                        help="Use guided generation to improve coherence")
    parser.add_argument("--topic-guidance", type=str, default="",
                        help="Topic to guide generation towards (used with --guided-generation)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--slow-generation", action="store_true",
                        help="Generate text slowly, showing each token as it's generated")
    
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
        n_layer=config_dict.get('n_layer', 18),
        n_head=config_dict.get('n_head', 16),
        n_embd=config_dict.get('n_embd', 768),
        dropout=config_dict.get('dropout', 0.15),
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
    
    # Define guided generation function
    def guided_generation(input_ids, topic=None):
        """Generate text with guidance to improve coherence"""
        # Initialize with input prompt
        current_ids = input_ids.clone()
        all_tokens = []
        
        # Prepare topic guidance if provided
        topic_guidance = None
        if topic and topic.strip():
            # Tokenize the topic and get its embedding
            topic_ids = tokenizer.encode(topic, return_tensors="pt").to(args.device)
            with torch.no_grad():
                topic_logits, _ = model(topic_ids)
                topic_embedding = torch.mean(topic_logits, dim=1)
        
        # Generate tokens one by one
        for _ in range(args.max_tokens):
            # Crop sequence if too long
            if current_ids.size(1) > model.config.block_size:
                current_ids = current_ids[:, -model.config.block_size:]
            
            # Get model prediction
            with torch.no_grad():
                logits, _ = model(current_ids)
                next_token_logits = logits[:, -1, :].clone()
            
            # Apply temperature
            next_token_logits = next_token_logits / args.temperature
            
            # Apply penalties
            if args.repetition_penalty > 1.0:
                for i in range(current_ids.shape[0]):
                    for token_id in set(current_ids[i].tolist()):
                        next_token_logits[i, token_id] /= args.repetition_penalty
            
            # Apply presence and frequency penalties (similar to OpenAI API)
            if args.presence_penalty > 0 or args.frequency_penalty > 0:
                for i in range(current_ids.shape[0]):
                    for token_id in set(current_ids[i].tolist()):
                        # Count token occurrences
                        frequency = current_ids[i].tolist().count(token_id)
                        # Apply penalties
                        if args.presence_penalty > 0:
                            next_token_logits[i, token_id] -= args.presence_penalty
                        if args.frequency_penalty > 0:
                            next_token_logits[i, token_id] -= frequency * args.frequency_penalty
            
            # Apply topic guidance if provided
            if topic_guidance is not None:
                # Get embeddings for potential next tokens
                potential_tokens = torch.topk(next_token_logits, k=100).indices[0]
                token_scores = []
                
                for token in potential_tokens:
                    # Create potential sequence with this token
                    potential_seq = torch.cat([current_ids, token.unsqueeze(0).unsqueeze(0)], dim=1)
                    # Get embedding
                    with torch.no_grad():
                        potential_logits, _ = model(potential_seq)
                        potential_embedding = torch.mean(potential_logits, dim=1)
                    # Calculate similarity with topic
                    similarity = torch.cosine_similarity(potential_embedding, topic_embedding, dim=1)
                    token_scores.append((token.item(), similarity.item()))
                
                # Boost scores of tokens that maintain topic coherence
                for token, score in token_scores:
                    next_token_logits[0, token] += score * 2.0  # Adjust weight as needed
            
            # Apply top-k filtering
            if args.top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, args.top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if args.top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > args.top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            all_tokens.append(next_token.item())
            
            # For slow generation mode, show the token as it's generated
            if args.slow_generation:
                token_text = tokenizer.decode([next_token.item()])
                print(token_text, end='', flush=True)
                time.sleep(0.05)  # Adjust speed as needed
        
        return current_ids, all_tokens
    
    # Interactive mode function
    def interactive_mode():
        print("\n===== INTERACTIVE MODE =====")
        print("Type your prompts below. Type 'exit' to quit.")
        print("Type 'settings' to adjust generation parameters.")
        
        # Use current args as settings
        settings = vars(args).copy()
        
        while True:
            # Get user input
            user_input = input("\nPrompt> ")
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'settings':
                print("\nCurrent settings:")
                for key, value in settings.items():
                    if key in ['temperature', 'top_k', 'top_p', 'repetition_penalty', 
                               'presence_penalty', 'frequency_penalty', 'max_tokens']:
                        print(f"{key} = {value}")
                
                # Allow user to change settings
                print("\nEnter new settings (parameter=value), or 'done' when finished:")
                while True:
                    setting_input = input("Setting> ")
                    if setting_input.lower() == 'done':
                        break
                    
                    try:
                        param, value = setting_input.split('=')
                        param = param.strip()
                        value = value.strip()
                        
                        if param in settings:
                            # Convert value to appropriate type
                            if param in ['temperature', 'top_p', 'repetition_penalty', 
                                         'presence_penalty', 'frequency_penalty']:
                                settings[param] = float(value)
                            elif param in ['top_k', 'max_tokens']:
                                settings[param] = int(value)
                            elif param in ['guided_generation']:
                                settings[param] = value.lower() == 'true'
                            print(f"Updated {param} to {settings[param]}")
                        else:
                            print(f"Unknown parameter: {param}")
                    except ValueError:
                        print("Invalid format. Use 'parameter=value'")
                
                continue
            
            if not user_input.strip():
                continue
            
            # Tokenize input
            input_ids = tokenizer.encode(user_input, return_tensors="pt").to(args.device)
            
            # Generate text based on current settings
            if settings['guided_generation']:
                print(f"\nGenerating with guidance...")
                output_ids, _ = guided_generation(input_ids, settings.get('topic_guidance', ''))
            else:
                print(f"\nGenerating...")
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=settings['max_tokens'],
                        temperature=settings['temperature'],
                        top_k=settings['top_k'] if settings['top_k'] > 0 else None,
                        top_p=settings['top_p'] if settings['top_p'] > 0 else None,
                        repetition_penalty=settings['repetition_penalty']
                    )
            
            # Decode and display
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"\nGenerated:\n{generated_text}")
    
    # Run interactive mode if requested
    if args.interactive:
        interactive_mode()
        return
    
    # Standard generation mode
    print("\n===== GENERATING TEXT =====")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Parameters: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, ")
    print(f"             repetition_penalty={args.repetition_penalty}, presence_penalty={args.presence_penalty}, frequency_penalty={args.frequency_penalty}")
    if args.guided_generation:
        print(f"Guided generation: enabled" + (f" (topic: {args.topic_guidance})" if args.topic_guidance else ""))
    print(f"Generating {args.num_samples} samples, each with max {args.max_tokens} tokens\n")
    
    for i in range(args.num_samples):
        print(f"\n----- Sample {i+1} -----")
        
        # Tokenize prompt
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
        
        # Generate with guidance if requested
        if args.guided_generation:
            output_ids, tokens = guided_generation(input_ids, args.topic_guidance)
        else:
            # Standard generation
            if args.slow_generation:
                # Generate tokens one by one for display
                current_ids = input_ids.clone()
                print(args.prompt, end='', flush=True)
                
                for _ in range(args.max_tokens):
                    # Get next token
                    with torch.no_grad():
                        next_token = model.generate(
                            current_ids,
                            max_new_tokens=1,
                            temperature=args.temperature,
                            top_k=args.top_k if args.top_k > 0 else None,
                            top_p=args.top_p if args.top_p > 0 else None,
                            repetition_penalty=args.repetition_penalty
                        )[:, -1:]
                    
                    # Add to sequence
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    
                    # Display token
                    token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                    print(token_text, end='', flush=True)
                    time.sleep(0.05)  # Adjust speed as needed
                
                output_ids = current_ids
                print()  # New line after generation
            else:
                # Generate all at once
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k if args.top_k > 0 else None,
                        top_p=args.top_p if args.top_p > 0 else None,
                        repetition_penalty=args.repetition_penalty
                    )
        
        # Decode if not already displayed
        if not args.slow_generation:
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(generated_text)
    
    print("\n===== GENERATION COMPLETE =====")

if __name__ == "__main__":
    main()
