import torch
from model import Msingi1, MsingiConfig
from tokenizers import Tokenizer
import argparse

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.9):
    """Generate text from a prompt."""
    # Encode the prompt
    input_ids = tokenizer.encode(prompt).ids
    input_ids = [tokenizer.token_to_id("<s>")] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    
    # Move to the same device as model
    input_ids = input_ids.to(next(model.parameters()).device)
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
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
    
    # Decode the generated tokens
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    # Clean up the text (remove special tokens)
    generated_text = generated_text.replace("<s>", "").replace("</s>", "").strip()
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Test the Msingi1 model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer/tokenizer.json', help='Path to the tokenizer')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to generate from')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling parameter')
    args = parser.parse_args()

    # Load the tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # Initialize model config and model
    config = MsingiConfig(
        hidden_size=512,  # Match saved model dimensions
        num_hidden_layers=12,
        num_attention_heads=8,  # Adjusted to be compatible with hidden_size
        vocab_size=35523
    )
    model = Msingi1(config)

    # Load the model weights
    checkpoint = torch.load(args.model_path)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle key renaming
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('layers.') and '.attention.' in k:
            # Map attention keys
            if 'query' in k:
                k = k.replace('query', 'q_proj')
            elif 'key' in k:
                k = k.replace('key', 'k_proj')
            elif 'value' in k:
                k = k.replace('value', 'v_proj')
        elif k.startswith('layernorm'):
            k = k.replace('layernorm', 'ln_f')
        new_state_dict[k] = v
    
    # Load the state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Generate text
    generated_text = generate_text(
        model, 
        tokenizer, 
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print(f"\nPrompt: {args.prompt}")
    print(f"\nGenerated text:\n{generated_text}")

if __name__ == '__main__':
    main()
