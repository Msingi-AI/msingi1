import os
import torch
from tokenizers import Tokenizer
from model import MsingiConfig, Msingi1

def load_model(checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load the trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path} to {device}")
    try:
        # Try different loading methods
        try:
            # Method 1: Direct load
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except:
            # Method 2: Load with pickle
            import pickle
            checkpoint = torch.load(checkpoint_path, map_location='cpu', pickle_module=pickle)
        
        print("Checkpoint loaded successfully")
        print("Checkpoint keys:", checkpoint.keys())
        
        # Get model config
        config = checkpoint['config']
        print("Model config:", config.__dict__)
        
        # Initialize model on CPU first
        print("Initializing model...")
        model = Msingi1(config)
        
        # Load state dict
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if it exists (from DataParallel/DistributedDataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        # Move to target device
        model = model.to(device)
        model.eval()
        print("Model initialized and loaded successfully")
        
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        print(f"Checkpoint file size: {os.path.getsize(checkpoint_path)} bytes")
        raise

def generate_text(
    model: Msingi1,
    tokenizer: Tokenizer,
    prompt: str,
    max_length: int = 30,
    temperature: float = 0.1,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Generate text from a prompt."""
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = [tokenizer.token_to_id("<s>")] + encoded.ids
    
    # Convert to tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits for next token
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
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
            
            # Stop if we generate EOS token
            if next_token.item() == tokenizer.token_to_id("</s>"):
                break
                
            # Append to input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # Decode generated text
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    # Clean up special tokens
    generated_text = generated_text.replace("<s>", "").replace("</s>", "").strip()
    
    return generated_text

def main():
    # Load tokenizer
    tokenizer_path = "tokenizer/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("Tokenizer loaded successfully")
    
    # Load model
    checkpoint_path = "checkpoints/best .pt"  # Using the checkpoint with space in filename
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model = load_model(checkpoint_path)
    print("Model loaded successfully!")
    
    # Interactive generation loop
    print("\nEnter prompts for text generation. Type 'quit' to exit.")
    print("You can also use these special commands:")
    print("  temp=X : Set temperature (0.1-2.0, default 0.7)")
    print("  len=X  : Set max length (10-500, default 100)")
    print("  top_k=X: Set top-k (1-100, default 50)")
    print("  top_p=X: Set top-p (0.1-1.0, default 0.9)")
    
    temperature = 0.7
    max_length = 100
    top_k = 50
    top_p = 0.9
    
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
            
            if prompt.lower() == 'quit':
                break
                
            # Handle parameter adjustments
            if prompt.startswith(('temp=', 'len=', 'top_k=', 'top_p=')):
                try:
                    param, value = prompt.split('=')
                    value = float(value)
                    if param == 'temp':
                        temperature = max(0.1, min(2.0, value))
                        print(f"Temperature set to {temperature}")
                    elif param == 'len':
                        max_length = max(10, min(500, int(value)))
                        print(f"Max length set to {max_length}")
                    elif param == 'top_k':
                        top_k = max(1, min(100, int(value)))
                        print(f"Top-k set to {top_k}")
                    elif param == 'top_p':
                        top_p = max(0.1, min(1.0, value))
                        print(f"Top-p set to {top_p}")
                except ValueError:
                    print("Invalid value. Please try again.")
                continue
            
            if not prompt:
                continue
            
            print("\nGenerating...")
            generated_text = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            print("\nGenerated text:")
            print("-" * 40)
            print(generated_text)
            print("-" * 40)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
