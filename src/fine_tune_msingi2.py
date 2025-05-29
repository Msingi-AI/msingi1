import os
import sys
import torch
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from model_v2 import Msingi2, Msingi2Config

# Set console to UTF-8 mode for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class ConversationalDataset(Dataset):
    """Dataset for fine-tuning on conversational data"""
    
    def __init__(self, data_path, tokenizer, block_size=1024):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load conversations from file
        with open(data_path, 'r', encoding='utf-8') as f:
            conversations = f.read().split('\n\n')  # Assuming conversations are separated by double newlines
        
        # Tokenize all conversations
        self.examples = []
        for conv in tqdm(conversations, desc="Tokenizing conversations"):
            if not conv.strip():
                continue
                
            # Add special tokens to mark conversation boundaries
            conv_text = f"<s>{conv}</s>"
            tokens = tokenizer.encode(conv_text)
            
            # Create examples of block_size length
            for i in range(0, len(tokens) - block_size + 1, block_size // 2):  # 50% overlap
                self.examples.append(tokens[i:i + block_size])
        
        print(f"Created {len(self.examples)} training examples from {len(conversations)} conversations")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.examples[idx], dtype=torch.long)
        return tokens

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Msingi2 model on conversational data")
    parser.add_argument("--model-path", type=str, default="msingi2_best_model", 
                        help="Path to the pretrained model directory")
    parser.add_argument("--tokenizer-path", type=str, 
                        default="tokenizer/swahili_unigram_32000/tokenizer.json",
                        help="Path to the tokenizer file")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to conversational data file")
    parser.add_argument("--output-dir", type=str, default="msingi2_ft_conversational",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps for learning rate scheduler")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Evaluate every X steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Check and add special tokens if needed
    for token_type, token in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            print(f"Adding {token_type}: {token}")
    
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
        dropout=0.1,  # Slightly higher dropout for fine-tuning
        bias=config_dict.get('bias', True),
        gradient_checkpointing=True  # Enable for memory efficiency
    )
    
    # Initialize model
    print(f"Initializing model with {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding size")
    model = Msingi2(config)
    
    # Load model weights
    model_path = os.path.join(args.model_path, "pytorch_model.bin")
    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    
    print(f"Model loaded successfully with {model.get_num_params():,} parameters")
    
    # Load dataset
    print(f"Loading conversational dataset from {args.data_path}")
    dataset = ConversationalDataset(args.data_path, tokenizer, block_size=config.block_size)
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    # Set up optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.01,
        learning_rate=args.lr,
        betas=(0.9, 0.95),
        device_type=args.device
    )
    
    # Set up scheduler
    total_steps = len(dataloader) * args.epochs // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    print(f"Starting fine-tuning for {args.epochs} epochs")
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(progress_bar):
            batch = batch.to(args.device)
            
            # Forward pass with mixed precision if enabled
            if args.fp16:
                with torch.cuda.amp.autocast():
                    logits, loss = model(batch, batch)
                    loss = loss / args.grad_accum_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % args.grad_accum_steps == 0 or step == len(dataloader) - 1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                # Standard forward and backward pass
                logits, loss = model(batch, batch)
                loss = loss / args.grad_accum_steps
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % args.grad_accum_steps == 0 or step == len(dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            # Update progress bar
            epoch_loss += loss.item() * args.grad_accum_steps
            progress_bar.set_postfix({"loss": loss.item() * args.grad_accum_steps})
            
            # Evaluation
            if global_step % args.eval_steps == 0:
                model.eval()
                eval_loss = 0
                eval_steps = 0
                
                # Sample a few batches for evaluation
                eval_batches = [next(iter(dataloader)) for _ in range(5)]
                
                with torch.no_grad():
                    for eval_batch in eval_batches:
                        eval_batch = eval_batch.to(args.device)
                        _, batch_loss = model(eval_batch, eval_batch)
                        eval_loss += batch_loss.item()
                        eval_steps += 1
                
                avg_eval_loss = eval_loss / eval_steps
                print(f"\nStep {global_step} - Eval Loss: {avg_eval_loss:.4f}")
                
                # Save best model
                if avg_eval_loss < best_loss:
                    best_loss = avg_eval_loss
                    print(f"New best model with loss: {best_loss:.4f}")
                    
                    # Save model
                    best_model_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(best_model_dir, exist_ok=True)
                    
                    # Save model weights
                    torch.save(model.state_dict(), os.path.join(best_model_dir, "pytorch_model.bin"))
                    
                    # Save model config
                    with open(os.path.join(best_model_dir, "model_config.json"), 'w') as f:
                        json.dump(config.__dict__, f)
                
                model.train()
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model weights
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
                
                # Save optimizer and scheduler states
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                
                # Save model config
                with open(os.path.join(checkpoint_dir, "model_config.json"), 'w') as f:
                    json.dump(config.__dict__, f)
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save epoch checkpoint
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save model weights
        torch.save(model.state_dict(), os.path.join(epoch_dir, "pytorch_model.bin"))
        
        # Save model config
        with open(os.path.join(epoch_dir, "model_config.json"), 'w') as f:
            json.dump(config.__dict__, f)
    
    print("Fine-tuning complete!")
    print(f"Best model saved to {os.path.join(args.output_dir, 'best_model')}")

if __name__ == "__main__":
    main()
