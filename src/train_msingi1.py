import os
import sys
import torch
import argparse
import numpy as np
import math
import time
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple
import wandb
from model import MsingiConfig, Msingi1
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Try to import wandb for logging, but don't fail if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

# Set PyTorch memory allocation and GPU optimization configs
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Better performance with async CUDA
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable CUDA device synchronization attributes

# Set torch backends for optimal performance
torch.backends.cudnn.benchmark = True  # Use cuDNN auto-tuner
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+ GPUs
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN

class TrainingConfig:
    """Configuration for Msingi1 training"""
    def __init__(
        self,
        # Data settings
        train_file: str = "data/train.txt",
        valid_file: str = "data/valid.txt",
        tokenizer_type: str = "bpe",  # Options: 'bpe' or 'unigram'
        tokenizer_path: str = None,  # If None, will be set based on tokenizer_type
        
        # Training settings
        num_epochs: int = 3,
        batch_size: int = 16,
        grad_accum_steps: int = 4,  # Effective batch size = 64
        sequence_length: int = 1024,
        
        # Optimization settings
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.03,
        min_lr_ratio: float = 0.1,  # min_lr = learning_rate * min_lr_ratio
        
        # Evaluation and saving
        eval_interval: int = 500,
        eval_iters: int = 100,
        save_interval: int = 1000,
        
        # Technical settings
        fp16: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        seed: int = 42,
        
        # Wandb settings
        use_wandb: bool = False,
        wandb_project: str = "msingi1",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        
        # Repetition penalty for loss
        rep_penalty_alpha: float = 0.1,
    ):
        self.train_file = train_file
        self.valid_file = valid_file
        self.tokenizer_path = tokenizer_path
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.sequence_length = sequence_length
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.save_interval = save_interval
        
        self.fp16 = fp16
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.seed = seed
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name or f"msingi1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self.rep_penalty_alpha = rep_penalty_alpha
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

class SwahiliDataset(Dataset):
    """Dataset for Swahili language modeling with efficient chunking and overlap"""
    def __init__(self, file_path: str, tokenizer_path: str, max_length: int, stride: Optional[int] = None):
        self.max_length = max_length
        self.stride = stride or (max_length // 2)  # Default to 50% overlap
        
        # Load the trained tokenizer
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        
        # Load and process text
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize the entire text at once (more efficient)
        print("Tokenizing text...")
        encoded = self.tokenizer.encode(text)
        self.tokens = encoded
        
        # Create chunks with overlap
        self.examples = []
        for i in range(0, len(self.tokens) - max_length + 1, self.stride):
            self.examples.append(self.tokens[i:i + max_length])
        
        print(f"Created {len(self.examples):,} examples with sequence length {max_length} and stride {self.stride}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Create input_ids and labels for causal LM
        input_ids = tokens[:-1]  # All tokens except last
        labels = tokens[1:]      # All tokens except first
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def compute_loss(logits, labels, rep_penalty_alpha=0.0):
    """Compute cross entropy loss with optional repetition penalty"""
    # Standard cross entropy loss
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Skip repetition penalty if alpha is 0
    if rep_penalty_alpha <= 0:
        return ce_loss
    
    # Compute repetition penalty
    batch_size, seq_len = labels.shape
    rep_penalty = 0
    
    # For each sequence in the batch
    for i in range(batch_size):
        # Get non-padded tokens
        seq = labels[i][labels[i] != -100]
        if len(seq) > 2:
            # Count repeated tokens in windows of 3
            for j in range(len(seq)-2):
                window = seq[j:j+3]
                unique_tokens = torch.unique(window)
                rep_penalty += 1 - (len(unique_tokens) / len(window))
    
    rep_penalty = rep_penalty / (batch_size * seq_len)
    
    # Combine losses
    total_loss = ce_loss + rep_penalty_alpha * rep_penalty
    return total_loss

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def evaluate(model, dataloader, config, device, fp16=False):
    """Evaluate model on validation data"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= config.eval_iters:
                break
                
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            if fp16:
                with autocast():
                    logits = model(input_ids)
                    loss = compute_loss(logits, labels, config.rep_penalty_alpha)
            else:
                logits = model(input_ids)
                loss = compute_loss(logits, labels, config.rep_penalty_alpha)
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0) * input_ids.size(1)
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    model.train()
    return {"loss": avg_loss, "perplexity": perplexity}

def save_checkpoint(model, optimizer, scheduler, scaler, config, epoch, global_step, step_in_epoch, path):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'epoch': epoch,
        'global_step': global_step,
        'step_in_epoch': step_in_epoch,
        'config': config.__dict__
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, model, optimizer, scheduler=None, scaler=None, device=None):
    """Load model checkpoint"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    step_in_epoch = checkpoint.get('step_in_epoch', 0)
    
    return epoch, global_step, step_in_epoch

def train(model_config, training_config):
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    random.seed(training_config.seed)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = Msingi1(model_config)
    model.to(device)
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    tokenizer_path = "tokenizer/swahili_unigram_32000/tokenizer.json"
    print(f"Loading Unigram tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Update model config with tokenizer vocab size
    if hasattr(tokenizer, 'vocab_size'):
        model_config.vocab_size = tokenizer.vocab_size
        print(f"Updated model vocab size to match tokenizer: {model_config.vocab_size}")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = SwahiliDataset(
        training_config.train_file,
        tokenizer_path,  # Use the Unigram tokenizer path
        training_config.sequence_length
    )
    
    valid_dataset = SwahiliDataset(
        training_config.valid_file,
        tokenizer_path,  # Use the Unigram tokenizer path
        training_config.sequence_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,  # Increased for dedicated GPU
        pin_memory=True,
        prefetch_factor=2  # Prefetch batches for better GPU utilization
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4,  # Increased for dedicated GPU
        pin_memory=True,
        prefetch_factor=2  # Prefetch batches for better GPU utilization
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # Calculate training steps and set up scheduler
    steps_per_epoch = len(train_loader) // training_config.grad_accum_steps
    total_training_steps = steps_per_epoch * training_config.num_epochs
    warmup_steps = int(total_training_steps * training_config.warmup_ratio)
    
    print(f"Training for {training_config.num_epochs} epochs, {steps_per_epoch} steps per epoch")
    print(f"Total training steps: {total_training_steps}, warmup steps: {warmup_steps}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    
    # Initialize mixed precision training
    scaler = None
    if training_config.fp16 and torch.cuda.is_available():
        print("Using mixed precision training")
        scaler = GradScaler()
    
    # Initialize WandB if available
    if training_config.use_wandb:
        wandb.init(
            project=training_config.wandb_project,
            entity=training_config.wandb_entity,
            name=training_config.wandb_run_name,
            config={
                **model_config.__dict__,
                **training_config.__dict__
            }
        )
        wandb.watch(model)
    
    # Load checkpoint if it exists
    start_epoch = 0
    global_step = 0
    resume_step = 0
    best_val_loss = float('inf')
    
    checkpoint_path = os.path.join(training_config.checkpoint_dir, 'latest.pt')
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        try:
            start_epoch, global_step, resume_step = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler, device
            )
            print(f'Resuming from epoch {start_epoch}, step {resume_step} (global step {global_step})')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    
    # Training loop
    model.train()
    accumulated_loss = 0
    accumulated_tokens = 0
    
    print("Starting training...")
    for epoch in range(start_epoch, training_config.num_epochs):
        epoch_start_time = time.time()
        
        # Skip steps if resuming
        if epoch == start_epoch and resume_step > 0:
            print(f"Skipping {resume_step} steps in first epoch")
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{training_config.num_epochs}")
        
        for step, batch in progress_bar:
            # Skip steps if resuming
            if epoch == start_epoch and step < resume_step:
                continue
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision if enabled
            if training_config.fp16:
                with autocast():
                    logits = model(input_ids)
                    loss = compute_loss(logits, labels, training_config.rep_penalty_alpha)
                    loss = loss / training_config.grad_accum_steps
            else:
                logits = model(input_ids)
                loss = compute_loss(logits, labels, training_config.rep_penalty_alpha)
                loss = loss / training_config.grad_accum_steps
            
            # Backward pass with mixed precision if enabled
            if training_config.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update statistics
            accumulated_loss += loss.item() * training_config.grad_accum_steps
            accumulated_tokens += input_ids.size(0) * input_ids.size(1)
            
            # Gradient accumulation
            if (step + 1) % training_config.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                # Gradient clipping
                if training_config.fp16:
                    scaler.unscale_(optimizer)
                
                clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                
                # Update parameters
                if training_config.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Increment global step
                global_step += 1
                
                # Log training progress
                if global_step % training_config.log_interval == 0:
                    # Calculate perplexity
                    avg_loss = accumulated_loss / accumulated_tokens
                    perplexity = math.exp(avg_loss)
                    
                    # Get current learning rate
                    lr = scheduler.get_last_lr()[0]
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'ppl': f"{perplexity:.2f}",
                        'lr': f"{lr:.6f}"
                    })
                    
                    # Log to wandb
                    if training_config.use_wandb:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/perplexity': perplexity,
                            'train/learning_rate': lr,
                            'train/epoch': epoch + step / len(train_loader),
                            'train/global_step': global_step
                        })
                    
                    # Reset accumulated metrics
                    accumulated_loss = 0
                    accumulated_tokens = 0
                
                # Evaluate on validation set
                if global_step % training_config.eval_interval == 0:
                    print("\nEvaluating on validation set...")
                    val_metrics = evaluate(model, valid_loader, training_config, device, training_config.fp16)
                    
                    print(f"Validation loss: {val_metrics['loss']:.4f}, perplexity: {val_metrics['perplexity']:.2f}")
                    
                    # Log to wandb
                    if training_config.use_wandb:
                        wandb.log({
                            'val/loss': val_metrics['loss'],
                            'val/perplexity': val_metrics['perplexity'],
                            'val/epoch': epoch + step / len(train_loader),
                            'val/global_step': global_step
                        })
                    
                    # Save best model
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        save_checkpoint(
                            model, optimizer, scheduler, scaler, training_config,
                            epoch, global_step, step,
                            os.path.join(training_config.checkpoint_dir, 'best.pt')
                        )
                        print(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save regular checkpoint
                if global_step % training_config.save_interval == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, training_config,
                        epoch, global_step, step,
                        os.path.join(training_config.checkpoint_dir, f'step_{global_step}.pt')
                    )
                    
                    # Also save as latest
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, training_config,
                        epoch, global_step, step,
                        os.path.join(training_config.checkpoint_dir, 'latest.pt')
                    )
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler, training_config,
            epoch + 1, global_step, 0,  # Reset step_in_epoch to 0 for new epoch
            os.path.join(training_config.checkpoint_dir, f'epoch_{epoch+1}.pt')
        )
        
        # Also save as latest
        save_checkpoint(
            model, optimizer, scheduler, scaler, training_config,
            epoch + 1, global_step, 0,
            os.path.join(training_config.checkpoint_dir, 'latest.pt')
        )
    
    # End of training
    print("Training completed!")
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, scaler, training_config,
        training_config.num_epochs, global_step, 0,
        os.path.join(training_config.checkpoint_dir, 'final.pt')
    )
    
    # Close wandb
    if training_config.use_wandb:
        wandb.finish()
    
    return model

def generate_sample_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=40, top_p=0.9):
    """Generate sample text from the model"""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
    
    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Train Msingi1 Swahili language model")
    
    # Data arguments
    parser.add_argument("--train-file", type=str, default="data/train.txt", help="Path to training file")
    parser.add_argument("--valid-file", type=str, default="data/valid.txt", help="Path to validation file")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--seq-length", type=int, default=1024, help="Sequence length")
    
    # Optimization arguments
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    
    # Technical arguments
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # WandB arguments
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="msingi1", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    
    args = parser.parse_args()
    
    # Create model config
    model_config = MsingiConfig(
        vocab_size=32000,  # Will be updated based on tokenizer
        max_position_embeddings=args.seq_length,
        n_layer=12,
        n_head=12,
        n_embd=768,
        intermediate_size=3072,
        dropout=0.1,
        rotary_emb=True,
        gradient_checkpointing=True
    )
    
    # Create training config
    training_config = TrainingConfig(
        train_file=args.train_file,
        valid_file=args.valid_file,
        
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        sequence_length=args.seq_length,
        
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        
        fp16=not args.no_fp16,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name
    )
    
    # Train model
    model = train(model_config, training_config)
    
    # Save final model and config
    model_dir = os.path.join(args.checkpoint_dir, "final_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(model_dir)
    
    # Save configs
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config.__dict__, f, indent=2)
    
    with open(os.path.join(model_dir, "training_config.json"), "w") as f:
        json.dump(training_config.__dict__, f, indent=2)
    
    print(f"Final model saved to {model_dir}")

if __name__ == "__main__":
    main()
