import os
import sys
import torch
import argparse
import numpy as np
import math
import time
import json
import random
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple

# Model and data processing
import wandb
from model import MsingiConfig, Msingi1
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
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
        tokenized_data_dir: str = "data/tokenized",
        tokenizer_path: str = "tokenizer/swahili_bpe_32000/tokenizer.json",
        
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
        seed: int = 42,
        fp16: bool = True,
        checkpoint_dir: str = "checkpoints",
        resume_from_checkpoint: Optional[str] = None,
        
        # Regularization
        dropout: float = 0.1,
        rep_penalty_alpha: float = 0.1,  # Repetition penalty strength
        
        # WandB settings
        use_wandb: bool = True,
        wandb_project: str = "msingi1",
        wandb_entity: str = None,
        wandb_run_name: str = f"msingi1-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    ):
        self.tokenized_data_dir = tokenized_data_dir
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
        
        self.seed = seed
        self.fp16 = fp16
        self.checkpoint_dir = checkpoint_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        
        self.dropout = dropout
        self.rep_penalty_alpha = rep_penalty_alpha
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name

class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized data stored in .pt files"""
    def __init__(self, data_dir, split="train"):
        self.data_files = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.pt")))
        if not self.data_files:
            raise ValueError(f"No tokenized data files found in {data_dir} for split {split}")
        print(f"Found {len(self.data_files)} {split} data files")
        
        # Load metadata if available
        metadata_path = os.path.join(data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                print(f"Loaded metadata: {self.metadata}")
        else:
            self.metadata = None
            
        # Cache for loaded batches
        self.cache = {}
        self.cache_size = 100  # Number of batches to keep in memory
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        # Load batch from file
        batch = torch.load(self.data_files[idx])
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove a random item from cache to prevent sequential bias
            keys = list(self.cache.keys())
            del self.cache[random.choice(keys)]
        
        self.cache[idx] = batch
        return batch

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
    penalty = 0.0
    
    # Only apply to sequences longer than 32 tokens
    if seq_len > 32:
        for i in range(batch_size):
            # Get non-padding tokens
            seq = labels[i]
            seq = seq[seq != -100]
            
            if len(seq) > 32:
                # Count token repetitions in the sequence
                unique, counts = torch.unique(seq, return_counts=True)
                # Penalize tokens that appear more than once
                rep_counts = torch.clamp(counts - 1, min=0)
                penalty += torch.sum(rep_counts.float()) / len(seq)
    
    penalty = penalty / batch_size
    return ce_loss + rep_penalty_alpha * penalty

def train_epoch(
    model, 
    train_loader, 
    optimizer, 
    scheduler, 
    scaler, 
    device, 
    config, 
    epoch, 
    global_step
):
    """Train for one epoch"""
    model.train()
    train_losses = []
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch_data in progress_bar:
        # Move batch to device
        input_ids = batch_data.to(device)
        
        # Create labels (shift input_ids right)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token prediction
        
        # Forward pass with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        
        for micro_batch_idx in range(0, input_ids.size(0), config.batch_size):
            micro_input_ids = input_ids[micro_batch_idx:micro_batch_idx + config.batch_size]
            micro_labels = labels[micro_batch_idx:micro_batch_idx + config.batch_size]
            
            # Skip empty micro batches at the end
            if micro_input_ids.size(0) == 0:
                continue
            
            # Forward pass with mixed precision
            if config.fp16 and scaler is not None:
                with autocast():
                    outputs = model(micro_input_ids)
                    loss = compute_loss(outputs, micro_labels, config.rep_penalty_alpha)
                    loss = loss / config.grad_accum_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
            else:
                outputs = model(micro_input_ids)
                loss = compute_loss(outputs, micro_labels, config.rep_penalty_alpha)
                loss = loss / config.grad_accum_steps
                loss.backward()
            
            train_losses.append(loss.item() * config.grad_accum_steps)
        
        # Gradient clipping and optimizer step
        if config.fp16 and scaler is not None:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        
        scheduler.step()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': train_losses[-1],
            'lr': scheduler.get_last_lr()[0]
        })
        
        global_step += 1
        
        # Log to wandb
        if config.use_wandb and global_step % 10 == 0:
            wandb.log({
                'train/loss': train_losses[-1],
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/epoch': epoch + batch_idx / len(train_loader),
                'train/global_step': global_step,
            })
        
        # Evaluate and save periodically
        if global_step % config.eval_interval == 0:
            eval_loss = evaluate(model, valid_loader, device, config)
            model.train()
            
            print(f"\nStep {global_step} | Eval loss: {eval_loss:.4f}")
            
            if config.use_wandb:
                wandb.log({
                    'eval/loss': eval_loss,
                    'eval/perplexity': math.exp(eval_loss),
                    'eval/global_step': global_step,
                })
        
        if global_step % config.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, config)
    
    return global_step, np.mean(train_losses)

def evaluate(model, eval_loader, device, config):
    """Evaluate model on validation set"""
    model.eval()
    eval_losses = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            if i >= config.eval_iters:
                break
                
            # Move batch to device
            input_ids = batch_data.to(device)
            
            # Create labels (shift input_ids right)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100
            
            # Forward pass
            outputs = model(input_ids)
            loss = compute_loss(outputs, labels)
            eval_losses.append(loss.item())
    
    return np.mean(eval_losses)

def save_checkpoint(model, optimizer, scheduler, global_step, config):
    """Save model checkpoint"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, f"msingi1_step_{global_step}.pt")
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
        'config': {
            'model': model.config.__dict__,
            'training': config.__dict__
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Also save the model in Hugging Face format
    model_dir = os.path.join(config.checkpoint_dir, f"msingi1_step_{global_step}")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    print(f"Saved model to {model_dir}")
    
    if config.use_wandb:
        wandb.save(checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    global_step = checkpoint['global_step']
    
    return global_step

def main(training_config, model_config):
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
    
    # Load tokenizer
    print(f"Loading tokenizer from {training_config.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=training_config.tokenizer_path)
    
    # Update model config with tokenizer vocab size
    if hasattr(tokenizer, 'vocab_size'):
        model_config.vocab_size = tokenizer.vocab_size
        print(f"Updated model vocab size to match tokenizer: {model_config.vocab_size}")
    
    # Prepare datasets
    print("Loading pre-tokenized datasets...")
    global train_loader, valid_loader
    
    train_dataset = PreTokenizedDataset(training_config.tokenized_data_dir, split="train")
    valid_dataset = PreTokenizedDataset(training_config.tokenized_data_dir, split="valid")
    
    # Create data loaders - note that we don't need to shuffle since the batches are pre-tokenized
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Each file is already a batch
        shuffle=True,  # Shuffle the order of batch files
        num_workers=2,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,  # Each file is already a batch
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # Calculate training steps and set up scheduler
    # For pre-tokenized data, each batch file is one step
    steps_per_epoch = len(train_loader)
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
    
    # Resume from checkpoint if specified
    global_step = 0
    if training_config.resume_from_checkpoint:
        global_step = load_checkpoint(
            training_config.resume_from_checkpoint,
            model, optimizer, scheduler
        )
    
    # Training loop
    print("Starting training...")
    for epoch in range(training_config.num_epochs):
        print(f"\nEpoch {epoch+1}/{training_config.num_epochs}")
        
        global_step, train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            scaler, device, training_config, epoch, global_step
        )
        
        print(f"Epoch {epoch+1} completed | Train loss: {train_loss:.4f}")
        
        # Evaluate at the end of each epoch
        eval_loss = evaluate(model, valid_loader, device, training_config)
        print(f"Epoch {epoch+1} | Eval loss: {eval_loss:.4f} | Perplexity: {math.exp(eval_loss):.2f}")
        
        if training_config.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch/train_loss': train_loss,
                'epoch/eval_loss': eval_loss,
                'epoch/perplexity': math.exp(eval_loss)
            })
        
        # Save checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, scheduler, global_step, training_config)
    
    print("Training completed!")
    
    # Final evaluation
    final_eval_loss = evaluate(model, valid_loader, device, training_config)
    print(f"Final eval loss: {final_eval_loss:.4f} | Perplexity: {math.exp(final_eval_loss):.2f}")
    
    # Save final model
    final_model_dir = os.path.join(training_config.checkpoint_dir, "msingi1_final")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    print(f"Saved final model to {final_model_dir}")
    
    if training_config.use_wandb:
        wandb.log({
            'final/eval_loss': final_eval_loss,
            'final/perplexity': math.exp(final_eval_loss)
        })
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Msingi1 Swahili language model")
    
    # Data arguments
    parser.add_argument("--tokenized-data-dir", type=str, default="data/tokenized",
                        help="Directory containing pre-tokenized data")
    parser.add_argument("--tokenizer-path", type=str, 
                        default="tokenizer/swahili_bpe_32000/tokenizer.json",
                        help="Path to tokenizer file")
    
    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--sequence-length", type=int, default=1024,
                        help="Maximum sequence length")
    
    # Optimization arguments
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="Warmup ratio")
    
    # Model arguments
    parser.add_argument("--n-layer", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=768,
                        help="Embedding dimension")
    
    # Technical arguments
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Resume training from checkpoint")
    
    # WandB arguments
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="msingi1",
                        help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="WandB entity name")
    parser.add_argument("--wandb-run-name", type=str, 
                        default=f"msingi1-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        help="WandB run name")
    
    args = parser.parse_args()
    
    # Create training config
    training_config = TrainingConfig(
        tokenized_data_dir=args.tokenized_data_dir,
        tokenizer_path=args.tokenizer_path,
        
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        sequence_length=args.sequence_length,
        
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        
        fp16=args.fp16,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )
    
    # Create model config
    model_config = MsingiConfig(
        vocab_size=32000,  # Will be updated based on tokenizer
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    
    main(training_config, model_config)
