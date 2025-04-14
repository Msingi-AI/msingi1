import os
import sys
import torch
import wandb
from model import MsingiConfig, Msingi1
from data_processor import load_dataset, prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from pathlib import Path
import math
from typing import Optional, List
import numpy as np
from tokenizers import Tokenizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from dataclasses import dataclass
from google.colab import drive
import argparse

# Mount Google Drive
try:
    drive.mount('/content/drive')
except:
    print("Warning: Could not mount Google Drive. Are you running in Colab?")

# Drive path for checkpoints
DRIVE_PATH = "/content/drive/MyDrive/msingi1"

def verify_drive_mount():
    """Verify Google Drive is mounted and paths exist"""
    if not os.path.exists("/content/drive"):
        raise RuntimeError("Google Drive is not mounted! Please run the notebook in Google Colab")
    
    if not os.path.exists(DRIVE_PATH):
        os.makedirs(DRIVE_PATH, exist_ok=True)
        print(f"Created msingi1 directory at {DRIVE_PATH}")
    
    checkpoint_dir = os.path.join(DRIVE_PATH, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoints directory at {checkpoint_dir}")
    
    # Test write permissions
    test_file = os.path.join(checkpoint_dir, 'test.txt')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("Successfully verified write permissions to checkpoint directory")
    except Exception as e:
        raise RuntimeError(f"Cannot write to checkpoint directory: {e}")

# Colab Drive path
DRIVE_PATH = "/content/drive/MyDrive/msingi1"

# Set PyTorch memory allocation config
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@dataclass
class TrainingConfig:
    num_epochs: int = 40         # 40 epochs
    batch_size: int = 8          # Batch size 8
    grad_accum_steps: int = 4    # Reduced to 4 as requested
    learning_rate: float = 5e-4  
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_iters: int = 2000    
    lr_decay_iters: int = 80000  
    min_lr: float = 3e-5
    eval_interval: int = 500
    log_interval: int = 10
    save_interval: int = 500
    fp16: bool = True
    sequence_length: int = 1024
    checkpoint_dir: str = "checkpoints"

class SwahiliDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer_path: str, max_length: int):
        # Load the trained tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Encode all texts
        self.examples = []
        for text in texts:
            # Encode and add BOS/EOS tokens
            encoded = self.tokenizer.encode(text)
            input_ids = [self.tokenizer.token_to_id("<s>")] + encoded.ids + [self.tokenizer.token_to_id("</s>")]
            
            # Create overlapping sequences of max_length
            for i in range(0, len(input_ids) - max_length + 1, max_length // 2):
                sequence = input_ids[i:i + max_length]
                if len(sequence) == max_length:
                    self.examples.append(sequence)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids = self.examples[idx][:-1]  # All tokens except last
        labels = self.examples[idx][1:]      # All tokens except first
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def compute_loss(logits, targets):
    """Compute loss efficiently for decoder-only model."""
    if logits.size(1) != 1:
        # During inference, we might get full sequence
        logits = logits[:, :-1, :]  # Remove last position
        targets = targets[:, 1:]     # Shift targets right
    else:
        # During training, we only get last token
        targets = targets[:, -1:]    # Only last token
    
    # Reshape for loss computation
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    
    # Compute loss
    loss = F.nll_loss(logits, targets)
    return loss

def train(model_config: MsingiConfig, train_texts: List[str], val_texts: Optional[List[str]] = None,
         training_config: Optional[TrainingConfig] = None, tokenizer_path: str = "tokenizer/tokenizer.json"):
    if training_config is None:
        training_config = TrainingConfig()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = Msingi1(model_config)
    model.to(device)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # Initialize mixed precision training
    scaler = None
    if training_config.fp16 and torch.cuda.is_available():
        print("Using mixed precision training")
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    
    # Load checkpoint if it exists
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_path = os.path.join(training_config.checkpoint_dir, 'latest.pt')
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f'Resuming from epoch {start_epoch}')
    
    # Create datasets
    train_dataset = SwahiliDataset(train_texts, tokenizer_path, training_config.sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    
    if val_texts:
        val_dataset = SwahiliDataset(val_texts, tokenizer_path, training_config.sequence_length)
        val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * training_config.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_iters,
        num_training_steps=num_training_steps
    )
    
    # Initialize wandb if available
    use_wandb = 'wandb' in sys.modules
    if use_wandb:
        wandb.init(project="msingi1", config=vars(training_config))
    
    # Training loop
    global_step = 0
    for epoch in range(start_epoch, training_config.num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with autocast(enabled=training_config.fp16 and torch.cuda.is_available()):
                logits = model(input_ids)  # Model returns logits directly
                
                # Compute loss
                loss = compute_loss(logits, labels)
                loss = loss / training_config.grad_accum_steps
            
            # Backward pass with mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % training_config.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Save checkpoint
                if (step + 1) % training_config.save_interval == 0:
                    avg_loss = total_loss / (step + 1)
                    is_best = False
                    
                    # Check if this is the best model
                    if val_texts:
                        val_loss = evaluate(model, val_loader, model_config, device, training_config.fp16)
                        is_best = val_loss < best_val_loss
                        if is_best:
                            best_val_loss = val_loss
                    
                    save_checkpoint(
                        model, optimizer, scaler, model_config,
                        epoch, step, avg_loss, best_val_loss,
                        training_config, is_best
                    )
            
            total_loss += loss.item() * training_config.grad_accum_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (step + 1),
                'lr': optimizer.param_groups[0]['lr'],
            })
            global_step += 1
            
            # Evaluation
            if val_texts and global_step % training_config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, model_config, device, training_config.fp16)
                
                if use_wandb:
                    wandb.log({
                        'val_loss': val_loss,
                        'train_loss': total_loss / (step + 1),
                        'epoch': epoch,
                        'global_step': global_step,
                    })
                
                model.train()  # Back to training mode
        
        # Save checkpoint at end of each epoch
        try:
            # Ensure checkpoint directory exists
            os.makedirs(training_config.checkpoint_dir, exist_ok=True)
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader),
                'config': model_config,
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            # Save epoch checkpoint
            epoch_path = os.path.join(training_config.checkpoint_dir, f'epoch_{epoch+1}.pt')
            print(f'\nSaving epoch checkpoint to {epoch_path}')
            torch.save(checkpoint, epoch_path)
            
            # Update latest checkpoint
            latest_path = os.path.join(training_config.checkpoint_dir, 'latest.pt')
            torch.save(checkpoint, latest_path)
            
            # Save as best if it's the best loss so far
            current_loss = total_loss / len(train_loader)
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                best_path = os.path.join(training_config.checkpoint_dir, 'best.pt')
                print(f'New best loss: {current_loss:.4f}, saving to {best_path}')
                torch.save(checkpoint, best_path)
            
            print('Epoch checkpoint saved successfully')
        except Exception as e:
            print(f'\nERROR saving epoch checkpoint: {str(e)}')
            print(f'Checkpoint directory: {training_config.checkpoint_dir}')
            print(f'Directory exists: {os.path.exists(training_config.checkpoint_dir)}')
            print(f'Directory is writable: {os.access(training_config.checkpoint_dir, os.W_OK)}')
            raise
        
        # Log epoch metrics
        avg_loss = total_loss / len(train_loader)
        if use_wandb:
            wandb.log({
                'train_loss': avg_loss,
                'epoch': epoch,
                'global_step': global_step,
            })
        
        print(f"Epoch {epoch+1}/{training_config.num_epochs} - Average Loss: {avg_loss:.4f}")

def evaluate(model, val_loader, config, device, fp16=False):
    """Evaluate the model on validation data with optional mixed precision."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            with autocast(enabled=fp16 and torch.cuda.is_available()):
                logits = model(input_ids)
                
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            total_loss += loss.item() * labels.ne(-100).sum().item()
            total_tokens += labels.ne(-100).sum().item()
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

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
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def save_checkpoint(model, optimizer, scaler, model_config, epoch, step, loss, best_loss, training_config, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model_config,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Save latest checkpoint
    latest_path = os.path.join(training_config.checkpoint_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)
    print(f'\nSaved latest checkpoint to {latest_path}')
    
    # Save best checkpoint if this is the best loss
    if is_best:
        best_path = os.path.join(training_config.checkpoint_dir, 'best.pt')
        torch.save(checkpoint, best_path)
        print(f'New best loss! Saved checkpoint to {best_path}')

def clean_text(text: str) -> str:
    """Clean and preprocess text data."""
    import re
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers mixed with text (keep pure numbers)
    text = re.sub(r'\w*\d+\w*', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove mixed language (likely English) words - words with non-Swahili characters
    text = re.sub(r'\b\w*[qxcv]\w*\b', '', text, flags=re.IGNORECASE)  # Swahili doesn't use q,x,c,v
    
    # Normalize quotes and dashes
    text = re.sub(r'[''"""]', '"', text)
    text = re.sub(r'[–—−]', '-', text)
    
    # Remove standalone numbers and special characters
    text = re.sub(r'(?<!\w)\d+(?!\w)', '', text)
    text = re.sub(r'[^\w\s\.\,\;\:\"\'\-\?\/\!]', '', text)
    
    # Remove repeated punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def prepare_dataset(texts: List[str]) -> List[str]:
    """Prepare dataset by cleaning and filtering texts."""
    cleaned_texts = []
    min_length = 100  # Minimum text length to keep
    
    for text in texts:
        cleaned = clean_text(text)
        # Only keep substantial texts
        if len(cleaned.split()) >= min_length:
            cleaned_texts.append(cleaned)
    
    return cleaned_texts

def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Msingi1 model')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory containing the dataset')
    parser.add_argument('--save_every', type=int, default=500, help='Save checkpoint every N steps')
    parser.add_argument('--log_every', type=int, default=10, help='Log metrics every N steps')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer/tokenizer.json', help='Path to tokenizer file')
    args = parser.parse_args()
    
    # Initialize model config
    model_config = MsingiConfig()
    
    # Initialize training config with command line arguments
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.1,
        max_grad_norm=1.0,
        warmup_iters=2000,
        lr_decay_iters=80000,
        min_lr=3e-5,
        eval_interval=500,
        log_interval=args.log_every,
        save_interval=args.save_every,
        fp16=args.fp16,
        sequence_length=1024,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Load and preprocess data
    train_path = os.path.join(args.data_dir, "train.txt")
    val_path = os.path.join(args.data_dir, "valid.txt")
    
    print(f"Loading training data from {train_path}")
    train_texts = load_dataset(train_path)
    
    print(f"Loading validation data from {val_path}")
    val_texts = load_dataset(val_path)
    
    # Train model
    train(model_config, train_texts, val_texts, training_config, args.tokenizer_path)

if __name__ == "__main__":
    main()
