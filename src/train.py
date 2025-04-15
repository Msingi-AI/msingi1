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
    num_epochs: int = 10  # Default 10 epochs
    batch_size: int = 4   # Using batch size 4
    grad_accum_steps: int = 16  # Adjusted to maintain effective batch size
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_iters: int = 1000
    lr_decay_iters: int = 20000
    min_lr: float = 3e-5
    eval_interval: int = 500
    eval_iters: int = 100
    save_interval: int = 500  # Save twice per epoch (674 steps total)
    fp16: bool = True
    sequence_length: int = 1024  # Keeping reduced sequence length for memory efficiency
    checkpoint_dir: str = os.path.join(DRIVE_PATH, 'checkpoints')  # Save to Drive

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

def compute_loss_with_penalty(logits, labels, alpha=0.1):
    """Compute cross entropy loss with a repetition penalty."""
    # Standard cross entropy loss
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Compute repetition penalty
    batch_size, seq_len = labels.shape
    rep_penalty = 0
    
    # For each sequence in the batch
    for i in range(batch_size):
        # Get non-padded tokens
        seq = labels[i][labels[i] != -100]
        if len(seq) > 1:
            # Count repeated tokens in windows of 3
            for j in range(len(seq)-2):
                window = seq[j:j+3]
                unique_tokens = torch.unique(window)
                rep_penalty += 1 - (len(unique_tokens) / len(window))
    
    rep_penalty = rep_penalty / (batch_size * seq_len)
    
    # Combine losses
    total_loss = ce_loss + alpha * rep_penalty
    return total_loss

def train(model_config: MsingiConfig, train_texts: List[str], val_texts: Optional[List[str]] = None,
         training_config: Optional[TrainingConfig] = None, tokenizer_path: str = "tokenizer/tokenizer.json"):
    if training_config is None:
        training_config = TrainingConfig()
    
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
                
                # Compute loss with repetition penalty
                loss = compute_loss_with_penalty(logits, labels)
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
                    try:
                        # Ensure checkpoint directory exists
                        os.makedirs(training_config.checkpoint_dir, exist_ok=True)
                        
                        # Create checkpoint
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss,
                            'config': model_config,
                        }
                        if scaler is not None:
                            checkpoint['scaler_state_dict'] = scaler.state_dict()
                        
                        # Save latest checkpoint
                        checkpoint_path = os.path.join(training_config.checkpoint_dir, 'latest.pt')
                        print(f'\nSaving checkpoint to {checkpoint_path}')
                        torch.save(checkpoint, checkpoint_path)
                        print('Checkpoint saved successfully')
                        
                        if val_texts:
                            val_loss = evaluate(model, val_loader, model_config, device, training_config.fp16)
                            
                            if use_wandb:
                                wandb.log({
                                    'val_loss': val_loss,
                                    'epoch': epoch,
                                    'global_step': global_step,
                                })
                            
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                patience_counter = 0
                                best_path = os.path.join(training_config.checkpoint_dir, 'best.pt')
                                print(f'New best validation loss: {val_loss:.4f}, saving to {best_path}')
                                torch.save(checkpoint, best_path)
                            else:
                                patience_counter += 1
                                if patience_counter >= 3:
                                    print(f'Early stopping triggered after {epoch + 1} epochs')
                                    # Save final checkpoint before stopping
                                    final_path = os.path.join(training_config.checkpoint_dir, 'final.pt')
                                    print(f'Saving final model to {final_path}')
                                    torch.save(checkpoint, final_path)
                                    break
                    except Exception as e:
                        print(f'\nERROR saving checkpoint: {str(e)}')
                        print(f'Checkpoint directory: {training_config.checkpoint_dir}')
                        print(f'Directory exists: {os.path.exists(training_config.checkpoint_dir)}')
                        print(f'Directory is writable: {os.access(training_config.checkpoint_dir, os.W_OK)}')
                        raise
            
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
            epoch_ckpt_path = os.path.join(training_config.checkpoint_dir, f"epoch_{epoch+1}.pt")
            latest_ckpt_path = os.path.join(training_config.checkpoint_dir, "latest.pt")
            best_ckpt_path = os.path.join(training_config.checkpoint_dir, "best.pt")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'config': model_config,
                'training_config': training_config,
                'val_loss': val_loss if val_texts else None,
                'global_step': global_step,
            }
            # Save epoch checkpoint
            torch.save(checkpoint, epoch_ckpt_path)
            print(f"[Checkpoint] Saved epoch checkpoint: {epoch_ckpt_path}")
            # Save latest checkpoint
            torch.save(checkpoint, latest_ckpt_path)
            print(f"[Checkpoint] Saved latest checkpoint: {latest_ckpt_path}")
            # Save best checkpoint if improved
            if val_texts:
                if (not hasattr(train, 'best_val_loss')) or (val_loss < train.best_val_loss):
                    train.best_val_loss = val_loss
                    torch.save(checkpoint, best_ckpt_path)
                    print(f"[Checkpoint] Saved best checkpoint: {best_ckpt_path} (val_loss={val_loss:.4f})")
        except Exception as e:
            print(f"[Checkpoint] Error saving checkpoint: {str(e)}")
        
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

if __name__ == "__main__":
    # Print current directory and list contents
    print(f"Current working directory: {os.getcwd()}")
    
    # Set up data paths - files are in data/data/Swahili data/Swahili data/
    data_dir = os.path.join("data", "data", "Swahili data", "Swahili data")
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "valid.txt")
    
    # Set up tokenizer path
    tokenizer_path = os.path.join("data", "tokenizer", "tokenizer.json")
    
    # Check if directory and files exist
    print(f"\nChecking paths:")
    print(f"Data directory exists: {os.path.exists(data_dir)}")
    
    if os.path.exists(data_dir):
        print("\nListing contents of data directory:")
        for item in os.listdir(data_dir):
            print(f"  {item}")
    
    print(f"\nChecking files:")
    print(f"train.txt exists: {os.path.exists(train_path)}")
    print(f"valid.txt exists: {os.path.exists(val_path)}")
    print(f"tokenizer.json exists: {os.path.exists(tokenizer_path)}")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}. Please ensure the tokenizer file is in the data/tokenizer directory.")
    
    print("\nLoading and preparing training data...")
    train_texts = load_dataset(train_path)
    train_texts = prepare_dataset(train_texts)
    
    print("Loading and preparing validation data...")
    val_texts = load_dataset(val_path)
    val_texts = prepare_dataset(val_texts)
    
    print(f"Loaded {len(train_texts)} training samples and {len(val_texts)} validation samples")
    
    # Initialize model config with target architecture
    model_config = MsingiConfig(
        vocab_size=32000,
        max_position_embeddings=1024,
        n_embd=384,
        n_layer=6,
        n_head=6,
        intermediate_size=1536,
        dropout=0.1,
        rotary_emb=True,
        gradient_checkpointing=True
    )
    
    # Create checkpoint directory in Google Drive
    drive_checkpoint_dir = os.path.join(DRIVE_PATH, 'checkpoints')
    os.makedirs(drive_checkpoint_dir, exist_ok=True)
    print(f"\nSaving checkpoints to: {drive_checkpoint_dir}")
    
    # Initialize training config with Drive path
    training_config = TrainingConfig(
        num_epochs=10,
        batch_size=4,  # Using batch size 4
        grad_accum_steps=16,  # Adjusted accumulation steps
        learning_rate=3e-4,
        weight_decay=0.1,
        max_grad_norm=1.0,
        warmup_iters=1000,
        lr_decay_iters=20000,
        min_lr=3e-5,
        eval_interval=500,
        eval_iters=100,
        save_interval=1000,
        fp16=True,
        sequence_length=1024,
        checkpoint_dir=drive_checkpoint_dir  # Use Drive path
    )
    
    # Create checkpoint directory
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    
    # Train model
    train(model_config=model_config,
          train_texts=train_texts,
          val_texts=val_texts,
          training_config=training_config,
          tokenizer_path=tokenizer_path)
