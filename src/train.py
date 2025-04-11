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

# Colab Drive path
DRIVE_PATH = "/content/drive/MyDrive/msingi1"

@dataclass
class TrainingConfig:
    num_epochs: int = 25
    batch_size: int = 8  # Increased for T4
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    grad_acc_steps: int = 8  # Reduced since we increased batch size
    save_steps: int = 500
    eval_steps: int = 100
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    checkpoint_dir: str = 'checkpoints'
    fp16: bool = True  # Enable mixed precision
    sequence_length: int = 1024
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    eval_every: int = 100
    save_every: int = 500

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
         training_config: Optional[TrainingConfig] = None):
    if training_config is None:
        training_config = TrainingConfig()
    
    # Create checkpoint directory
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    
    # Initialize mixed precision training
    scaler = GradScaler('cuda') if training_config.fp16 and torch.cuda.is_available() else None
    # Initialize wandb
    use_wandb = True
    if use_wandb:
        wandb.init(
            project="msingi1",
            config={
                "architecture": {
                    "hidden_size": model_config.hidden_size,
                    "num_layers": model_config.num_hidden_layers,
                    "num_heads": model_config.num_attention_heads,
                    "vocab_size": model_config.vocab_size
                },
                "training": vars(training_config),
                "dataset": {
                    "num_samples": len(train_texts),
                    "sequence_length": training_config.sequence_length
                }
            }
        )
    
    # Create checkpoint directory
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    # Initialize model and move to device
    model = Msingi1(model_config)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    
    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(os.path.join(training_config.checkpoint_dir, 'latest.pt')):
        checkpoint = torch.load(os.path.join(training_config.checkpoint_dir, 'latest.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Resuming from epoch {start_epoch}')
    
    # Create datasets
    train_dataset = SwahiliDataset(train_texts, "tokenizer/tokenizer.json", training_config.sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    
    if val_texts:
        val_dataset = SwahiliDataset(val_texts, "tokenizer/tokenizer.json", training_config.sequence_length)
        val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * training_config.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
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
                outputs = model(input_ids)
                logits = outputs.logits
                
                # Compute loss with repetition penalty
                loss = compute_loss_with_penalty(logits, labels)
                loss = loss / training_config.gradient_accumulation_steps
            
            # Backward pass with mixed precision
            if training_config.fp16 and torch.cuda.is_available():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                if training_config.fp16 and torch.cuda.is_available():
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
                if (step + 1) % training_config.save_every == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'loss': loss.item(),
                        'best_val_loss': best_val_loss
                    }
                    torch.save(checkpoint, os.path.join(training_config.checkpoint_dir, 'latest.pt'))
                    
                    # Early stopping check
                    if val_texts:
                        val_loss = evaluate(model, val_loader, model_config, device, training_config.fp16)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            torch.save(checkpoint, os.path.join(training_config.checkpoint_dir, 'best.pt'))
                        else:
                            patience_counter += 1
                            if patience_counter >= training_config.early_stopping_patience:
                                print(f'Early stopping triggered after {epoch + 1} epochs')
                                break
            
            total_loss += loss.item() * training_config.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss/(step+1):.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })
            
            global_step += 1
            
            # Evaluation
            if val_texts and global_step % training_config.eval_every == 0:
                val_loss = evaluate(model, val_loader, model_config, device, training_config.fp16)
                
                if use_wandb:
                    wandb.log({
                        "train_loss": total_loss / (step + 1),
                        "val_loss": val_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    }, step=global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = os.path.join(training_config.checkpoint_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'loss': val_loss,
                        'best_val_loss': best_val_loss
                    }, model_path)
            
            lr_scheduler.step()
    
    # Save final model
    model_path = os.path.join(training_config.checkpoint_dir, "final_model.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss.item(),
        'best_val_loss': best_val_loss
    }, model_path)
    
    if use_wandb:
        wandb.finish()

def evaluate(model, val_loader, config, device, fp16=False):
    """Evaluate the model on validation data with optional mixed precision."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            with autocast(enabled=fp16 and torch.cuda.is_available()):
                outputs = model(input_ids)
                logits = outputs
                
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

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
    print("\nListing contents of current directory:")
    for item in os.listdir('.'):
        print(f"  {item}")
    
    # Set up data paths - data files are in data/Swahili data directory
    data_dir = "data/Swahili data"
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "valid.txt")
    
    # Check if directory and files exist
    print(f"\nChecking paths:")
    print(f"Data directory exists: {os.path.exists(data_dir)}")
    if os.path.exists(data_dir):
        print("\nListing contents of data directory:")
        for item in os.listdir(data_dir):
            print(f"  {item}")
    
    print(f"\nChecking data files:")
    print(f"train.txt exists: {os.path.exists(train_path)}")
    print(f"valid.txt exists: {os.path.exists(val_path)}")
    
    print("\nLoading and preparing training data...")
    train_texts = load_dataset(train_path)
    train_texts = prepare_dataset(train_texts)
    
    print("Loading and preparing validation data...")
    val_texts = load_dataset(val_path)
    val_texts = prepare_dataset(val_texts)
    
    print(f"Loaded {len(train_texts)} training samples and {len(val_texts)} validation samples")
    
    # Initialize model config with smaller architecture for testing
    model_config = MsingiConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        vocab_size=50000,
        block_size=2048,
        dropout=0.1
    )
    
    # Initialize training config
    training_config = TrainingConfig(
        learning_rate=3e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        warmup_iters=1000,
        lr_decay_iters=20000,
        min_lr=3e-5,
        batch_size=4,
        grad_acc_steps=16,  # Effective batch size = 64
        max_iters=50000,
        eval_interval=500,
        eval_iters=100,
        save_interval=1000,
        fp16=True,
        sequence_length=512,  # Shorter sequences for better memory usage
        checkpoint_dir='checkpoints'
    )
    
    # Create checkpoint directory
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    
    # Train model
    train(model_config=model_config,
          train_texts=train_texts,
          val_texts=val_texts,
          training_config=training_config)
