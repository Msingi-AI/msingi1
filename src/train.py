import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from model import Msingi1, MsingiConfig
from data_processor import extract_dataset
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import math
from typing import Optional, List
import numpy as np
from tokenizers import Tokenizer

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

def train(config: MsingiConfig, train_texts: List[str], val_texts: Optional[List[str]] = None,
         num_epochs: int = 100,
         batch_size: int = 4,
         learning_rate: float = 3e-4,
         warmup_steps: int = 1000,
         grad_acc_steps: int = 16,
         save_steps: int = 1000,
         checkpoint_dir: str = 'checkpoints',
         device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    # Initialize wandb
    use_wandb = True
    if use_wandb:
        wandb.init(
            project="msingi1",
            config={
                "architecture": "Msingi1",
                "dataset": "Swahili",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_acc_steps,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
            }
        )
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model and move to device
    model = Msingi1(config)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(os.path.join(checkpoint_dir, 'latest.pt')):
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'latest.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Resuming from epoch {start_epoch}')
    
    # Create datasets
    train_dataset = SwahiliDataset(train_texts, "tokenizer/tokenizer.json", 1024)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_texts:
        val_dataset = SwahiliDataset(val_texts, "tokenizer/tokenizer.json", 1024)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Scale loss by gradient accumulation steps
            loss = loss / grad_acc_steps
            loss.backward()
            
            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if device == 'tpu':
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
                # Save checkpoint
                if (step + 1) % save_steps == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pt'))
                    
                    # Save best model
                    if val_texts and loss.item() < best_val_loss:
                        best_val_loss = loss.item()
                        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pt'))
            
            total_loss += loss.item() * grad_acc_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss/(step+1):.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })
            
            global_step += 1
            
            # Evaluation
            if val_texts and global_step % 500 == 0:
                val_loss = evaluate(model, val_loader, config, device)
                
                if use_wandb:
                    wandb.log({
                        "train_loss": total_loss / (step + 1),
                        "val_loss": val_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    }, step=global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save(model.state_dict(), model_path)
            
            lr_scheduler.step()
    
    # Save final model
    model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    
    if use_wandb:
        wandb.finish()

def evaluate(model, val_loader, config, device):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
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

if __name__ == "__main__":
    # Load dataset
    texts = extract_dataset("archive.zip")
    
    # Split into train/val
    val_size = int(len(texts) * 0.1)
    train_texts = texts[val_size:]
    val_texts = texts[:val_size]
    
    # Initialize config
    config = MsingiConfig()
    
    # Detect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif os.environ.get('COLAB_TPU_ADDR'):
        device = 'tpu'
        # TPU-specific imports and setup
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
    
    print(f'Training on {device}')
    
    # Train model
    train(config, train_texts, val_texts,
          num_epochs=100,
          batch_size=4,
          learning_rate=3e-4,
          warmup_steps=1000,
          grad_acc_steps=16,
          save_steps=1000,
          checkpoint_dir='checkpoints',
          device=device)
