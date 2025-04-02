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

def train(
    config: MsingiConfig,
    train_texts: List[str],
    val_texts: Optional[List[str]] = None,
    num_epochs: int = 10,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    max_length: int = 1024,
    warmup_steps: int = 1000,
    save_steps: int = 1000,
    eval_steps: int = 500,
    save_dir: str = "checkpoints",
    tokenizer_path: str = "tokenizer/tokenizer.json",
    use_wandb: bool = True,
):
    """Train the Msingi1 model."""
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="msingi1",
            config={
                "architecture": "Msingi1",
                "dataset": "Swahili",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "warmup_steps": warmup_steps,
            }
        )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Msingi1(config).to(device)
    
    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Create datasets
    train_dataset = SwahiliDataset(train_texts, tokenizer_path, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_texts:
        val_dataset = SwahiliDataset(val_texts, tokenizer_path, max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
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
    
    for epoch in range(num_epochs):
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
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss/(step+1):.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })
            
            global_step += 1
            
            # Evaluation
            if val_texts and global_step % eval_steps == 0:
                val_loss = evaluate(model, val_loader, config, device)
                
                if use_wandb:
                    wandb.log({
                        "train_loss": total_loss / (step + 1),
                        "val_loss": val_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    }, step=global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = os.path.join(save_dir, "best_model.pt")
                    torch.save(model.state_dict(), model_path)
            
            # Save checkpoint
            if global_step % save_steps == 0:
                model_path = os.path.join(save_dir, f"checkpoint-{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": total_loss / (step + 1),
                }, model_path)
    
    # Save final model
    model_path = os.path.join(save_dir, "final_model.pt")
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
    
    # Train the model
    train(
        config=config,
        train_texts=train_texts,
        val_texts=val_texts,
        num_epochs=10,
        batch_size=4,  # Small batch size for Colab
        gradient_accumulation_steps=16,  # Accumulate gradients to simulate larger batch
        learning_rate=3e-4,
        max_length=1024,
        warmup_steps=1000,
        save_steps=1000,
        eval_steps=500,
        use_wandb=True  # Set to False if you don't want to use wandb
    )
