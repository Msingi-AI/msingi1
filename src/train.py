import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from model import Msingi1, MsingiConfig
from data_processor import extract_dataset
from tqdm import tqdm
import wandb
import os
from pathlib import Path

class SwahiliDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

def train(
    model,
    train_dataloader,
    optimizer,
    device,
    num_epochs,
    save_dir="checkpoints",
    save_steps=100,
    logging_steps=10
):
    model.train()
    wandb.init(project="msingi1")
    
    global_step = 0
    best_loss = float('inf')
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"\nTraining on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Save steps: {save_steps}")
    print(f"Logging steps: {logging_steps}\n")
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/global_step:.4f}',
                'step': global_step
            })
            
            # Log to wandb
            if global_step % logging_steps == 0:
                wandb.log({
                    'loss': loss.item(),
                    'epoch': epoch,
                    'step': global_step
                })
            
            # Save checkpoint
            if global_step % save_steps == 0:
                avg_loss = total_loss / global_step
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    checkpoint_path = save_dir / f'best_model_step_{global_step}.pt'
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, checkpoint_path)
                    print(f"\nSaved best model checkpoint to {checkpoint_path}")
        
        # Save epoch checkpoint
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_dataloader),
        }, checkpoint_path)
        
        print(f"\nEpoch {epoch+1} average loss: {total_loss/len(train_dataloader):.4f}")

def main():
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    texts = extract_dataset("archive.zip")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained('tokenizer')
    
    # Initialize config and model
    print("\nInitializing model...")
    config = MsingiConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=512,  # Smaller context window for Colab
        hidden_size=256,  # Smaller model for Colab
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
    )
    
    model = Msingi1(config)
    model.to(device)
    
    # Create dataset and dataloader
    print("\nPreparing dataset...")
    dataset = SwahiliDataset(texts, tokenizer, max_length=512)
    train_dataloader = DataLoader(
        dataset,
        batch_size=4,  # Small batch size for Colab
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train
    print("\nStarting training...")
    train(
        model,
        train_dataloader,
        optimizer,
        device,
        num_epochs=10,
        save_steps=100,
        logging_steps=10
    )

if __name__ == "__main__":
    main()
