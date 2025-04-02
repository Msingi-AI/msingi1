import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from model import Msingi1, MsingiConfig
from data_processor import extract_dataset, get_dataset_stats
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import json
import torch.cuda.amp as amp

class SwahiliDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                 max_length=max_length, return_tensors='pt')
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, save_dir):
    """Save training checkpoint to disk"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config.__dict__
    }
    
    path = Path(save_dir) / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, path)
    
    # Save latest checkpoint reference
    latest = {
        'latest_checkpoint': f'checkpoint_epoch_{epoch}.pt',
        'epoch': epoch,
        'loss': loss
    }
    with open(Path(save_dir) / 'latest.json', 'w') as f:
        json.dump(latest, f)

def load_checkpoint(model, optimizer, scheduler, save_dir):
    """Load latest checkpoint from disk"""
    try:
        with open(Path(save_dir) / 'latest.json', 'r') as f:
            latest = json.load(f)
        
        checkpoint = torch.load(Path(save_dir) / latest['latest_checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']
    except FileNotFoundError:
        return 0, float('inf')

def train(model, train_dataloader, optimizer, scheduler, device, num_epochs, save_dir, resume_training=True):
    model.train()
    wandb.init(project="msingi1", config=model.config.__dict__)
    
    # Initialize mixed precision training
    scaler = amp.GradScaler()
    
    # Resume from checkpoint if available
    start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, save_dir) if resume_training else (0, float('inf'))
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            wandb.log({
                'loss': loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch+1, avg_loss, model.config, save_dir)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(save_dir) / 'best_model.pt')

def main(save_dir='drive/MyDrive/msingi1_checkpoints'):
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and process dataset
    print("Loading dataset...")
    texts = extract_dataset("archive.zip")
    stats = get_dataset_stats(texts)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Initialize config and model
    config = MsingiConfig(
        vocab_size=50000,
        max_position_embeddings=512,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        num_experts=4,
        expert_capacity=2,
        moe_layer_frequency=2
    )
    
    model = Msingi1(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Train tokenizer
    from tokenizers import ByteLevelBPETokenizer
    
    print("\nTraining tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=config.vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
    )
    
    # Save tokenizer
    tokenizer_path = Path(save_dir) / 'tokenizer'
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_model(str(tokenizer_path))
    
    # Convert to PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>"
    )
    
    # Create dataset and dataloader
    print("\nPreparing dataset...")
    dataset = SwahiliDataset(texts, tokenizer, max_length=512)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Train
    print("\nStarting training...")
    train(model, train_dataloader, optimizer, scheduler, device, num_epochs=10, save_dir=save_dir)

if __name__ == "__main__":
    main()
