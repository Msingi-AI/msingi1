import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from model import Msingi1, MsingiConfig
from data_processor import extract_dataset, get_dataset_stats
from tqdm import tqdm
import wandb
import os

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

def train(model, train_dataloader, optimizer, device, num_epochs):
    model.train()
    wandb.init(project="msingi1")
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            wandb.log({
                'loss': loss.item(),
                'epoch': epoch
            })
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
        
        # Save checkpoint
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt')

def main():
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
        max_position_embeddings=512,  # Smaller context window for Colab
        hidden_size=256,  # Smaller model for Colab
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
    )
    
    model = Msingi1(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
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
    os.makedirs('tokenizer', exist_ok=True)
    tokenizer.save_model('tokenizer')
    
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
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train
    print("\nStarting training...")
    train(model, train_dataloader, optimizer, device, num_epochs=10)

if __name__ == "__main__":
    main()
