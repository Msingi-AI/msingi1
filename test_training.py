import torch
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from src.model import Msingi1, MsingiConfig
from src.data_processor import SwahiliDataset, extract_dataset
from tqdm import tqdm

def test_train(batch_size=2, max_length=512):
    print("Loading tokenizer...")
    tokenizer = ByteLevelBPETokenizer(
        "tokenizer/vocab.json",
        "tokenizer/merges.txt"
    )

    print("Loading dataset...")
    # Create a small test text for quick testing
    test_texts = [
        "Habari ya leo? Ninafurahi kukutana nawe.",
        "Karibu Tanzania, nchi nzuri sana.",
        "Mimi ni mwanafunzi wa Kiswahili.",
        "Jina langu ni Msingi, ninafundisha lugha."
    ]

    print("Creating dataset...")
    dataset = SwahiliDataset(
        texts=test_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=256
    )

    print("Creating dataloader...")
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    print("Initializing model...")
    config = MsingiConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        num_experts=8,
        expert_capacity=32,
        moe_layers=[2, 4]
    )

    model = Msingi1(config)
    if torch.cuda.is_available():
        print("Moving model to GPU...")
        model = model.cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    print("\nStarting training for one epoch...")
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    try:
        for step, batch in enumerate(progress_bar):
            # Move batch to GPU if available
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            
            # Forward pass
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise e

    print(f"\nTraining completed! Average loss: {total_loss / len(train_loader):.4f}")
    
    # Test text generation
    print("\nTesting text generation...")
    model.eval()
    test_prompt = "Habari ya"
    with torch.no_grad():
        encoded = tokenizer.encode(test_prompt)
        input_ids = torch.tensor([encoded.ids])
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        outputs = model.generate(
            input_ids=input_ids,
            max_length=50,
            num_return_sequences=1
        )
        generated_text = tokenizer.decode(outputs[0].tolist())
        print(f"Prompt: {test_prompt}")
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    test_train()
