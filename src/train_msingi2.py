import os
import sys
import torch
import argparse
import numpy as np
import math
import time
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple
import inspect

# Set tokenizers parallelism environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model_v2 import Msingi2, Msingi2Config
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from transformers import PreTrainedTokenizerFast
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
    """Configuration for Msingi2 training"""
    def __init__(
        self,
        # Data settings
        tokens_dir: str = "msingi_tokens",
        tokenizer_path: str = "tokenizer/swahili_unigram_32000/tokenizer.json",
        
        # Training settings
        num_epochs: int = 4,
        batch_size: int = 8,  
        grad_accum_steps: int = 8,  
        sequence_length: int = 1024,
        
        # Optimization settings
        learning_rate: float = 3e-4,  
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.05,  
        min_lr_ratio: float = 0.1,  
        
        # Evaluation and saving
        eval_interval: int = 500,
        eval_iters: int = 100,
        save_interval: int = 1000,
        
        # Technical settings
        fp16: bool = True,
        checkpoint_dir: str = "checkpoints_msingi2",
        log_interval: int = 10,
        seed: int = 42,
        
        # Wandb settings
        use_wandb: bool = True,
        wandb_project: str = "msingi2_v2",  # Updated project name
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        
        # Repetition penalty for loss
        rep_penalty_alpha: float = 0.1,
    ):
        self.tokens_dir = tokens_dir
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
        
        self.fp16 = fp16
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.seed = seed
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name or f"msingi2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self.rep_penalty_alpha = rep_penalty_alpha
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2)

class ShardedTokenDataset(Dataset):
    """Dataset that loads tokens from sharded numpy files"""
    
    def __init__(self, tokens_dir, split, seq_length):
        """
        Initialize the dataset.
        
        Args:
            tokens_dir: Directory containing token shards
            split: 'train' or 'val'
            seq_length: Sequence length for training
        """
        self.tokens_dir = tokens_dir
        self.split = split
        self.seq_length = seq_length
        
        # Find all shards for this split
        self.shard_paths = []
        for filename in os.listdir(tokens_dir):
            if filename.startswith(f"msingi_{split}_") and filename.endswith(".npy"):
                self.shard_paths.append(os.path.join(tokens_dir, filename))
        
        self.shard_paths = sorted(self.shard_paths)
        
        if len(self.shard_paths) == 0:
            raise ValueError(f"No {split} shards found in {tokens_dir}")
        
        # Load metadata
        metadata_path = os.path.join(tokens_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            print(f"Warning: No metadata.json found in {tokens_dir}")
            self.metadata = {}
        
        # Calculate total tokens
        self.total_tokens = 0
        for path in self.shard_paths:
            shard_size = os.path.getsize(path) // 2  # uint16 = 2 bytes
            self.total_tokens += shard_size
        
        # Initialize state
        self.current_shard_idx = -1
        self.current_shard_data = None
        
        print(f"Loaded {split} dataset with {len(self.shard_paths)} shards, {self.total_tokens:,} tokens")
    
    def __len__(self):
        """Return an approximate length of the dataset"""
        # This is approximate and used for progress bars
        return self.total_tokens // self.seq_length
    
    def __getitem__(self, idx):
        """Get a sequence of tokens starting at a random position"""
        # Generate a random position within the dataset
        seq_start = random.randint(0, self.total_tokens - self.seq_length - 1)
        
        # Find which shard contains this position
        tokens_seen = 0
        target_shard_idx = 0
        
        for i, path in enumerate(self.shard_paths):
            shard_size = os.path.getsize(path) // 2  # uint16 = 2 bytes
            if tokens_seen + shard_size > seq_start:
                target_shard_idx = i
                break
            tokens_seen += shard_size
        
        # Load shard if needed
        if target_shard_idx != self.current_shard_idx:
            self.current_shard_idx = target_shard_idx
            self.current_shard_data = np.load(self.shard_paths[target_shard_idx])
        
        # Get position within shard
        pos_in_shard = seq_start - tokens_seen
        
        # Get sequence
        if pos_in_shard + self.seq_length + 1 <= len(self.current_shard_data):
            # Sequence fits in current shard
            tokens = self.current_shard_data[pos_in_shard:pos_in_shard + self.seq_length + 1]
        else:
            # Sequence spans two shards
            tokens = self.current_shard_data[pos_in_shard:]
            remaining = self.seq_length + 1 - len(tokens)
            
            # Load next shard
            next_shard_idx = (target_shard_idx + 1) % len(self.shard_paths)
            next_shard_data = np.load(self.shard_paths[next_shard_idx])
            tokens = np.concatenate([tokens, next_shard_data[:remaining]])
        
        # Create input_ids and labels for causal LM
        input_ids = tokens[:-1]  # All tokens except last
        labels = tokens[1:]      # All tokens except first
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

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
    rep_penalty = 0
    
    # For each sequence in the batch
    for i in range(batch_size):
        # Get non-padded tokens
        seq = labels[i][labels[i] != -100]
        if len(seq) > 2:
            # Count repeated tokens in windows of 3
            for j in range(len(seq)-2):
                window = seq[j:j+3]
                unique_tokens = torch.unique(window)
                rep_penalty += 1 - (len(unique_tokens) / len(window))
    
    rep_penalty = rep_penalty / (batch_size * seq_len)
    
    # Combine losses
    total_loss = ce_loss + rep_penalty_alpha * rep_penalty
    return total_loss

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
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def generate_sample_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.9, top_k=40, repetition_penalty=1.1):
    """
    Generate sample text from the model given a prompt
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text

def evaluate(model, dataloader, config, device, fp16=False, return_metrics=False):
    """Evaluate model on validation data"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0  # For token prediction accuracy
    
    # For perplexity distribution
    batch_perplexities = []
    token_losses = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= config.eval_iters:
                break
                
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            if fp16:
                with autocast():
                    logits, loss = model(input_ids, labels)
            else:
                logits, loss = model(input_ids, labels)
            
            # Calculate token prediction accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = (labels != -100)  # Ignore padding tokens
            correct = ((predictions == labels) & mask).sum().item()
            total_correct += correct
            
            # Track per-batch perplexity
            batch_loss = loss.item()
            batch_perplexities.append(math.exp(batch_loss))
            
            # Calculate per-token losses for visualization (sample up to 1000 tokens)
            if len(token_losses) < 1000:
                with torch.no_grad():
                    # Get per-token loss
                    log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
                    token_log_probs = -log_probs.gather(1, labels.view(-1, 1))
                    token_log_probs = token_log_probs.view(labels.shape)
                    
                    # Only consider non-padding tokens
                    valid_tokens = (labels != -100).flatten()
                    if valid_tokens.sum() > 0:
                        valid_losses = token_log_probs.flatten()[valid_tokens]
                        token_losses.extend(valid_losses.cpu().tolist()[:100])  # Limit to 100 tokens per batch
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0) * input_ids.size(1)
            total_valid_tokens = mask.sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0
    
    if return_metrics:
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "batch_perplexities": batch_perplexities,
            "token_losses": token_losses
        }
    
    return avg_loss

def train(model_config: Msingi2Config, training_config: TrainingConfig, resume_from=None):
    """Train the Msingi2 model using sharded token datasets"""
    # Set random seeds for reproducibility
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    random.seed(training_config.seed)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = Msingi2(model_config)
    
    # Track starting epoch and global step for resumption
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint_path = Path(resume_from)
        
        # Load model weights
        model_path = checkpoint_path / "pytorch_model.bin"
        if model_path.exists():
            print(f"Loading model weights from {model_path}")
            model.load_state_dict(torch.load(str(model_path), map_location=device))
        else:
            raise ValueError(f"Model weights not found at {model_path}")
        
        # Determine starting epoch from checkpoint name
        if "epoch-" in checkpoint_path.name:
            try:
                start_epoch = int(checkpoint_path.name.split("-")[1])
                print(f"Resuming from epoch {start_epoch}")
            except (ValueError, IndexError):
                print("Could not determine epoch from checkpoint name, starting from epoch 0")
        
        # Load training state if available
        training_state_path = checkpoint_path / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
                global_step = training_state.get("global_step", 0)
                best_val_loss = training_state.get("best_val_loss", float('inf'))
                print(f"Resuming from global step {global_step} with best validation loss {best_val_loss:.4f}")
        else:
            # If no training state file exists, create one based on the epoch number
            print(f"No training state found, creating one based on epoch {start_epoch}")
            # Estimate global step based on epoch
            # Assuming ~10000 steps per epoch based on your dataset size
            global_step = start_epoch * 10000
    
    model.to(device)
    
    # Load tokenizer
    tokenizer_path = training_config.tokenizer_path
    print(f"Loading Unigram tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Update model config with tokenizer vocab size
    if hasattr(tokenizer, 'vocab_size'):
        model_config.vocab_size = tokenizer.vocab_size
        print(f"Updated model vocab size to match tokenizer: {model_config.vocab_size}")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = ShardedTokenDataset(
        training_config.tokens_dir,
        "train",
        training_config.sequence_length
    )
    
    valid_dataset = ShardedTokenDataset(
        training_config.tokens_dir,
        "val",
        training_config.sequence_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Set up optimizer
    optimizer = model.configure_optimizers(
        weight_decay=training_config.weight_decay,
        learning_rate=training_config.learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if device.type == 'cuda' else 'cpu'
    )
    
    # Set up learning rate scheduler
    num_training_steps = len(train_loader) * training_config.num_epochs // training_config.grad_accum_steps
    num_warmup_steps = int(num_training_steps * training_config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Set up mixed precision training
    scaler = GradScaler() if training_config.fp16 else None
    
    # Initialize WandB if enabled
    if training_config.use_wandb:
        print("\n===== Initializing Weights & Biases =====")
        print(f"Project: {training_config.wandb_project}")
        print(f"Run name: {training_config.wandb_run_name}")
        try:
            wandb.init(
                project=training_config.wandb_project,
                entity=training_config.wandb_entity,
                name=training_config.wandb_run_name,
                config={
                    "model_config": model_config.__dict__,
                    "training_config": training_config.__dict__
                }
            )
            if wandb.run is not None:
                print(f"Successfully initialized wandb! Run ID: {wandb.run.id}")
                print(f"View run at: {wandb.run.get_url()}")
            else:
                print("WARNING: wandb.init() completed but wandb.run is None!")
        except Exception as e:
            print(f"ERROR initializing wandb: {str(e)}")
            print("Training will continue without wandb logging.")
            training_config.use_wandb = False
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(start_epoch, training_config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision if enabled
            if training_config.fp16:
                with autocast():
                    logits, loss = model(input_ids, labels)
                    loss = loss / training_config.grad_accum_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % training_config.grad_accum_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                    
                    # Optimizer step with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    
                    global_step += 1
            else:
                # Standard forward and backward pass
                logits, loss = model(input_ids, labels)
                loss = loss / training_config.grad_accum_steps
                
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % training_config.grad_accum_steps == 0:
                    # Gradient clipping
                    clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    
                    global_step += 1
            
            # Track loss
            epoch_loss += loss.item() * training_config.grad_accum_steps * input_ids.size(0)
            epoch_tokens += input_ids.size(0) * input_ids.size(1)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item() * training_config.grad_accum_steps,
                "lr": scheduler.get_last_lr()[0],
                "step": global_step
            })
            
            # Log metrics
            if global_step > 0 and batch_idx % training_config.log_interval == 0:
                if training_config.use_wandb:
                    wandb.log({
                        "train/loss": loss.item() * training_config.grad_accum_steps,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + batch_idx / len(train_loader),
                        "train/global_step": global_step
                    })
            
            # Evaluate and save checkpoint
            if global_step > 0 and global_step % training_config.eval_interval == 0:
                # Get detailed evaluation metrics
                eval_metrics = evaluate(model, valid_loader, training_config, device, training_config.fp16, return_metrics=True)
                val_loss = eval_metrics["loss"]
                val_ppl = eval_metrics["perplexity"]
                val_acc = eval_metrics["accuracy"]
                
                # Convert accuracy to percentage for display (0-100 scale)
                print(f"\nStep {global_step} | Validation loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f} | Accuracy: {val_acc*100:.2f}%")
                
                # Generate a sample text for qualitative evaluation
                if hasattr(tokenizer, "decode"):
                    try:
                        sample_prompt = "Habari ya leo? "
                        sample_text = generate_sample_text(model, tokenizer, sample_prompt, max_length=100)
                        print(f"\nSample generation:\n{sample_text}\n")
                    except Exception as e:
                        print(f"Error generating sample: {e}")
                
                if training_config.use_wandb:
                    # Log metrics
                    wandb_log = {
                        "eval/loss": val_loss,
                        "eval/perplexity": val_ppl,
                        "eval/accuracy": val_acc,
                        "eval/global_step": global_step,
                        "train/val_loss_ratio": epoch_loss / val_loss if epoch_loss > 0 else 0,
                    }
                    
                    # Log sample text
                    if 'sample_text' in locals():
                        wandb_log["samples/text"] = wandb.Html(f"<pre>{sample_text}</pre>")
                    
                    # Create perplexity histogram
                    if len(eval_metrics["token_losses"]) > 0:
                        wandb_log["eval/token_loss_hist"] = wandb.Histogram(eval_metrics["token_losses"])
                    
                    # Create perplexity distribution plot
                    if len(eval_metrics["batch_perplexities"]) > 0:
                        wandb_log["eval/batch_perplexity"] = wandb.Histogram(eval_metrics["batch_perplexities"])
                    
                    wandb.log(wandb_log)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(training_config.checkpoint_dir, "best_model")
                    os.makedirs(best_model_path, exist_ok=True)
                    
                    # Save model
                    torch.save(model.state_dict(), os.path.join(best_model_path, "pytorch_model.bin"))
                    
                    # Save configs
                    with open(os.path.join(best_model_path, "model_config.json"), 'w') as f:
                        json.dump(model_config.__dict__, f, indent=2)
                    
                    training_config.save(os.path.join(best_model_path, "training_config.json"))
                    
                    print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
            # Save regular checkpoint
            if global_step > 0 and global_step % training_config.save_interval == 0:
                checkpoint_path = os.path.join(training_config.checkpoint_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save model
                torch.save(model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
                
                # Save configs
                with open(os.path.join(checkpoint_path, "model_config.json"), 'w') as f:
                    json.dump(model_config.__dict__, f, indent=2)
                
                training_config.save(os.path.join(checkpoint_path, "training_config.json"))
                
                print(f"Saved checkpoint at step {global_step}")
        
        # End of epoch
        epoch_loss = epoch_loss / epoch_tokens
        epoch_ppl = math.exp(epoch_loss)
        print(f"Epoch {epoch+1}/{training_config.num_epochs} complete | Loss: {epoch_loss:.4f} | Perplexity: {epoch_ppl:.2f}")
        
        # Run full validation at end of epoch
        print("Running full validation...")
        eval_metrics = evaluate(model, valid_loader, training_config, device, training_config.fp16, return_metrics=True)
        val_loss = eval_metrics["loss"]
        val_ppl = eval_metrics["perplexity"]
        val_acc = eval_metrics["accuracy"]
        
        print(f"Validation | Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f} | Accuracy: {val_acc:.2%}")
        
        # Plot training vs validation loss
        if training_config.use_wandb:
            # Create learning curve plot
            wandb_log = {
                "train/epoch_loss": epoch_loss,
                "train/epoch_perplexity": epoch_ppl,
                "train/epoch": epoch + 1,
                "epoch_eval/loss": val_loss,
                "epoch_eval/perplexity": val_ppl,
                "epoch_eval/accuracy": val_acc,
                "epoch_eval/train_val_gap": epoch_loss - val_loss
            }
            
            # Create a custom chart for overfitting visualization
            data = [[epoch + 1, epoch_loss, val_loss]]
            table = wandb.Table(data=data, columns=["epoch", "train_loss", "val_loss"])
            wandb_log["learning_curves"] = wandb.plot.line(
                table, "epoch", ["train_loss", "val_loss"], 
                title="Training vs Validation Loss")
                
            wandb.log(wandb_log)
        
        # Save epoch checkpoint
        epoch_path = os.path.join(training_config.checkpoint_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_path, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(epoch_path, "pytorch_model.bin"))
        
        # Save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(epoch_path, "optimizer.pt"))
        
        # Save training state
        training_state = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_val_loss": best_val_loss
        }
        with open(os.path.join(epoch_path, "training_state.json"), 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Save configs
        with open(os.path.join(epoch_path, "model_config.json"), 'w') as f:
            json.dump(model_config.__dict__, f, indent=2)
        
        training_config.save(os.path.join(epoch_path, "training_config.json"))
        
        print(f"Saved checkpoint for epoch {epoch+1}")
    
    print("Training complete!")
    return model, best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train Msingi2 Swahili language model with sharded tokens")
    
    # Data arguments
    parser.add_argument("--tokens-dir", type=str, default="msingi_tokens", help="Directory containing token shards")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--seq-length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from a checkpoint directory")
    
    # Optimization arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    
    # Technical arguments
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_msingi2", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # WandB arguments
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="msingi2_v2", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    
    # Model arguments
    parser.add_argument("--n-layer", type=int, default=18, help="Number of layers")
    parser.add_argument("--n-head", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=768, help="Embedding dimension")
    
    args = parser.parse_args()
    
    # Create model config
    model_config = Msingi2Config(
        vocab_size=32000,  # Will be updated based on tokenizer
        block_size=args.seq_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.1,
        gradient_checkpointing=True
    )
    
    # Ensure embedding dimension is divisible by number of heads
    if model_config.n_embd % model_config.n_head != 0:
        # Adjust n_embd to be divisible by n_head
        model_config.n_embd = model_config.n_head * (model_config.n_embd // model_config.n_head)
        print(f"Adjusted embedding dimension to {model_config.n_embd} to be divisible by {model_config.n_head} heads")
    
    # Create training config
    training_config = TrainingConfig(
        tokens_dir=args.tokens_dir,
        
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        sequence_length=args.seq_length,
        
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        
        fp16=not args.no_fp16,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name
    )
    
    # Train the model
    model, best_val_loss = train(model_config, training_config, args.resume_from)
    
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
