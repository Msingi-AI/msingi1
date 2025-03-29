"""
Script to generate a Colab-compatible notebook for Msingi1 training.
"""
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cells
markdown_cells = [
    {
        "cell_type": "markdown",
        "metadata": {"id": "view-in-github"},
        "source": [
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/msingi1/blob/main/Msingi1_Training.ipynb)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {"id": "intro"},
        "source": [
            "# Msingi1 Training Notebook\n",
            "\n",
            "This notebook trains the Msingi1 Swahili language model using Mixture of Experts architecture."
        ]
    }
]

# Code cells
code_cells = [
    {
        "cell_type": "code",
        "metadata": {"id": "mount_drive"},
        "source": [
            "from google.colab import drive\n",
            "drive.mount('/content/drive')"
        ],
        "outputs": []
    },
    {
        "cell_type": "code",
        "metadata": {"id": "setup_env"},
        "source": [
            "# Clone repository\n",
            "!git clone https://github.com/your-username/msingi1.git\n",
            "%cd msingi1\n",
            "\n",
            "# Install dependencies and setup environment\n",
            "!python setup_colab.py"
        ],
        "outputs": []
    },
    {
        "cell_type": "code",
        "metadata": {"id": "load_data"},
        "source": [
            "import torch\n",
            "from torch.utils.data import DataLoader\n",
            "from tokenizers import ByteLevelBPETokenizer\n",
            "from src.data_processor import SwahiliDataset, extract_dataset\n",
            "\n",
            "# Load tokenizer\n",
            "tokenizer = ByteLevelBPETokenizer(\n",
            "    'tokenizer/vocab.json',\n",
            "    'tokenizer/merges.txt'\n",
            ")\n",
            "\n",
            "# Extract dataset\n",
            "texts = extract_dataset('data/swahili_dataset.zip')\n",
            "\n",
            "# Create dataset\n",
            "dataset = SwahiliDataset(\n",
            "    texts=texts,\n",
            "    tokenizer=tokenizer,\n",
            "    max_length=1024,\n",
            "    stride=512\n",
            ")\n",
            "\n",
            "# Create dataloader\n",
            "train_loader = DataLoader(\n",
            "    dataset,\n",
            "    batch_size=4,  # Adjust based on GPU memory\n",
            "    shuffle=True,\n",
            "    num_workers=2\n",
            ")"
        ],
        "outputs": []
    },
    {
        "cell_type": "code",
        "metadata": {"id": "init_model"},
        "source": [
            "from src.model import Msingi1, MsingiConfig\n",
            "\n",
            "# Initialize model configuration\n",
            "config = MsingiConfig(\n",
            "    vocab_size=32000,\n",
            "    hidden_size=768,\n",
            "    num_hidden_layers=12,\n",
            "    num_attention_heads=12,\n",
            "    intermediate_size=3072,\n",
            "    num_experts=8,\n",
            "    expert_capacity=32\n",
            ")\n",
            "\n",
            "# Initialize model\n",
            "model = Msingi1(config)\n",
            "model = model.cuda()  # Move to GPU\n",
            "\n",
            "# Initialize optimizer\n",
            "optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)\n",
            "\n",
            "# Learning rate scheduler with warmup\n",
            "from transformers import get_linear_schedule_with_warmup\n",
            "\n",
            "num_training_steps = len(train_loader) * 50  # 50 epochs\n",
            "num_warmup_steps = num_training_steps // 10  # 10% warmup\n",
            "\n",
            "scheduler = get_linear_schedule_with_warmup(\n",
            "    optimizer,\n",
            "    num_warmup_steps=num_warmup_steps,\n",
            "    num_training_steps=num_training_steps\n",
            ")"
        ],
        "outputs": []
    },
    {
        "cell_type": "code",
        "metadata": {"id": "training_loop"},
        "source": [
            "import os\n",
            "from tqdm.notebook import tqdm\n",
            "import wandb\n",
            "\n",
            "# Initialize wandb\n",
            "wandb.init(project='msingi1', name='training_run_1')\n",
            "\n",
            "# Training parameters\n",
            "num_epochs = 50\n",
            "gradient_accumulation_steps = 4\n",
            "checkpoint_dir = '/content/drive/MyDrive/msingi1/checkpoints'\n",
            "os.makedirs(checkpoint_dir, exist_ok=True)\n",
            "\n",
            "# Training loop\n",
            "for epoch in range(num_epochs):\n",
            "    model.train()\n",
            "    total_loss = 0\n",
            "    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')\n",
            "    \n",
            "    for step, batch in enumerate(progress_bar):\n",
            "        # Move batch to GPU\n",
            "        input_ids = batch['input_ids'].cuda()\n",
            "        labels = batch['labels'].cuda()\n",
            "        \n",
            "        # Forward pass\n",
            "        outputs = model(input_ids=input_ids, labels=labels)\n",
            "        loss = outputs.loss / gradient_accumulation_steps\n",
            "        loss.backward()\n",
            "        \n",
            "        # Update weights every gradient_accumulation_steps\n",
            "        if (step + 1) % gradient_accumulation_steps == 0:\n",
            "            optimizer.step()\n",
            "            scheduler.step()\n",
            "            optimizer.zero_grad()\n",
            "        \n",
            "        # Update progress\n",
            "        total_loss += loss.item() * gradient_accumulation_steps\n",
            "        progress_bar.set_postfix({'loss': total_loss / (step + 1)})\n",
            "        \n",
            "        # Log to wandb\n",
            "        wandb.log({\n",
            "            'loss': loss.item() * gradient_accumulation_steps,\n",
            "            'learning_rate': scheduler.get_last_lr()[0]\n",
            "        })\n",
            "    \n",
            "    # Save checkpoint every 5 epochs\n",
            "    if (epoch + 1) % 5 == 0:\n",
            "        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')\n",
            "        torch.save({\n",
            "            'epoch': epoch + 1,\n",
            "            'model_state_dict': model.state_dict(),\n",
            "            'optimizer_state_dict': optimizer.state_dict(),\n",
            "            'scheduler_state_dict': scheduler.state_dict(),\n",
            "            'loss': total_loss / len(train_loader),\n",
            "        }, checkpoint_path)\n",
            "        print(f'\\nCheckpoint saved: {checkpoint_path}')\n",
            "\n",
            "print('Training completed!')\n",
            "wandb.finish()"
        ],
        "outputs": []
    },
    {
        "cell_type": "code",
        "metadata": {"id": "generate_text"},
        "source": [
            "def generate_text(prompt: str, max_length: int = 100):\n",
            "    model.eval()\n",
            "    with torch.no_grad():\n",
            "        # Encode prompt\n",
            "        encoded = tokenizer.encode(prompt)\n",
            "        input_ids = torch.tensor([encoded.ids]).cuda()\n",
            "        \n",
            "        # Generate\n",
            "        outputs = model.generate(\n",
            "            input_ids=input_ids,\n",
            "            max_length=max_length,\n",
            "            num_return_sequences=1,\n",
            "            no_repeat_ngram_size=2,\n",
            "            temperature=0.7\n",
            "        )\n",
            "        \n",
            "        # Decode and return\n",
            "        return tokenizer.decode(outputs[0].tolist())\n",
            "\n",
            "# Test generation\n",
            "prompt = 'Habari ya leo?'\n",
            "generated_text = generate_text(prompt)\n",
            "print(f'Prompt: {prompt}')\n",
            "print(f'Generated: {generated_text}')"
        ],
        "outputs": []
    }
]

# Add cells to notebook
for cell in markdown_cells + code_cells:
    nb.cells.append(nbf.v4.new_cell(**cell))

# Set notebook metadata
nb.metadata = {
    "accelerator": "GPU",
    "colab": {
        "name": "Msingi1_Training.ipynb",
        "provenance": [],
        "collapsed_sections": [],
        "machine_shape": "hm",
        "include_colab_link": True
    },
    "kernelspec": {
        "display_name": "Python 3",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.7.12"
    }
}

# Write the notebook
with open('Msingi1_Training.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
