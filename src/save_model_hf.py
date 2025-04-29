"""
Save Msingi1 model in Hugging Face Transformers format.

This script converts a trained Msingi1 model checkpoint to the standard Hugging Face
format, making it compatible with the transformers library and shareable on the
Hugging Face Hub.
"""

import os
import json
import torch
import shutil
from pathlib import Path
from tokenizers import Tokenizer
from model import Msingi1, MsingiConfig

def save_model_hf_format(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_path: str = "tokenizer/tokenizer.json",
    model_name: str = "msingi1-swahili",
    push_to_hub: bool = False,
    hub_token: str = None,
    hub_model_id: str = None,
):
    """
    Save a Msingi1 model in Hugging Face Transformers format.
    
    Args:
        checkpoint_path: Path to the PyTorch checkpoint file (.pt)
        output_dir: Directory where the model will be saved in HF format
        tokenizer_path: Path to the tokenizer.json file
        model_name: Name of the model (used in config and model card)
        push_to_hub: Whether to push the model to the Hugging Face Hub
        hub_token: Hugging Face API token (if pushing to Hub)
        hub_model_id: Model ID on the Hub (if None, will use model_name)
    """
    print(f"Converting checkpoint from {checkpoint_path} to HF format...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract model and config
    model_state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", None)
    
    if config is None:
        raise ValueError("Config not found in checkpoint. Cannot convert to HF format.")
    
    # Create config.json
    config_dict = {
        "model_type": "msingi1",
        "architectures": ["Msingi1ForCausalLM"],
        "vocab_size": config.vocab_size,
        "hidden_size": config.n_embd,
        "num_hidden_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "intermediate_size": config.intermediate_size,
        "hidden_act": "gelu",
        "max_position_embeddings": config.max_position_embeddings,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "initializer_range": config.initializer_range,
        "use_cache": config.use_cache,
        "pad_token_id": 3,  # Assuming <pad> is at index 3
        "bos_token_id": 0,  # Assuming <s> is at index 0
        "eos_token_id": 1,  # Assuming </s> is at index 1
        "rotary_emb": config.rotary_emb,
        "tie_word_embeddings": True,
        "gradient_checkpointing": config.gradient_checkpointing,
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Convert state dict to HF format
    hf_state_dict = {}
    
    # Mapping from our model keys to HF keys
    key_mapping = {
        "embeddings.weight": "transformer.wte.weight",
        "ln_f.weight": "transformer.ln_f.weight",
        "ln_f.bias": "transformer.ln_f.bias",
    }
    
    # Layer mappings
    for i in range(config.n_layer):
        # Attention weights
        key_mapping[f"layers.{i}.attention.wq.weight"] = f"transformer.h.{i}.attn.q_proj.weight"
        key_mapping[f"layers.{i}.attention.wk.weight"] = f"transformer.h.{i}.attn.k_proj.weight"
        key_mapping[f"layers.{i}.attention.wv.weight"] = f"transformer.h.{i}.attn.v_proj.weight"
        key_mapping[f"layers.{i}.attention.wo.weight"] = f"transformer.h.{i}.attn.c_proj.weight"
        
        # Layer norms
        key_mapping[f"layers.{i}.ln1.weight"] = f"transformer.h.{i}.ln_1.weight"
        key_mapping[f"layers.{i}.ln1.bias"] = f"transformer.h.{i}.ln_1.bias"
        key_mapping[f"layers.{i}.ln2.weight"] = f"transformer.h.{i}.ln_2.weight"
        key_mapping[f"layers.{i}.ln2.bias"] = f"transformer.h.{i}.ln_2.bias"
        
        # MLP layers
        key_mapping[f"layers.{i}.mlp.0.weight"] = f"transformer.h.{i}.mlp.c_fc.weight"
        key_mapping[f"layers.{i}.mlp.0.bias"] = f"transformer.h.{i}.mlp.c_fc.bias"
        key_mapping[f"layers.{i}.mlp.2.weight"] = f"transformer.h.{i}.mlp.c_proj.weight"
        key_mapping[f"layers.{i}.mlp.2.bias"] = f"transformer.h.{i}.mlp.c_proj.bias"
    
    # LM head (tied with embeddings)
    key_mapping["lm_head.weight"] = "lm_head.weight"
    
    # Convert keys
    for old_key, new_key in key_mapping.items():
        if old_key in model_state_dict:
            hf_state_dict[new_key] = model_state_dict[old_key]
    
    # Save model weights
    torch.save(hf_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    
    # Copy tokenizer files
    print(f"Copying tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Save tokenizer files in HF format
    tokenizer_config = {
        "model_type": "gpt2",
        "add_prefix_space": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Copy the original tokenizer.json
    shutil.copy(tokenizer_path, os.path.join(output_dir, "tokenizer.json"))
    
    # Create special_tokens_map.json
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens, f, indent=2)
    
    # Create a basic README.md/model card
    model_card = f"""---
language:
- sw
license: apache-2.0
tags:
- swahili
- text-generation
- causal-lm
datasets:
- custom-swahili-corpus
---

# {model_name}

Msingi1 is a decoder-only transformer language model optimized for Swahili text generation.

## Model Details

- **Model Type:** Decoder-only transformer (GPT-style)
- **Language:** Swahili
- **Vocabulary Size:** {config.vocab_size}
- **Hidden Size:** {config.n_embd}
- **Layers:** {config.n_layer}
- **Attention Heads:** {config.n_head}
- **Parameters:** ~{(config.n_layer * (12 * config.n_embd**2) + config.vocab_size * config.n_embd) // 1_000_000}M
- **Context Length:** {config.max_position_embeddings}
- **Training Data:** Custom Swahili corpus

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{hub_model_id or model_name}")
model = AutoModelForCausalLM.from_pretrained("{hub_model_id or model_name}")

# Generate text
input_text = "Habari ya leo"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

This model was trained on a custom Swahili corpus with approximately 40 million words.
The training used a causal language modeling objective with mixed precision and gradient checkpointing.

## Limitations

- The model was trained on a limited dataset and may not cover all domains or dialects of Swahili.
- As with all language models, it may generate incorrect or biased content.
- The model is best suited for general text generation and may not perform well on specialized tasks.

## Citation

If you use this model, please cite:

```
@misc{msingi1-swahili,
  author = {{Msingi AI Team}},
  title = {Msingi1: A Swahili Language Model},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/Msingi-AI/msingi1}}
}
```
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    print(f"Model successfully saved in HF format at {output_dir}")
    
    # Push to Hugging Face Hub if requested
    if push_to_hub:
        try:
            from huggingface_hub import HfApi
            
            if hub_token is None:
                raise ValueError("hub_token is required to push to Hugging Face Hub")
            
            model_id = hub_model_id or model_name
            api = HfApi(token=hub_token)
            
            print(f"Pushing model to Hugging Face Hub as {model_id}")
            api.create_repo(model_id, private=False, exist_ok=True)
            api.upload_folder(
                folder_path=output_dir,
                repo_id=model_id,
                commit_message="Upload model in Hugging Face format"
            )
            print(f"Model successfully pushed to https://huggingface.co/{model_id}")
        except ImportError:
            print("huggingface_hub package not installed. Please install it to push to Hub.")
        except Exception as e:
            print(f"Error pushing to Hub: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Msingi1 checkpoint to HF format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--model_name", type=str, default="msingi1-swahili", help="Model name")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--hub_token", type=str, help="Hugging Face API token")
    parser.add_argument("--hub_model_id", type=str, help="Model ID on Hub")
    
    args = parser.parse_args()
    
    save_model_hf_format(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        model_name=args.model_name,
        push_to_hub=args.push_to_hub,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id
    )
