# Msingi1 Model Card

## Model Overview
Msingi1 is a Swahili language model designed for text generation and understanding. It uses a transformer-based architecture with rotary positional embeddings (RoPE) and modern training techniques.

## Model Architecture

### Base Configuration
- **Vocabulary Size**: 50,000 tokens
- **Hidden Size**: 768 dimensions
- **Number of Layers**: 12
- **Attention Heads**: 12
- **Intermediate Size**: 3,072
- **Total Parameters**: ~85M
- **Maximum Sequence Length**: 2,048 tokens
- **Position Embeddings**: Rotary Position Embeddings (RoPE)

### Key Components
1. **Token Embeddings**: 
   - Dimension: 768
   - Shared with output layer (weight tying)

2. **Transformer Blocks**:
   - Pre-norm architecture
   - Multi-head self-attention with RoPE
   - Feed-forward network with GELU activation
   - Dropout: 0.1
   - Layer normalization epsilon: 1e-5

3. **Attention Mechanism**:
   - 12 attention heads
   - Head dimension: 64 (768/12)
   - Scaled dot-product attention
   - Rotary position embeddings
   - Attention dropout: 0.1

## Training Details

### Training Configuration
- **Batch Size**: 4
- **Gradient Accumulation Steps**: 16 (effective batch size: 64)
- **Learning Rate**: 3e-4
- **Weight Decay**: 0.1
- **Warmup Steps**: 1,000
- **Training Sequence Length**: 1,024
- **Optimizer**: AdamW
- **Learning Rate Schedule**: Cosine with warmup
- **Mixed Precision**: FP16 training enabled
- **Gradient Checkpointing**: Supported

### Memory Optimizations
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)
- Configurable sequence length
- Memory-efficient attention implementation

## Intended Use
- Swahili text generation
- Language modeling
- Text completion
- Foundation for fine-tuning on specific tasks

## Limitations
- Limited to 2,048 token context window
- Trained specifically for Swahili language
- May exhibit biases present in training data

## Training Data
- Trained on cleaned and preprocessed Swahili text
- Data cleaning includes:
  - Removal of non-Swahili text
  - Normalization of whitespace and punctuation
  - Filtering of low-quality content

## Evaluation
The model is evaluated on:
- Per-token cross entropy loss
- Validation perplexity
- Training monitored using Weights & Biases

## Training Results
Training results will be updated once training is completed.

## License
[Add license information]

## Citation
[Add citation information]

## Contact
[Add contact information]
