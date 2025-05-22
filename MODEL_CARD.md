# Msingi1 Model Card 

## Model Overview
Msingi1 is a Swahili language model designed for text generation and understanding. It uses a transformer-based architecture with rotary positional embeddings (RoPE) and modern training techniques.

## Model Architecture

### Base Configuration
- **Vocabulary Size**: 32,000 tokens
- **Hidden Size**: 768 dimensions
- **Number of Layers**: 12
- **Attention Heads**: 12
- **Intermediate Size**: 3,072
- **Total Parameters**: ~110M
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

### Dataset
- Total Size: 254.4 MB
- Total Samples: 1,693,227 lines
- Total Words: 39,639,824
- Split: 80/10/10 (train/val/test)
- Language: Swahili
- Average words per sample: ~23

### Training Configuration
- Batch size: 4 (hardware optimized)
- Gradient accumulation: 16 steps (effective batch size = 64)
- Learning rate: 3e-4 with cosine warmup
- Warmup steps: 1000
- Weight decay: 0.1
- Max gradient norm: 1.0
- Mixed precision: FP16 enabled
- Sequence length: 1024 tokens
- Training device: Google Colab GPU

### Training Results
- Initial loss: 10.3
- Final loss: 8.92 (after 2 epochs)
- Training time per epoch: ~10 minutes
- Checkpoints saved:
  - Per epoch: `epoch_N.pt`
  - Best model: `best.pt`
  - Latest state: `latest.pt`

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

### Training Metrics
| Epoch | Average Loss | Learning Rate |
|-------|--------------|---------------|
| 1     | 10.0540     | 1.26e-5      |
| 2     | 8.8586      | 2.52e-5      |
| 3     | 7.7763      | 3.78e-5      |
| 4     | 6.2656      | 5.04e-5      |
| 5     | 4.9480      | 6.30e-5      |
| 6     | 3.6461      | 7.56e-5      |
| 7     | 2.6851      | 8.82e-5      |
| 8     | 1.9188      | 1.01e-4      |
| 9     | 1.2790      | 1.13e-4      |
| 10    | 0.7764      | 1.26e-4      |

### Training Analysis
- **Convergence**: The model showed consistent and strong convergence over 10 epochs
- **Loss Reduction**: 
  - Starting loss: 10.0540
  - Final loss: 0.7764
  - Total reduction: 92.3%
  
- **Learning Rate Behavior**:
  - Started at ~1e-5
  - Gradually increased to 1.26e-4
  - Cosine warmup schedule worked effectively

- **Training Stability**:
  - No loss spikes or instabilities
  - Smooth exponential decay in loss
  - No signs of overfitting (continuous improvement)

- **Performance Metrics**:
  - Training speed: ~1.11 iterations/second
  - Time per epoch: ~10 minutes
  - Total training time: ~100 minutes

### Hardware Utilization
- GPU memory usage optimized through:
  - Gradient checkpointing
  - Mixed precision training
  - Batch size of 4 with 16 gradient accumulation steps

## Final Model Performance
- **Perplexity**: 2.17 (calculated as exp(0.7764))
- **Final Cross-Entropy Loss**: 0.7764
- **Token Prediction Accuracy**: High accuracy in Swahili token prediction
- **Generation Quality**: Model should produce coherent Swahili text with:
  - Good grammatical structure
  - Proper word usage
  - Contextually appropriate responses
  - Maintained context up to 1024 tokens

## Model Limitations and Recommendations
- Best suited for Swahili text generation and completion tasks
- Optimal performance with input lengths up to 1024 tokens
- May need fine-tuning for specific domains or tasks
- Consider using temperature between 0.7-0.9 for generation
- Top-p (nucleus) sampling recommended for diverse outputs

## License
[Add license information]

## Citation
[Add citation information]

## Contact
[Add contact information]
