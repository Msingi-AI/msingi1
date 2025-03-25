![image](https://github.com/user-attachments/assets/e1a5e701-c03b-4a40-a032-9d472c1991ea)


# Msingi1: A Swahili Language Model with Mixture of Experts

Msingi1 is an advanced Swahili language model that leverages the Mixture of Experts (MoE) architecture for efficient and scalable natural language processing. Built with PyTorch and FastMoE, it's designed to provide high-quality text generation and understanding for the Swahili language.

## Features

- **Mixture of Experts Architecture**
  - 8 expert networks for specialized language processing
  - MoE layers at strategic positions (2 and 4)
  - Efficient routing using NaiveGate
  - Expert capacity of 32 tokens

- **Model Architecture**
  - Hidden size: 768
  - Intermediate size: 3072
  - 12 attention heads
  - Maximum sequence length: 1024 tokens
  - ByteLevelBPE tokenizer with 32k vocabulary

- **Training Optimizations**
  - Sliding window tokenization with configurable stride
  - Efficient checkpointing system
  - Google Colab integration
  - Gradient accumulation support

## Project Structure

```
msingi1/
├── src/
│   ├── model.py          # Core model architecture with MoE
│   ├── data_processor.py # Dataset and data handling
│   └── __init__.py      # Package initialization
├── tokenizer/
│   ├── vocab.json       # BPE vocabulary (32k tokens)
│   └── merges.txt       # BPE merge rules
├── notebooks/
│   └── Msingi1_Training.ipynb  # Colab training notebook
├── setup.py             # Package installation
├── setup_colab.py       # Colab environment setup
└── requirements.txt     # Project dependencies
```

## Installation

### Local Development
```bash
# Clone the repository
git clone https://github.com/your-username/msingi1.git
cd msingi1

# Install dependencies
pip install -e .
```

### Google Colab
1. Open the `Msingi1_Training.ipynb` notebook in Google Colab
2. Run the setup cells:
```python
!git clone https://github.com/your-username/msingi1.git
%cd msingi1
!python setup_colab.py
```

## Dataset

The model is trained on a curated Swahili text dataset:
- Size: 7.7 MB of text data
- Words: 1.4 million
- Characters: 9 million
- Format: Plain text and JSON

## Training

### Hardware Requirements
- GPU: Google Colab's T4/P100 GPU (sufficient for training)
- RAM: 12GB+ recommended
- Storage: 10GB+ for model checkpoints

### Training Process
1. Data preprocessing with sliding window tokenization
2. Model training with gradient accumulation
3. Regular checkpointing to Google Drive
4. Progress tracking with Weights & Biases

### Hyperparameters
- Batch size: Configurable based on GPU memory
- Learning rate: 5e-5 with warmup
- Training epochs: 50
- Checkpoint frequency: Every 5 epochs

## Usage

```python
from src.model import Msingi1, MsingiConfig
from tokenizers import ByteLevelBPETokenizer

# Load tokenizer
tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# Initialize model
config = MsingiConfig()
model = Msingi1(config)

# Generate text
input_text = "Habari ya leo?"
encoded = tokenizer.encode(input_text)
output_ids = model.generate(
    input_ids=encoded.ids,
    max_length=100,
    num_return_sequences=1
)
output_text = tokenizer.decode(output_ids[0])
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastMoE team for the efficient MoE implementation
- PyTorch team for the deep learning framework
- Google Colab for providing free GPU resources
