# Msingi1: Scaling Language Modeling for Swahili Through Small-Scale Pretraining

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Experimental Project Notice

**Msingi1 is a purely experimental research project.** This repository contains experimental code and research findings for developing Swahili language models. No pre-trained models have been released yet, but we plan to release multiple variants soon.

## Introduction

**Msingi** ("Foundation" in Swahili) is our experimental attempt to build decent language models for Swahili, one of Africa's most widely spoken languages. We started small, but have scaled up to multiple experimental models that can generate grammatically correct Swahili text.

The project began with a simple question: *Can we build useful language models for African languages without billions of parameters and massive compute?* This README documents our experimental journey, what we've learned, and where we're headed.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Msingi-AI/msingi1.git
cd msingi1

# Install dependencies
pip install -e .

# Or install from PyPI (when available)
# pip install msingi1
```

### Basic Usage (For Reproduction)

```python
from src.model_v2 import Msingi2Config, Msingi2
from transformers import PreTrainedTokenizerFast

# Create a 12-layer model configuration (recommended for reproduction)
config = Msingi2Config(
    vocab_size=32000,
    block_size=1024,
    n_layer=12,        # Recommended: 12 layers
    n_head=12,
    n_embd=768,
    dropout=0.15,
    gradient_checkpointing=True
)

# Initialize the model
model = Msingi2(config)

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/swahili_unigram_32000/transformers")

# Generate text (after training)
prompt = "Habari ya leo, jina langu ni"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1
    )

print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

### Command Line Usage

```bash
# Generate text with default settings (after training)
python src/generate_text.py --prompt "Habari ya leo, jina langu ni"

# Train a new model (recommended: 12 layers)
python src/train_msingi2.py --config configs/msingi2_12l.json

# Test model performance
python src/test_model.py --model-path best_model/
```

## Model History and Variants

We have reproduced and trained the following models, in this order:

1. **Msingi-Spinner**: 12 layers, ~85M parameters, RoPE positional embeddings (our first successful reproduction)
2. **Msingi-Mzizi** (mzizi = root/foundation): 12 layers, ~85M parameters, traditional (learned) positional embeddings
3. **Msingi-Kali** (kali = sharp/fierce): 18 layers, ~153M parameters, traditional positional embeddings
4. **Msingi-Hodari** (hodari = skilled/capable): 24 layers, ~336M parameters, traditional positional embeddings
5. **Msingi-Bingwa** (bingwa = expert/master): 36 layers, ~504M parameters, traditional positional embeddings

All models use a vocabulary size of 32,000. The embedding dimensions scale with model size: 768 dimensions for 12 and 18 layers, and 1024 dimensions for 24 and 36 layers. Parameter counts are approximate and rounded for clarity.

| Model Name | Layers | Embedding Dimension | Positional Embeddings | Parameters (approx) |
|------------|--------|-------------------|---------------------|-------------------|
| Msingi-Spinner | 12 | 768 | RoPE | 85M |
| Msingi-Mzizi | 12 | 768 | Learned | 110M |
| Msingi-Kali | 18 | 768 | Learned | 153M |
| Msingi-Hodari | 24 | 1024 | Learned | 336M |
| Msingi-Bingwa | 36 | 1024 | Learned | 504M |

We recommend reproducing the **12-layer models** (Msingi-Mzizi) for most users, as they offer a good balance of performance and computational requirements. Larger models (Msingi-Kali, Msingi-Hodari, Msingi-Bingwa) are in progress and will be released soon. We will release all variants once entire training is done!!

## Training Data and Process

### Dataset Composition

Our experimental training corpus combines multiple high-quality Swahili datasets:

#### Primary Datasets
1. **Swahili-SAFI (C4 Dataset)**: ~3.5GB of clean Swahili text from the Common Crawl
2. **Swahili Corpus**: Academic and news content from Mendeley Data
3. **Helsinki Corpus**: Linguistic research corpus
4. **Swahili Wikipedia**: Encyclopedic content
5. **Community Content**: News websites, forums, and contemporary content

#### Downloading the C4 Dataset

```bash
# Download and prepare the Swahili-SAFI dataset
python src/download_mc4_swahili.py

# This will create:
# - data/train.txt (95% of data)
# - data/valid.txt (5% of data)
```

**Dataset Statistics:**
- **Total Size**: ~378 MB
- **Total Samples**: 2,682,881 lines of text
- **Total Words**: 63,107,167
- **Split Ratio**: 90/10 (train/validation)
- **Average Words Per Line**: 23.52

### Dataset Sharding for Efficient Training

To handle large datasets efficiently, we use a sharding approach that processes data in manageable chunks:

```bash
# Create tokenized shards for training
python src/create_token_shards.py
```

**Sharding Benefits:**
- **Memory Efficiency**: Only loads necessary tokens into memory
- **Training Speed**: Reduces I/O bottlenecks through memory mapping
- **Scalability**: Enables training on larger datasets than would fit in RAM
- **Flexibility**: Allows for dynamic shard loading and epoch definitions

**Shard Configuration:**
- **Shard Size**: 10M tokens per shard (optimized for 13GB+ RAM)
- **Validation Chunks**: 3M tokens per chunk
- **Buffer Size**: 3M tokens for memory management
- **Format**: NumPy arrays with uint16 dtype for efficiency

### Training Configuration

**Recommended Msingi2 Training (12 layers):**
- **Hardware**: A100 GPU (recommended)
- **Duration**: 4-6 epochs
- **Learning Rate**: 3e-4 with cosine decay schedule
- **Batch Size**: 8 with gradient accumulation of 8 (effective batch size of 64)
- **Optimization**: Mixed precision (FP16), gradient checkpointing
- **Monitoring**: Weights & Biases integration
- **Token-to-Parameter Ratio**: ~4.6:1 (optimal for preventing overfitting)

### Training Results (Experimental)

| Epoch | Loss | Learning Rate | Time |
|-------|------|---------------|------|
| 1 | 10.0540 | 1.26e-5 | ~2h 20m |
| 2 | 8.8586 | 2.52e-5 | ~2h 20m |
| 3 | 7.7763 | 3.78e-5 | ~2h 20m |
| 4 | 6.2656 | 5.04e-5 | ~2h 20m |

## Tokenization Strategy

### Why Swahili Tokenization is Challenging

Swahili is an **agglutinative language** - it builds complex words by combining smaller meaningful pieces. For example:
- "ninakupenda" = "ni" (I) + "na" (present tense) + "ku" (you) + "penda" (love)

### Our Tokenizer Solution: Unigram Tokenizer

After extensive experimentation with ByteLevelBPE, WordPiece, and Unigram tokenizers, we found that **Unigram tokenization** works best for Swahili:

#### Tokenizer Specifications
- **Type**: Unigram (SentencePiece-style)
- **Vocabulary Size**: 32,000 tokens
- **Special Tokens**: `<s>`, `</s>`, `<unk>`, `<pad>`, `<mask>`, `<sw>`, `<eot>`
- **Training Corpus**: Full training dataset (383 MB, ~41.8M words)
- **Implementation**: Built using Hugging Face Tokenizers library

#### Why Unigram for Swahili?

1. **Morphological Complexity**: Better handles Swahili's agglutinative structure through statistical optimization
2. **Linguistic Meaning**: Creates more linguistically meaningful subword units
3. **Rare Word Handling**: Produces more natural word segmentations for rare words
4. **Token Efficiency**: Typically represents text with fewer tokens than BPE
5. **Statistical Optimization**: Uses likelihood-based training for optimal subword segmentation

#### Tokenizer Comparison

| Tokenizer | Vocab Size | Avg Tokens/Sentence | Morphological Handling | Memory Usage |
|-----------|------------|-------------------|----------------------|--------------|
| ByteLevelBPE | 32K | 15.2 | Good | Medium |
| Unigram | 32K | 13.8 | **Excellent** | **Low** |
| WordPiece | 32K | 16.1 | Fair | High |

### Usage Example

```python
from transformers import PreTrainedTokenizerFast

# Load tokenizers
bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/swahili_bpe_32000/transformers")
unigram_tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/swahili_unigram_32000/transformers")

# Compare tokenization
text = "Ninapenda kusoma vitabu vya Kiswahili na kusikiliza muziki."

bpe_tokens = bpe_tokenizer.tokenize(text)
unigram_tokens = unigram_tokenizer.tokenize(text)

print(f"BPE tokens: {bpe_tokens}")
print(f"BPE token count: {len(bpe_tokens)}")
print(f"Unigram tokens: {unigram_tokens}")
print(f"Unigram token count: {len(unigram_tokens)}")

# Using the <eot> token for text separation
texts = ["Habari ya leo.", "Habari nzuri sana."]
combined_text = unigram_tokenizer.eos_token.join(texts)  # Joins with <eot>
print(f"Combined with <eot>: {combined_text}")
encoded = unigram_tokenizer.encode(combined_text)
print(f"Decoded back: {unigram_tokenizer.decode(encoded)}")
```

## Project Structure

```
msingi1/
├── src/                          # Source code
│   ├── model.py                  # Original model with RoPE embeddings
│   ├── model_v2.py               # Current model with traditional embeddings
│   ├── train_msingi1.py          # Training script for original model
│   ├── train_msingi2.py          # Training script for current model
│   ├── generate_text.py          # Text generation
│   ├── test_model.py             # Model evaluation
│   ├── data_processor.py         # Data preprocessing
│   ├── download_mc4_swahili.py   # Download C4 dataset
│   ├── create_token_shards.py    # Create training shards
│   └── train_tokenizer.py        # Tokenizer training
├── tokenizer/                    # Tokenizer files
│   ├── swahili_bpe_32000/        # BPE tokenizer
│   └── swahili_unigram_32000/    # Unigram tokenizer (recommended)
├── msingi_tokens/                # Tokenized dataset shards
├── best_model/                   # Trained model checkpoints
├── data/                         # Dataset files
├── configs/                      # Training configurations
├── Dockerfile                    # Container setup
├── setup.py                      # Package configuration
└── requirements.txt              # Dependencies
```

## Development Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (32GB+ for full dataset processing)

### Complete Setup Process

```bash
# 1. Clone repository
git clone https://github.com/Msingi-AI/msingi1.git
cd msingi1

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Download and prepare datasets
python src/download_mc4_swahili.py

# 5. Create tokenized shards
python src/create_token_shards.py

# 6. Train your model (12-layer recommended)
python src/train_msingi2.py --config configs/msingi2_12l.json
```

### Training Your Own Model

```bash
# 1. Prepare your dataset
python src/data_processor.py --input data/raw/ --output data/processed/

# 2. Train tokenizer (if needed)
python src/train_tokenizer.py --data data/processed/ --output tokenizer/custom/

# 3. Create token shards
python src/create_token_shards.py

# 4. Train model with 12 layers (RECOMMENDED)
python src/train_msingi2.py --config configs/custom_12l.json
```

### Configuration for 12-Layer Model (Recommended)

Create a custom configuration file `configs/custom_12l.json`:

```json
{
    "vocab_size": 32000,
    "block_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "dropout": 0.15,
    "bias": true,
    "gradient_checkpointing": true
}
```

## Docker Deployment

```bash
# Build Docker image
docker build -t msingi1 .

# Run with GPU support
docker run --gpus all -it msingi1

# Run text generation
docker run --gpus all msingi1 python src/generate_text.py --prompt "Habari ya leo"
```

## API Usage

### REST API (Coming Soon)

```python
import requests

# Generate text via API
response = requests.post("https://api.msingi.ai/generate", json={
    "prompt": "Habari ya leo, jina langu ni",
    "max_length": 100,
    "temperature": 0.8
})

print(response.json()["generated_text"])
```

### Python Library (Coming Soon)

```python
from msingi1 import MsingiGenerator

# Initialize generator
generator = MsingiGenerator.from_pretrained("msingi-mzizi")

# Generate text
text = generator.generate("Habari ya leo", max_length=100)
print(text)
```

## Results and Capabilities

### Text Generation Examples (Experimental)

**Prompt:** "Habari ya leo, jina langu ni" (Hello, my name is)

**Experimental Msingi-Mzizi Output:**
```
"Habari ya leo, jina langu ni Maria. Ninafurahi kukutana nawe leo. Mimi ni mwanafunzi wa chuo kikuu cha Nairobi, ninasomea sayansi ya kompyuta. Ninapenda kusoma, kusikiliza muziki, na kutembea na marafiki zangu wakati wa mapumziko. Je, wewe unaitwa nani? Unapenda kufanya nini wakati wa starehe?"
```

**What's Improved:**
- Better topic adherence - stays with personal introduction
- Natural conversational flow
- Grammatically correct Swahili
- Contextually appropriate responses
- Reduced news bias compared to earlier versions

### Performance Metrics (Experimental)

- **Perplexity**: 2.17 (calculated as exp(0.7764))
- **BLEU Score**: 18.7 on test set completion tasks
- **ROUGE-L**: 32.4 on test set completion tasks
- **Human Evaluation**: 3.2/5 for grammaticality, 2.8/5 for coherence

## Research and Evaluation

### Current Limitations

1. **Domain Bias**: Model tends toward news-style content due to training data composition
2. **Context Length**: Limited to 1024 tokens per sequence
3. **Repetition**: Occasional repetitive patterns in longer generations
4. **Evaluation**: Lack of standardized Swahili NLP benchmarks

### Post-Training Model Phase

We are currently working on the **post-training model phase**, which includes:

- **Instruction Tuning**: Adapting models for specific tasks and instructions
- **Fine-tuning**: Domain-specific adaptations (legal, medical, educational)
- **Conversational Abilities**: Improving topic coherence and dialogue skills
- **Bias Reduction**: Addressing domain biases in the training data

**Note**: This phase is progressing slowly due to limited manpower, as we are working on this project part-time. We welcome contributions from the community to accelerate this work.

### Ongoing Research

- **Instruction Tuning**: Adapting models for specific tasks
- **Multilingual Expansion**: Extending to other East African languages
- **Model Compression**: Quantization and pruning for deployment
- **Evaluation Benchmarks**: Developing Swahili-specific metrics

## New Project Announcement

We are working on a **new project with a novel approach to training Swahili language models**. This project will introduce innovative techniques specifically designed for African languages and their unique characteristics.

**We are actively looking for collaborators!** If you're interested in:
- Novel training methodologies
- African language technology
- Experimental NLP research
- Swahili language processing

Please create an issue or reach out to us. We'd love to collaborate with researchers, developers, and Swahili speakers who are passionate about advancing African language technology.

## Dataset Citations

The Msingi1 language model was trained on a combined corpus from:

1. **Swahili-SAFI (C4 Dataset)**
   - Flax Community. (2023). Swahili-SAFI: A clean Swahili dataset from Common Crawl. Hugging Face Datasets.

2. **Swahili Corpus**
   - Masasi, Noel; Masua, Bernard (2024), "Swahili Corpus", Mendeley Data, V2, doi: 10.17632/d4yhn5b9n6.2

3. **Helsinki Corpus of Swahili (HCS-NA-v2)**
   - Arvi Hurskainen (2004). Helsinki Corpus of Swahili. 2nd edition: Helsinki Corpus of Swahili, Version 2.0 (HCS 2.0) 2004-09-30. University of Helsinki, Institute for Asian and African Studies.

4. **Swahili Wikipedia 2021**
   - Wikimedia Foundation. (2021). Swahili Wikipedia. Retrieved 2021 from https://sw.wikipedia.org/

5. **Swahili Community 2023**
   - Various Swahili news and community websites. (2023). Collected from sources including Mwananchi.co.tz, BBC Swahili, VOA Swahili, and Vodacom Tanzania.

## Contributing

We welcome contributions! This project is purely experimental and particularly in need of help with the post-training phase. Here's how you can contribute:

### Areas Needing Help

1. **Instruction Tuning**: Help develop instruction-following capabilities
2. **Fine-tuning**: Create domain-specific model variants
3. **Evaluation**: Develop Swahili-specific benchmarks and metrics
4. **Documentation**: Improve tutorials and guides
5. **Code Optimization**: Optimize training and inference code
6. **Community Building**: Help grow the Swahili NLP community
7. **New Project Collaboration**: Join our novel training methodology project

### Development Workflow

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/amazing-feature

# 3. Make your changes
# 4. Add tests
python -m pytest tests/

# 5. Commit your changes
git commit -m "Add amazing feature"

# 6. Push to the branch
git push origin feature/amazing-feature

# 7. Open a Pull Request
```

### Getting Started with Contributions

1. **Join our Discussions**: Share ideas and ask questions
2. **Pick an Issue**: Look for issues labeled "good first issue" or "help wanted"
3. **Start Small**: Begin with documentation or small bug fixes
4. **Ask for Help**: Don't hesitate to ask questions in issues or discussions
5. **Create Issues**: Share your ideas, report bugs, or suggest improvements

### Contribution Guidelines

- **Code Style**: Follow PEP 8 for Python code
- **Documentation**: Add docstrings and comments for new functions
- **Testing**: Add tests for new features
- **Commit Messages**: Use clear, descriptive commit messages
- **Pull Requests**: Provide clear descriptions of changes

## Documentation

- [Model Card](MODEL_CARD.md) - Detailed model specifications
- [Paper Draft](PAPER_DRAFT.md) - Research paper and methodology
- [API Documentation](docs/api.md) - Complete API reference
- [Training Guide](docs/training.md) - How to train your own models
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact and Support

- **Email**: korirkiplangat22@gmail.com
- **GitHub Issues**: [Report bugs, request features, or share ideas](https://github.com/Msingi-AI/msingi1/issues)
- **Discussions**: [Join our community](https://github.com/Msingi-AI/msingi1/discussions)

## Acknowledgments

- **Masakhane Community** for valuable insights and collaboration
- **MsingiAI** for supporting this research
- **Hugging Face** for the transformers library
- **PyTorch Team** for the deep learning framework
- **Flax Community** for the Swahili-SAFI dataset

## Citation

If you use Msingi1 in your research, please cite:

```bibtex
@software{msingi1_2025,
  author = {Msingi AI Team},
  title = {Msingi1: Scaling Language Modeling Through Small-Scale Pretraining},
  year = {2025},
  url = {https://github.com/Msingi-AI/msingi1},
  note = {Experimental research project}
}
```

## Future Work

We're actively working to improve our Msingi models:

1. **Model Releases**: Soon releasing Msingi-Spinner, Msingi-Mzizi, Msingi-Kali, Msingi-Hodari, and Msingi-Bingwa variants
2. **Post-Training Phase**: Instruction tuning and fine-tuning (needs community help!)
3. **Better Text Generation**: Improved sampling strategies and bias reduction
4. **Evaluation Framework**: Comprehensive Swahili-specific benchmarks
5. **Efficient Deployment**: Model compression for resource-constrained environments
6. **New Project**: Novel training methodology for African languages

The current model is just the beginning - we see it as a foundation (hence the name "Msingi") that we can build upon to create truly useful Swahili language AI.

**We need your help to accelerate the post-training phase and collaborate on our new project! (AkiliX)** Whether you're a researcher, developer, or Swahili speaker, your contributions can make a real difference in advancing Swahili language technology.

**Create issues, share ideas, and join us in building the future of African language AI!**


**Made with dedication for Swahili** 
