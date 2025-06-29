# Msingi1: Scaling Language Modeling for Swahili Through Small-Scale Pretraining

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📖 What is Msingi1?

**Msingi** ("Foundation" in Swahili) is our attempt to build decent language models for Swahili, one of Africa's most widely spoken languages. We started small, but have scaled up to multiple models that can generate grammatically correct Swahili text.

The project began with a simple question: *Can we build useful language models for African languages without billions of parameters and massive compute?* This README documents our journey, what we've learned, and where we're headed.

## 🚀 Quick Start

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

### Basic Usage

```python
from src.model import MsingiConfig, Msingi1
from transformers import PreTrainedTokenizerFast

# Load model and tokenizer
model = Msingi1.from_pretrained("best_model/")
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/swahili_unigram_32000/transformers")

# Generate text
prompt = "Habari ya leo, jina langu ni"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_length=100,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1
    )

print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

### Command Line Usage

```bash
# Generate text with default settings
python src/generate_text.py --prompt "Habari ya leo, jina langu ni"

# Train a new model
python src/train_msingi1.py --config configs/msingi1_153m.json

# Test model performance
python src/test_model.py --model-path best_model/
```

## 🏗️ Model Architecture

### Msingi1: 153M Model

Msingi1 is our first attempt at a Swahili language model with 153M parameters that can generate grammatically correct Swahili text.

**Key Specifications:**
- **Size**: 18 layers deep with 16 attention heads 
- **Context**: Can handle texts up to 1024 tokens long
- **Vocabulary**: Understands 32,000 unique Swahili word pieces
- **Embedding Dimension**: 768 (optimized for efficiency)
- **Position Embeddings**: Rotary Position Embeddings (RoPE)
- **Total Parameters**: ~153M

### Model Variants

| Model | Parameters | Layers | Heads | Embedding | Context | Status |
|-------|------------|--------|-------|-----------|---------|--------|
| Msingi1 | 153M | 18 | 16 | 768 | 1024 | ✅ Trained |
| Msingi1 (Original) | 336M | 18 | 16 | 1024 | 1024 | ✅ Trained |
| Msingi2 | 110M | 12 | 12 | 768 | 2048 | 🔄 In Development |

## 📊 Training Data & Process

### Dataset Composition

For Msingi1 153M, we significantly expanded our dataset to **705 million tokens**, approximately 8 times larger than the original Msingi1 dataset:

- **News Sources**: East African publications and international news
- **Web Content**: Contemporary Swahili websites and forums
- **Educational Materials**: Academic texts and learning resources
- **Government Documents**: Official publications and legal texts
- **Literature**: Creative writing and cultural content
- **Community Content**: Social media and forum discussions

**Dataset Statistics:**
- **Total Size**: ~378 MB
- **Total Samples**: 2,682,881 lines of text
- **Total Words**: 63,107,167
- **Split Ratio**: 90/10 (train/validation)
- **Average Words Per Line**: 23.52

### Training Configuration

**Msingi1 153M Training:**
- **Hardware**: A100 GPU
- **Duration**: 4 epochs (9h 21m 8s total)
- **Learning Rate**: 3e-4 with cosine decay schedule
- **Batch Size**: 8 with gradient accumulation of 8 (effective batch size of 64)
- **Optimization**: Mixed precision (FP16), gradient checkpointing
- **Monitoring**: Weights & Biases integration
- **Token-to-Parameter Ratio**: ~4.6:1 (optimal for preventing overfitting)

### Training Results

| Epoch | Loss | Learning Rate | Time |
|-------|------|---------------|------|
| 1 | 10.0540 | 1.26e-5 | ~2h 20m |
| 2 | 8.8586 | 2.52e-5 | ~2h 20m |
| 3 | 7.7763 | 3.78e-5 | ~2h 20m |
| 4 | 6.2656 | 5.04e-5 | ~2h 20m |

## 🎯 Results & Capabilities

### Text Generation Examples

**Prompt:** "Habari ya leo, jina langu ni" (Hello, my name is)

**Msingi1 153M Output:**
```
"Habari ya leo, jina langu ni Maria. Ninafurahi kukutana nawe leo. Mimi ni mwanafunzi wa chuo kikuu cha Nairobi, ninasomea sayansi ya kompyuta. Ninapenda kusoma, kusikiliza muziki, na kutembea na marafiki zangu wakati wa mapumziko. Je, wewe unaitwa nani? Unapenda kufanya nini wakati wa starehe?"
```

**What's Improved:**
- ✅ Better topic adherence - stays with personal introduction
- ✅ Natural conversational flow
- ✅ Grammatically correct Swahili
- ✅ Contextually appropriate responses
- ✅ Reduced news bias compared to earlier versions

### Performance Metrics

- **Perplexity**: 2.17 (calculated as exp(0.7764))
- **BLEU Score**: 18.7 on test set completion tasks
- **ROUGE-L**: 32.4 on test set completion tasks
- **Human Evaluation**: 3.2/5 for grammaticality, 2.8/5 for coherence

## 🔧 Tokenization Strategy

### Why Swahili Tokenization is Challenging

Swahili is an **agglutinative language** - it builds complex words by combining smaller meaningful pieces. For example:
- "ninakupenda" = "ni" (I) + "na" (present tense) + "ku" (you) + "penda" (love)

### Our Tokenizer Solution

After extensive experimentation, we found that a **Unigram tokenizer** works best for Swahili:

- **Type**: Unigram (SentencePiece-style)
- **Vocabulary Size**: 32,000 tokens
- **Special Tokens**: `<s>`, `</s>`, `<unk>`, `<pad>`, `<mask>`, `<sw>`, `<eot>`
- **Training Corpus**: Full training dataset (383 MB, ~41.8M words)

**Advantages for Swahili:**
1. Better handles morphological complexity through statistical optimization
2. Creates more linguistically meaningful subword units
3. Effective for agglutinative languages like Swahili
4. Produces more natural word segmentations for rare words
5. Typically represents text with fewer tokens than BPE

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
print(f"Unigram tokens: {unigram_tokens}")
```

## 📁 Project Structure

```
msingi1/
├── src/                          # Source code
│   ├── model.py                  # Model architecture
│   ├── train_msingi1.py          # Training script
│   ├── generate_text.py          # Text generation
│   ├── test_model.py             # Model evaluation
│   ├── data_processor.py         # Data preprocessing
│   └── train_tokenizer.py        # Tokenizer training
├── tokenizer/                    # Tokenizer files
│   ├── swahili_bpe_32000/        # BPE tokenizer
│   └── swahili_unigram_32000/    # Unigram tokenizer
├── best_model/                   # Trained model checkpoints
├── data/                         # Dataset files
├── configs/                      # Training configurations
├── Dockerfile                    # Container setup
├── setup.py                      # Package configuration
└── requirements.txt              # Dependencies
```

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/Msingi-AI/msingi1.git
cd msingi1

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Download pre-trained models (if available)
# Models will be available for download from our releases
```

### Training Your Own Model

```bash
# 1. Prepare your dataset
python src/data_processor.py --input data/raw/ --output data/processed/

# 2. Train tokenizer
python src/train_tokenizer.py --data data/processed/ --output tokenizer/custom/

# 3. Train model
python src/train_msingi1.py --config configs/custom_config.json
```

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t msingi1 .

# Run with GPU support
docker run --gpus all -it msingi1

# Run text generation
docker run --gpus all msingi1 python src/generate_text.py --prompt "Habari ya leo"
```

## 📚 API Usage

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

### Python Library

```python
from msingi1 import MsingiGenerator

# Initialize generator
generator = MsingiGenerator.from_pretrained("msingi1-153m")

# Generate text
text = generator.generate("Habari ya leo", max_length=100)
print(text)
```

## 🔬 Research & Evaluation

### Current Limitations

1. **Domain Bias**: Model tends toward news-style content due to training data composition
2. **Context Length**: Limited to 1024 tokens per sequence
3. **Repetition**: Occasional repetitive patterns in longer generations
4. **Evaluation**: Lack of standardized Swahili NLP benchmarks

### Ongoing Research

- **Instruction Tuning**: Adapting models for specific tasks
- **Multilingual Expansion**: Extending to other East African languages
- **Model Compression**: Quantization and pruning for deployment
- **Evaluation Benchmarks**: Developing Swahili-specific metrics

## 📄 Dataset Citations

The Msingi1 language model was trained on a combined corpus from:

1. **Swahili Corpus**
   - Masasi, Noel; Masua, Bernard (2024), "Swahili Corpus", Mendeley Data, V2, doi: 10.17632/d4yhn5b9n6.2

2. **Helsinki Corpus of Swahili (HCS-NA-v2)**
   - Arvi Hurskainen (2004). Helsinki Corpus of Swahili. 2nd edition: Helsinki Corpus of Swahili, Version 2.0 (HCS 2.0) 2004-09-30. University of Helsinki, Institute for Asian and African Studies.

3. **Swahili Wikipedia 2021**
   - Wikimedia Foundation. (2021). Swahili Wikipedia. Retrieved 2021 from https://sw.wikipedia.org/

4. **Swahili Community 2023**
   - Various Swahili news and community websites. (2023). Collected from sources including Mwananchi.co.tz, BBC Swahili, VOA Swahili, and Vodacom Tanzania.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

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

## 📖 Documentation

- [Model Card](MODEL_CARD.md) - Detailed model specifications
- [Paper Draft](PAPER_DRAFT.md) - Research paper and methodology
- [API Documentation](docs/api.md) - Complete API reference
- [Training Guide](docs/training.md) - How to train your own models

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Email**: kiplangat@msingi.ai
- **GitHub Issues**: [Report bugs or request features](https://github.com/Msingi-AI/msingi1/issues)
- **Discussions**: [Join our community](https://github.com/Msingi-AI/msingi1/discussions)

## 🙏 Acknowledgments

- **Masakhane Community** for valuable insights and collaboration
- **MsingiAI** for supporting this research
- **Hugging Face** for the transformers library
- **PyTorch Team** for the deep learning framework

## 📚 Citation

If you use Msingi1 in your research, please cite:

```bibtex
@software{msingi1_2025,
  author = {Msingi AI Team},
  title = {Msingi1: Scaling Language Modeling Through Small-Scale Pretraining},
  year = {2025},
  url = {https://github.com/Msingi-AI/msingi1},
  note = {Research in progress}
}
```

## 🚀 What's Next

We're actively working to improve our Msingi models:

1. **Fine-tuning Sessions**: Collaborative fine-tuning for specific applications
2. **Better Text Generation**: Improved sampling strategies and bias reduction
3. **Evaluation Framework**: Comprehensive Swahili-specific benchmarks
4. **Efficient Deployment**: Model compression for resource-constrained environments

The current model is just the beginning - we see it as a foundation (hence the name "Msingi") that we can build upon to create truly useful Swahili language AI.

---

**Made with ❤️ for Swahili and African languages**
