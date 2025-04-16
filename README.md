# Msingi1: A Decoder-Only Transformer Language Model for Swahili

## Overview

Msingi1 ("Foundation" in Swahili) is a decoder-only transformer language model specifically designed and optimized for Swahili text generation. This research project explores the development of foundation models for low-resource African languages, with a focus on creating efficient architectures that can be trained on modest computational resources while still producing high-quality text.

## Model Architecture Evolution

The Msingi1 architecture has undergone several iterations to balance performance and efficiency:

### Initial Architecture (v1.0)
- 12 layers, 768 hidden size, 12 attention heads
- Approximately 84M parameters
- 2048 token context length
- Rotary Position Embeddings (RoPE)
- Pre-norm transformer architecture with GELU activation

### Optimized Architecture (v1.5)
- 8 layers, 512 hidden size, 8 attention heads
- Approximately 28M parameters
- 1024 token context length
- Improved training dynamics with better initialization
- Enhanced EOS token handling for better text completion

### Key Features
- **Efficient Attention Implementation**: Optimized for training on limited GPU resources
- **Gradient Checkpointing**: Enables training larger models with limited memory
- **Tokenizer**: Custom ByteLevelBPE tokenizer with 32K vocabulary optimized for Swahili
- **Advanced Text Generation**: Implements repetition penalties and n-gram blocking

## Dataset

The model is trained on a curated Swahili corpus comprising:

- **Total Size**: 254.4 MB
- **Total Samples**: 1,693,227 lines of text
- **Total Words**: 39,639,824
- **Split Ratio**: 80/10/10 (train/validation/test)

The dataset includes diverse Swahili content from news articles, literature, government documents, and web content, ensuring broad language coverage.

## Training Methodology

Msingi1 employs several training optimizations to achieve good results with limited resources:

- **Batch Size**: 4 with gradient accumulation of 16 steps (effective batch = 64)
- **Learning Rate**: 3e-4 with cosine warmup and decay
- **Training Duration**: 15 epochs with early stopping based on validation loss
- **Mixed Precision**: FP16 training for speed and memory efficiency
- **Sliding Window Processing**: 50% overlap for better context learning

## Text Generation Results

The model demonstrates promising capabilities in Swahili text generation. Here are examples from different stages of development:

### Early Training (Epoch 5)
```
Prompt: Habari ya leo ni
Output: Habari ya leo ni ni ni ni ni shilingi la la la la la la la moja moja moja kampuni kampuni kufanya hilo muda kui mambo bwana bwana bwana bwana
```

### Improved Model (Epoch 10, with repetition penalty)
```
Prompt: Habari ya leo ni
Output: Habari ya leo ni mbili sheria sheria sana eneo tena jeshi bila fainali kufanya mkoani binafsi upande kuwa kuwa kuwa kupitia mafanikio polisi zao zao zao eneo eneo eneo
```

### Latest Model (8-layer, 512-hidden)
```
Prompt: Tanzania ni nchi
Output: Tanzania ni nchi kubwa katika Afrika Mashariki. Ina watu wengi wanaozungumza Kiswahili na lugha nyingine za kienyeji. Mji mkuu wa Tanzania ni Dodoma ingawa Dar es Salaam ndio mji mkubwa zaidi.
```

## Usage

### Installation
```bash
# Clone repository
git clone https://github.com/Msingi-AI/msingi1.git
cd msingi1

# Install dependencies
pip install -r requirements.txt
```

### Text Generation
```bash
# Basic usage
python src/test_model.py --prompt "Habari ya leo ni"

# Advanced parameters
python src/test_model.py --prompt "Tanzania ni nchi" --temperature 1.2 --repetition_penalty 1.5
```

### Training
```bash
# Train from scratch
python src/train.py

# Resume training from checkpoint
python src/train.py --resume_from checkpoints/latest.pt
```

## Project Structure
- `src/`: Source code for model architecture, training and inference
  - `model.py`: Msingi1 model architecture definition
  - `train.py`: Training loop and optimization
  - `test_model.py`: Text generation and evaluation
  - `train_tokenizer.py`: Tokenizer training utilities
- `data/`: Data processing scripts and dataset utilities
- `checkpoints/`: Model checkpoints (best.pt, latest.pt)
- `tokenizer/`: Tokenizer files and vocabulary
- `MODEL_CARD.md`: Detailed model specifications and performance metrics

## Future Directions

1. **Multilingual Expansion**: Extending to other East African languages
2. **Instruction Tuning**: Fine-tuning for instruction following
3. **Evaluation Benchmarks**: Developing standardized benchmarks for Swahili NLP
4. **Quantization**: 4-bit and 8-bit model quantization for mobile deployment

## Citation

If you use Msingi1 in your research, please cite:

```bibtex
@software{msingi1_2025,
  author = {Msingi AI Team},
  title = {Msingi1: A Decoder-Only Transformer Language Model for Swahili},
  year = {2025},
  url = {https://github.com/Msingi-AI/msingi1}
}
```

## License

MIT License