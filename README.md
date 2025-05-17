# Msingi1: Scaling Language Modeling for Swahili Through Small-Scale Pretraining
 
## Research Overview

Msingi1 ("Foundation" in Swahili) documents our ongoing research journey in developing decoder-only transformer language models for Swahili. This project is an experimental exploration of foundation models for low-resource African languages, with a focus on understanding the trade-offs between model size, training efficiency, and text generation quality.

> **Note**: This project is a work-in-progress research effort and not yet ready for production use. We're sharing our journey to contribute to the growing body of knowledge on African language NLP.

## Architectural Exploration

Throughout this research journey, we've experimented with different model architectures:

### Initial Exploration (v1.0)
- 12 layers, 768 hidden size, 12 attention heads
- Approximately 84M parameters
- 2048 token context length
- Rotary Position Embeddings (RoPE)
- Pre-norm transformer architecture with GELU activation

### Current Experiments (v1.5)
- 8 layers, 512 hidden size, 8 attention heads
- Approximately 28M parameters
- 1024 token context length
- Improved training dynamics with better initialization
- Enhanced EOS token handling for better text completion

### Research Focus Areas
- **Efficient Attention Mechanisms**: Exploring optimizations for limited GPU resources
- **Gradient Checkpointing**: Testing memory-efficient training approaches
- **Tokenization Strategies**: Investigating ByteLevelBPE tokenization with 32K vocabulary for Swahili
- **Text Generation Techniques**: Experimenting with repetition penalties and n-gram blocking

## Dataset Characteristics

Our experiments use a comprehensive Swahili corpus with the following characteristics:

- **Total Size**: ~378 MB
- **Total Samples**: 2,682,881 lines of text
- **Total Words**: 63,107,167
- **Split Ratio**: 90/10 (train/validation)
- **Average Words Per Line**: 23.52

The dataset includes diverse Swahili content from:
- News articles and publications
- Literature and books
- Government documents and parliamentary proceedings
- Wikipedia articles
- Contemporary community content and FAQs
- Mobile service documentation
- Web content
- Biblical text (Swahili Bible)

### Dataset Citations

The Msingi1 language model was trained on a combined corpus created from the following sources:

1. **Swahili Corpus**
   - Masasi, Noel; Masua, Bernard (2024), "Swahili Corpus", Mendeley Data, V2, doi: 10.17632/d4yhn5b9n6.2

2. **Helsinki Corpus of Swahili (HCS-NA-v2)**
   - Arvi Hurskainen (2004). Helsinki Corpus of Swahili. 2nd edition: Helsinki Corpus of Swahili, Version 2.0 (HCS 2.0) 2004-09-30. University of Helsinki, Institute for Asian and African Studies.

3. **Swahili Wikipedia 2021**
   - Wikimedia Foundation. (2021). Swahili Wikipedia. Retrieved 2021 from https://sw.wikipedia.org/

4. **Swahili Community 2023**
   - Various Swahili news and community websites. (2023). Collected from sources including Mwananchi.co.tz, BBC Swahili, VOA Swahili, and Vodacom Tanzania.
### Dataset Citations

The Msingi1 language model was trained on a combined corpus created from the following two primary sources:

1. **Swahili Corpus**
   - Masasi, Noel; Masua, Bernard (2024), "Swahili Corpus", Mendeley Data, V2, doi: 10.17632/d4yhn5b9n6.2

2. **Helsinki Corpus of Swahili (HCS-NA-v2)**
   - Arvi Hurskainen (2004). Helsinki Corpus of Swahili. 2nd edition: Helsinki Corpus of Swahili, Version 2.0 (HCS 2.0) 2004-09-30. University of Helsinki, Institute for Asian and African Studies.

## Experimental Training Approach

Our current training methodology includes:

- **Batch Size**: 4 with gradient accumulation of 16 steps (effective batch = 64)
- **Learning Rate**: 3e-4 with cosine warmup and decay
- **Training Duration**: Experimenting with 10-15 epochs with early stopping
- **Mixed Precision**: FP16 training for speed and memory efficiency
- **Sliding Window Processing**: 50% overlap for better context learning

## Preliminary Text Generation Observations

Our experiments show the evolution of the model's capabilities. Here are some examples from different stages:

### Early Experiments (Epoch 5)
```
Prompt: Habari ya leo ni
Output: Habari ya leo ni ni ni ni ni shilingi la la la la la la la moja moja moja kampuni kampuni kufanya hilo muda kui mambo bwana bwana bwana bwana
```

### Mid-Training Observations (Epoch 10, with repetition penalty)
```
Prompt: Habari ya leo ni
Output: Habari ya leo ni mbili sheria sheria sana eneo tena jeshi bila fainali kufanya mkoani binafsi upande kuwa kuwa kuwa kupitia mafanikio polisi zao zao zao eneo eneo eneo
```

### Target Quality (Not Yet Achieved)
```
Prompt: Tanzania ni nchi
Target: Tanzania ni nchi kubwa katika Afrika Mashariki. Ina watu wengi wanaozungumza Kiswahili na lugha nyingine za kienyeji. Mji mkuu wa Tanzania ni Dodoma ingawa Dar es Salaam ndio mji mkubwa zaidi.
```

## Development Setup

For researchers interested in replicating or building upon our experiments:

```bash
# Clone repository
git clone https://github.com/Msingi-AI/msingi1.git
cd msingi1

# Install dependencies
pip install -r requirements.txt
```

### Experimental Scripts
```bash
# Text generation experiments
python src/test_model.py --prompt "Habari ya leo ni" --temperature 1.2 --repetition_penalty 1.5

# Training experiments
python src/train.py
```

## Project Structure
- `src/`: Experimental code for model architecture, training and inference
  - `model.py`: Msingi1 model architecture definition
  - `train.py`: Training loop and optimization experiments
  - `test_model.py`: Text generation and evaluation tools
  - `train_tokenizer.py`: Tokenizer training utilities
- `data/`: Data processing scripts and dataset utilities
- `checkpoints/`: Experimental model checkpoints
- `tokenizer/`: Tokenizer files and vocabulary
- `MODEL_CARD.md`: Research notes on model specifications

## Research Directions

Our ongoing and future research questions include:

1. **Multilingual Potential**: How can we extend to other East African languages?
2. **Instruction Tuning**: What approaches work best for instruction following in Swahili?
3. **Evaluation Challenges**: How do we develop standardized benchmarks for Swahili NLP?
4. **Efficiency Research**: What quantization approaches work best for African language models?

## Research Citation

If you build upon our research in your work, please cite:

```bibtex
@software{msingi1_2025,
  author = {Msingi AI Team},
  title = {Msingi1:Scaling Language Modeling for Swahili Through Small-Scale Pretraining},
  year = {2025},
  url = {https://github.com/Msingi-AI/msingi1},
  note = {Research in progress}
}
```

## License

MIT License
