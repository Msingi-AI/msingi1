# Msingi1: Scaling Language Modeling for Swahili Through Small-Scale Pretraining
 
## Research Overview

Msingi1 ("Foundation" in Swahili) documents our ongoing research journey in developing decoder-only transformer language models for Swahili. This project is an experimental exploration of foundation models for low-resource African languages, with a focus on understanding the trade-offs between model size, training efficiency, and text generation quality.

> **Note**: This project is a work-in-progress research effort and not yet ready for production use. We're sharing our journey to contribute to the growing body of knowledge on African language NLP.

## Architectural Exploration

Throughout this research journey, we've experimented with different model architectures:

### Current Architecture
- 12 layers, 768 hidden size, 12 attention heads
- Approximately 110M parameters (with 32K vocabulary)
- 2048 token context length
- Rotary Position Embeddings (RoPE)
- Pre-norm transformer architecture with GELU activation
- Flash Attention optimization for efficient training
- Gradient checkpointing for memory efficiency

### Previous Experiments
- 8 layers, 512 hidden size, 8 attention heads (~28M parameters)
- Various context length configurations (1024-2048 tokens)
- Different initialization strategies and training dynamics
- Exploration of EOS token handling for better text completion

### Research Focus Areas
- **Efficient Attention Mechanisms**: Exploring optimizations for limited GPU resources
- **Gradient Checkpointing**: Testing memory-efficient training approaches
- **Tokenization Strategies**: Implementing both ByteLevelBPE and Unigram tokenization with 32K vocabulary for Swahili
- **Text Generation Techniques**: Experimenting with repetition penalties and n-gram blocking

## Tokenizer Implementation

We've implemented two types of tokenizers optimized for Swahili, each with their own strengths:

### ByteLevelBPE Tokenizer

- **Type**: ByteLevelBPE (Byte-Level Byte Pair Encoding)
- **Vocabulary Size**: 32,000 tokens
- **Special Tokens**: `<s>`, `</s>`, `<unk>`, `<pad>`, `<mask>`, `<sw>`, `<eot>`
- **Training Corpus**: Full training dataset (383 MB, ~41.8M words)
- **Implementation**: Built using Hugging Face Tokenizers library

The ByteLevelBPE approach offers these advantages for Swahili:
1. Handles unknown words gracefully through character-level fallback
2. Captures frequent subword patterns efficiently
3. Works well with subword regularities in Bantu languages
4. Maintains compatibility with mainstream transformer architectures

### Unigram Tokenizer

- **Type**: Unigram (SentencePiece-style)
- **Vocabulary Size**: 32,000 tokens
- **Special Tokens**: `<s>`, `</s>`, `<unk>`, `<pad>`, `<mask>`, `<sw>`, `<eot>`
- **Training Corpus**: Full training dataset (383 MB, ~41.8M words)
- **Implementation**: Built using Hugging Face Tokenizers library

The Unigram approach is particularly well-suited for Swahili because:
1. It better handles morphological complexity through statistical optimization
2. It creates more linguistically meaningful subword units
3. It's especially effective for agglutinative languages like Swahili
4. It often produces more natural word segmentations for rare words
5. It typically represents text with fewer tokens than BPE

Both tokenizers are available in both native format and Hugging Face Transformers format in the `tokenizer/` directory. They can be loaded and used as follows:

```python
from transformers import PreTrainedTokenizerFast

# Load the ByteLevelBPE tokenizer
bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/swahili_bpe_32000/transformers")

# Load the Unigram tokenizer
unigram_tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/swahili_unigram_32000/transformers")

# Example text
text = "Ninapenda kusoma vitabu vya Kiswahili na kusikiliza muziki."

# Compare tokenization approaches
bpe_tokens = bpe_tokenizer.tokenize(text)
unigram_tokens = unigram_tokenizer.tokenize(text)

print(f"BPE tokens: {bpe_tokens}")
print(f"BPE token count: {len(bpe_tokens)}")
print(f"Unigram tokens: {unigram_tokens}")
print(f"Unigram token count: {len(unigram_tokens)}")

# Using the <eot> token for text separation
texts = ["Habari ya leo.", "Habari nzuri sana."]
combined_text = bpe_tokenizer.eos_token.join(texts)  # Joins with <eot>
print(f"Combined with <eot>: {combined_text}")
encoded = bpe_tokenizer.encode(combined_text)
print(f"Decoded back: {bpe_tokenizer.decode(encoded)}")
```

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
- **Gradient Checkpointing**: Memory-efficient backpropagation
- **Sharded Dataset**: Memory-mapped token shards for efficient loading

## Dataset Sharding

To efficiently train on our large Swahili corpus while minimizing memory usage, we've implemented a sharded token dataset approach:

1. **Tokenization Process**:
   - The raw text corpus is tokenized using our Unigram tokenizer
   - Each document is separated with an `<eot>` token
   - Tokens are stored as memory-mapped NumPy arrays for efficient access

2. **Sharding Strategy**:
   - Training data is split into ~10M token shards
   - Validation data is kept in a single shard
   - Each shard is stored as a separate `.npy` file
   - Memory mapping enables loading only the required portions during training

3. **Shard Distribution**:

| Shard File | Type | Size (MB) | Tokens | Description |
|------------|------|-----------|--------|-------------|
| msingi_train_000000.npy | Training | 19.1 | 10,000,000 | Training shard 1 |
| msingi_train_000001.npy | Training | 19.1 | 10,000,000 | Training shard 2 |
| msingi_train_000002.npy | Training | 19.1 | 10,000,000 | Training shard 3 |
| msingi_train_000003.npy | Training | 19.1 | 10,000,000 | Training shard 4 |
| msingi_train_000004.npy | Training | 19.1 | 10,000,000 | Training shard 5 |
| msingi_train_000005.npy | Training | 19.1 | 10,000,000 | Training shard 6 |
| msingi_train_000006.npy | Training | 19.1 | 10,000,000 | Training shard 7 |
| msingi_train_000007.npy | Training | 19.1 | 10,000,000 | Training shard 8 |
| msingi_train_000008.npy | Training | 3.0 | 1,551,598 | Training shard 9 (partial) |
| msingi_val_000000.npy | Validation | 17.0 | 8,904,050 | Validation shard |
| **Total** | | **173.8** | **91,551,598** | **Full dataset** |

4. **Benefits of Sharding**:
   - **Memory Efficiency**: Only loads necessary tokens into memory
   - **Training Speed**: Reduces I/O bottlenecks through memory mapping
   - **Scalability**: Enables training on larger datasets than would fit in RAM
   - **Flexibility**: Allows for dynamic shard loading and epoch definitions

5. **Implementation**:
   - `create_token_shards.py`: Tokenizes and creates the sharded dataset
   - `train_with_shards.py`: Implements efficient training with the sharded approach
   - `ShardedTokenDataset` class: Handles dynamic loading of token shards

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
