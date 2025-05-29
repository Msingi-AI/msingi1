# Msingi1: Scaling Language Modelling Through Small-Scale Pretraining

## What is Msingi1?

Msingi1 ("Foundation" in Swahili) is our attempt to build a decent language model for Swahili, one of Africa's most widely spoken languages. We started small, but have now scaled up to a 336M parameter model that can generate grammatically correct Swahili text.

The project began with a simple question: *Can we build useful language models for African languages without billions of parameters and massive compute?* This README documents our journey, what we've learned, and where we're headed.

## The Model: What's Under the Hood

Msingi1 is a 336 million parameter transformer language model - think of it as a smaller cousin to models like GPT-2. Here's what makes it tick:

- **Size**: 24 layers deep with 16 attention heads (336M parameters total)
- **Context**: Can handle texts up to 1024 tokens long
- **Vocabulary**: Understands 32,000 unique Swahili word pieces

### Our Journey to 336M

We didn't start this big. Our first experiments were with tiny 28M parameter models, then we tried 110M, and now we're at 336M. Each step taught us something important:

- **Position Embeddings**: We started with fancy Rotary Position Embeddings (RoPE), but found that old-school learned position embeddings actually work better for Swahili
- **Memory Tricks**: We use gradient checkpointing and mixed precision to train bigger models without needing expensive hardware
- **Data Loading**: We chop our dataset into bite-sized pieces (shards) that can be loaded efficiently without running out of memory

## Training: How We Taught It Swahili

### The Data

We fed Msingi1 a diverse diet of Swahili text - about 88.6 million tokens (roughly 68.2 million words) from:

- News articles (lots of these!)
- Government documents
- Literature and books
- Religious texts
- Wikipedia articles
- Web content

We split this into a training set (90%) and validation set (10%) to make sure the model was learning properly.

### The Training Process

Training was surprisingly efficient:

- Each epoch (full pass through the data) took only about 30 minutes on good GPU hardware
- We processed 8 examples at a time, but used a trick called gradient accumulation to effectively train on 64 examples at once
- We've completed 3 epochs so far and are extending to 5 to see if we can improve further

### What We've Learned

After training for 3 epochs:

1. The model learned Swahili grammar remarkably well - it handles the complex prefix and suffix systems that make Swahili challenging
2. Training is surprisingly fast - each epoch takes just 30 minutes on decent hardware
3. The model has a "news bias" - it tends to drift toward news-style content because that's what dominated our training data

## Results: What Msingi1 Can (and Can't) Do

Let's look at what happens when we ask Msingi1 to complete a simple greeting:

**Prompt:** "Habari ya leo, jina langu ni" (Hello, my name is)

**Sample 1:**
```
"habari ya leo, jina langu ni je, wewe kama umeoa?pia ni vyema sasa viongozi wa serikali ambao wanaongozwa na sheria hii ya kuzuia rushwa nchini.kwa sababu kwa hali ilivyo, nasikitika kwamba suala la akinamama kutumia jembe la mkono limekuwa sugu kwa sababu hata mtoto akipanda chini anakomaa kidogo."
```

**Sample 2:**
```
"habari ya leo, jina langu ni abood aliiambia kituo hicho kwamba serikali imekuwa ikichukua hatua za kisheria kuzuia usafirishaji wa mazao hayo ambayo hayakuagizwa kutoka nje ya nchi kama ilivyo kwa makampuni mengine.kwa mara nyingine tena rais obama atakutana na viongozi wa vyama vya wafanyakazi..."
```

### What's Going On Here?

These examples show both the strengths and limitations of our current model:

**The Good:**
- The Swahili grammar is spot-on - the model handles complex word structures correctly
- The text flows naturally with proper sentence construction

**The Not-So-Good:**
- The model has a serious case of "news brain" - it quickly veers into news reporting style
- It can't stay on topic - what started as a personal introduction jumps to politics and government
- It mentions specific entities like "Obama" that it learned from news articles

This behavior makes perfect sense when you consider what the model learned from: our training data was heavily weighted toward news articles and government documents. The model is simply doing what it learned to do - continue text in the style it saw most often during training.

## What's Next: Improving Msingi1

We're actively working to make Msingi1 better:

1. **More Training**: We're currently running epochs 4-5 to see if more training helps with coherence

2. **Better Text Generation**: We're experimenting with different settings (temperature, top-p sampling) to reduce the "news brain" effect

3. **Conversation Tuning**: We're planning to fine-tune the model on conversational data to help it maintain topic focus

4. **Better Evaluation**: We're developing Swahili-specific ways to measure how good the model actually is

The current model is just the beginning - we see it as a foundation (hence the name "Msingi") that we can build upon to create truly useful Swahili language AI.

## Tokenization: Teaching the Model to Read Swahili

Before a language model can learn, it needs to break text into pieces it can understand. This process is called tokenization, and it's especially important for Swahili.

### Why Swahili Tokenization is Tricky

Swahili is an agglutinative language - it builds complex words by gluing together smaller meaningful pieces. For example, "ninakupenda" (I love you) combines "ni" (I) + "na" (present tense) + "ku" (you) + "penda" (love).

After experimenting with different approaches, we found that a **Unigram tokenizer** works best for Swahili. It's better at breaking words into meaningful pieces rather than arbitrary chunks.

### Our Tokenizer

- Understands 32,000 unique word pieces
- Trained on our full Swahili dataset
- Handles Swahili's complex word structure better than alternatives

For the technically curious: We compared Unigram with ByteLevelBPE tokenization and found that while both performed similarly in raw metrics, the Unigram tokenizer produced more linguistically sensible word splits for Swahili.

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
  title = {Msingi1:Scaling Language Modeling Through Small-Scale Pretraining},
  year = {2025},
  url = {https://github.com/Msingi-AI/msingi1},
  note = {Research in progress}
}
```

## License

MIT License
