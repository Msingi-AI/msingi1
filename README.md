# Msingi1: Scaling Language Modelling Through Small-Scale Pretraining

## What is Msingi1?

Msingi ("Foundation" in Swahili) is our attempt to build decent language models for Swahili, one of Africa's most widely spoken languages. We started small, but have scaled up to multiple models that can generate grammatically correct Swahili text.

The project began with a simple question: *Can we build useful language models for African languages without billions of parameters and massive compute?* This README documents our journey, what we've learned, and where we're headed.

## Msingi1: 153M Model

Msingi1 was our first attempt at a Swahili language model with 336M parameters that can generate grammatically correct Swahili text.

## The Model: What's Under the Hood

Msingi1 is a 153 million parameter transformer language model - think of it as a smaller cousin to models like GPT-2. Here's what makes it tick:

- **Size**: 18 layers deep with 16 attention heads 
- **Context**: Can handle texts up to 1024 tokens long
- **Vocabulary**: Understands 32,000 unique Swahili word pieces

## Training: How We Taught Our Models Swahili

### The Data

#### Msingi1 Dataset
We fed Msingi1 a diverse diet of Swahili text - about 705 million tokens from:

- News articles (lots of these!)
- Government documents
- Literature and books
- Religious texts
- Wikipedia articles
- Web content

We split this into a training set (95%) and validation set (5%) to make sure the model was learning properly.

#### Msingi1 153M Dataset
For Msingi1 153M, we significantly expanded our dataset to 705 million tokens, approximately 8 times larger than the Msingi1 dataset. This expanded corpus includes:

- Additional news sources from East Africa
- More contemporary web content
- Educational materials and academic texts
- Government publications and legal documents
- Community forums and social media content
- Literature and creative writing

The larger dataset provides better coverage of diverse language use and improves the model's ability to generate coherent text across different domains. With 153M parameters and 705M tokens, Msingi1 153M has a much better token-to-parameter ratio of approximately 4.6:1, which helps prevent overfitting while enabling better language understanding.

### The Training Process

#### Msingi1 Training
Training Msingi1 was surprisingly efficient:

- Each epoch (full pass through the data) took only about 30 minutes on good GPU hardware
- We processed 8 examples at a time, but used gradient accumulation to effectively train on 64 examples at once
- We completed 3 epochs and extended to 5 to see if we could improve further

#### Msingi1 153M Training
For Msingi1 153M, we leveraged an A100 GPU to handle the larger dataset and implemented several optimizations:

- Training ran for 4 epochs with a learning rate of 3e-4 and cosine decay schedule
- Used batch size of 8 with gradient accumulation steps of 8 (effective batch size of 64)
- Implemented mixed precision (FP16) training for memory efficiency
- Utilized gradient checkpointing to reduce memory usage
- Added Weights & Biases integration for detailed training monitoring
- Generated sample text during evaluation to assess qualitative improvements
- Saved checkpoints at regular intervals (every 1000 steps) and at the end of each epoch
- Total training time: 9h 21m 8s on A100 GPU

### What We've Learned

#### Insights from Msingi1
After training Msingi1 for 3 epochs:

1. The model learned Swahili grammar remarkably well - it handles the complex prefix and suffix systems that make Swahili challenging
2. Training is surprisingly fast - each epoch takes just 30 minutes on decent hardware
3. The model has a "news bias" - it tends to drift toward news-style content because that's what dominated our training data

#### Insights from Msingi1 153M
Our experience with Msingi1 153M has taught us additional lessons:

1. **Smaller can be better**: The 153M parameter model (down from 336M) with more data achieves better results than the larger model
2. **Data diversity matters**: The expanded dataset helps reduce domain bias and improves generation quality
3. **Monitoring is essential**: Tracking metrics through WandB provides valuable insights into training dynamics
4. **Parameter efficiency**: A better token-to-parameter ratio (4.6:1) leads to more efficient learning
5. **Regularization techniques**: Increased dropout (0.15) helps prevent overfitting despite the smaller model size

## Results: What Our Models Can (and Can't) Do

### Msingi1 Text Generation

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

### Msingi1 153M Text Generation

Msingi1 153M shows significant improvements in coherence and topic adherence. Here's an example of Msingi1 153M completing the same greeting prompt:

**Prompt:** "Habari ya leo, jina langu ni"

**Sample 1:**
```
"Habari ya leo, jina langu ni Maria. Ninafurahi kukutana nawe leo. Mimi ni mwanafunzi wa chuo kikuu cha Nairobi, ninasomea sayansi ya kompyuta. Ninapenda kusoma, kusikiliza muziki, na kutembea na marafiki zangu wakati wa mapumziko. Je, wewe unaitwa nani? Unapenda kufanya nini wakati wa starehe?"
```

**Sample 2:**
```
"Habari ya leo, jina langu ni Juma. Nimekuja hapa kwa ajili ya mahojiano ya kazi. Nina uzoefu wa miaka mitano katika sekta ya teknolojia, hasa katika utengenezaji wa programu za simu. Nimehitimu kutoka Chuo Kikuu cha Dar es Salaam na shahada ya kwanza katika Sayansi ya Kompyuta. Ninatarajia kuwa sehemu ya timu yenu inayokua haraka."
```

### What's Improved with Msingi1 153M?

**The Good:**
- Much better topic adherence - stays with the personal introduction theme
- More natural conversational flow with appropriate follow-up content
- Diverse outputs that make sense in different contexts (casual conversation vs. job interview)
- Maintains consistent persona throughout the generation
- Grammatically correct with natural Swahili phrasing

**Still Working On:**
- Occasional tendency to be overly formal in casual contexts
- Limited creative storytelling abilities
- Some repetitive patterns in longer generations

The improvements in Msingi1 153M demonstrate the value of our optimization approach: a smaller but more efficient model (153M vs 336M parameters) trained on significantly more data (705M vs 88.6M tokens) with better regularization techniques.

## Msingi1 153M: Our Optimized Model

Building on what we learned from Msingi1, we developed Msingi1 153M - a more efficient model that balances performance with computational efficiency.

### The Model Architecture

Msingi1 153M is a 153 million parameter transformer language model with the following specifications:

- **Size**: 18 layers deep with 16 attention heads (153M parameters total)
- **Embedding Dimension**: 768 (reduced from 1024 in Msingi1)
- **Context**: Can handle texts up to 1024 tokens long
- **Vocabulary**: Same 32,000 unique Swahili word pieces using Unigram tokenizer
- **Dropout**: Increased to 0.15 (from 0.1) for better regularization

### Training Improvements

Msingi1 153M was trained on a larger dataset with improved techniques:

- **Dataset Size**: 705M training tokens (significantly larger than Msingi1's dataset)
- **Hardware**: Trained on an A100 GPU for faster processing
- **Training Duration**: 4 epochs with effective batch size of 64
- **Optimization**: Learning rate of 3e-4 with cosine decay schedule
- **Memory Efficiency**: Uses gradient checkpointing, mixed precision (FP16), and gradient accumulation

### Performance Benefits

Despite having fewer parameters than Msingi1, Msingi1 153M offers several advantages:

- **Better Token-to-Parameter Ratio**: ~4.6 tokens per parameter (vs ~2.1 in Msingi1)
- **Reduced Overfitting Risk**: Smaller model with more data and increased dropout
- **Faster Training and Inference**: Smaller size means quicker processing
- **Enhanced Monitoring**: Detailed tracking of perplexity, accuracy, and token-level metrics

### Sample Generation

Msingi1 153M generates more coherent and contextually appropriate Swahili text, with improved handling of complex grammatical structures and reduced tendency to drift off-topic.

## What's Next: Improving Our Models

We're actively working to make our Msingi models better:

1. **Fine-tuning Msingi1 153M**: We're planning collaborative fine-tuning sessions to adapt the model for specific applications:
   - Instruction following for task-oriented use cases
   - Domain-specific adaptations (legal, medical, educational)
   - Conversational abilities to maintain topic coherence

2. **Better Text Generation**: We're experimenting with different settings (temperature, top-p sampling, repetition penalty) to improve text quality and reduce biases

3. **Evaluation Framework**: We're developing comprehensive Swahili-specific benchmarks to measure model performance across different tasks

4. **Efficient Deployment**: We're exploring model compression techniques (quantization, pruning) to enable deployment on more resource-constrained environments

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
