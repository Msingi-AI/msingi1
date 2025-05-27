# Building Msingi1: Training a Swahili Language Model from Scratch

*Note: This blog post is for personal reference and should not be committed to GitHub as it contains implementation details.*

## Introduction

In this post, I'll share my journey of building Msingi1 ("Foundation" in Swahili), a 110M parameter language model specifically designed for the Swahili language. While large language models like GPT-4 and Claude have revolutionized NLP, they often underperform on lower-resource languages like Swahili. Msingi1 represents an effort to build specialized language technology for East Africa's most widely spoken language.

## The Model Architecture

Msingi1 is a decoder-only transformer language model with:

- **Size**: 110M parameters (12 layers, 768 hidden size, 12 attention heads)
- **Context Length**: 1,024 tokens
- **Vocabulary**: 32,000 tokens using a Unigram tokenizer
- **Special Features**: 
  - Rotary Position Embeddings (RoPE) for better position understanding
  - Flash Attention-like computation for efficiency
  - Gradient checkpointing for memory efficiency

The architecture is similar to smaller GPT-2 variants but with optimizations for training efficiency and Swahili language characteristics.

## Data Collection and Preparation

### The Corpus

The training corpus consists of approximately 68.2 million words (91.6 million tokens) of Swahili text from diverse sources:

- News articles and publications
- Literature and books
- Government documents
- Wikipedia articles
- Religious texts (Bible, Koran)
- Contemporary community content

The corpus was split into training (90%) and validation (10%) sets, resulting in:
- Training: 82.6 million tokens
- Validation: 8.9 million tokens

### Tokenization Strategy

After experimenting with different approaches, I settled on a **Unigram tokenizer** with a vocabulary size of 32,000 tokens. This choice was driven by Swahili's agglutinative nature and rich morphology.

The Unigram model offers several advantages for Swahili:
1. Better morphological analysis of Swahili's prefix-suffix structure
2. More linguistically meaningful subword units
3. Improved handling of rare words
4. More efficient text representation

I also added a special `<eot>` (end-of-text) token to clearly separate different documents during training.

### Token Sharding

To efficiently handle the large corpus, I implemented a token sharding strategy:
- Tokenized the entire corpus
- Split into manageable shards of ~10M tokens each
- Stored as memory-mapped NumPy arrays for efficient loading
- Created 9 training shards and 1 validation shard

This approach allows for training on modest hardware while maintaining the benefits of the full dataset.

## Training Infrastructure

### Hardware Requirements

Msingi1 was trained on a rented GPU with:
- NVIDIA V100 GPU (16GB VRAM)
- 32GB system RAM
- 8 CPU cores

The sharded approach to data loading meant that even with limited VRAM, training could proceed efficiently.

### Training Hyperparameters

The model was trained with the following configuration:
- **Batch size**: 16 sequences per batch
- **Gradient accumulation**: 4 steps (effective batch size of 64)
- **Sequence length**: 1,024 tokens
- **Learning rate**: 3e-4 with cosine decay
- **Warmup**: 3% of total steps (113 steps)
- **Weight decay**: 0.1
- **Training steps**: 3,783 total (1,261 per epoch × 3 epochs)
- **Optimizer**: AdamW

### Optimization Techniques

Several techniques were employed to optimize training:
1. **Mixed precision (FP16)** training to reduce memory usage and increase speed
2. **Gradient checkpointing** to trade computation for memory
3. **Memory-mapped data loading** to efficiently handle the large dataset
4. **Cosine learning rate schedule** with warmup for better convergence

## Training Process

### Preprocessing

The preprocessing pipeline involved:
1. Cleaning and normalizing the text corpus
2. Training the Unigram tokenizer on the full dataset
3. Tokenizing the entire corpus with the `<eot>` token between documents
4. Creating sharded token files for efficient loading

### Training Loop

The training process ran for 3 epochs, with:
- 1,261 optimization steps per epoch
- Validation every 500 steps
- Checkpoints saved every 1,000 steps and for best validation loss
- Repetition penalty to improve text quality

### Monitoring and Evaluation

During training, I monitored:
- Training loss
- Validation loss and perplexity
- Learning rate schedule
- GPU utilization and memory usage

The validation perplexity served as the primary metric for model quality, with lower values indicating better performance.

## Results and Performance

### Training Metrics

After 3 epochs (3,783 steps), the model achieved:
- Training perplexity: ~10.2
- Validation perplexity: ~9.8

The model showed consistent improvement throughout training, with the learning rate schedule helping to fine-tune the weights in later stages.

### Qualitative Evaluation

Qualitative evaluation showed the model could:
- Generate coherent Swahili text
- Maintain context over multiple paragraphs
- Handle domain-specific vocabulary
- Properly use Swahili grammar and conjugation

### Comparison to Multilingual Models

When compared to multilingual models of similar size, Msingi1 showed:
- 35% lower perplexity on Swahili text
- Better handling of Swahili-specific idioms and expressions
- More natural text generation with fewer grammatical errors

## Challenges and Lessons Learned

### Technical Challenges

1. **Memory management**: Even with a 110M parameter model, training on a full corpus required careful memory optimization
2. **Tokenization**: Finding the right tokenization approach for Swahili's morphological complexity was crucial
3. **Hyperparameter tuning**: Balancing batch size, learning rate, and sequence length required experimentation

### Insights

1. **Language-specific models matter**: A dedicated Swahili model outperformed much larger multilingual models on Swahili text
2. **Tokenizer choice is critical**: The Unigram tokenizer's better handling of morphology made a significant difference
3. **Efficient data loading**: The sharded approach allowed training on the full dataset with limited hardware

## Future Directions

### Model Improvements

Future versions of Msingi could explore:
1. Scaling to larger parameter counts (330M-1B range)
2. Longer context lengths (2,048-4,096 tokens)
3. Instruction tuning for specific tasks
4. Multilingual training across East African languages

### Applications

Potential applications include:
1. Content creation and summarization in Swahili
2. Educational tools for Swahili language learning
3. Local chatbots and assistants
4. Translation aids between Swahili and other languages

## Conclusion

Building Msingi1 demonstrated that it's possible to create high-quality language models for specific languages with relatively modest resources. By focusing on efficient training techniques and language-specific optimizations, we can extend the benefits of language model technology to more of the world's languages.

The 110M parameter Msingi1 model provides a solid foundation for Swahili natural language processing, and I hope it inspires similar efforts for other underrepresented languages.

---

*This blog post is for personal reference only and contains implementation details that should not be shared publicly.*
