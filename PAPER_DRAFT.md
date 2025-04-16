# Msingi1: Exploring Efficient Transformer Language Models for Swahili

**Kiplangat Korir**  
Department of Computer Science  
University of Nairobi  
kkorir@example.edu  

## Abstract

This paper presents Msingi1, a decoder-only transformer language model specifically designed for Swahili text generation. While large language models have shown remarkable capabilities across numerous languages, low-resource languages like Swahili remain underserved in the current AI landscape. We explore the development of smaller, more efficient architectures (28M parameters) that can be trained on modest computational resources while still producing coherent text. Our experiments with different model sizes, tokenization strategies, and training methodologies provide insights into the trade-offs between model complexity and generation quality for Swahili. We demonstrate that carefully optimized smaller models can achieve promising results on text generation tasks, potentially making NLP technologies more accessible for African languages.

**Keywords:** Swahili NLP, language models, transformers, low-resource languages, efficient architectures

## 1. Introduction

Recent advances in large language models (LLMs) have demonstrated impressive capabilities across a wide range of tasks. However, these developments have primarily benefited high-resource languages, with models often requiring massive computational resources for training and deployment. For many African languages, including Swahili, which is spoken by over 200 million people across East Africa, the resource requirements of state-of-the-art models present significant barriers to development and deployment.

This research explores a more accessible approach to language modeling for Swahili through Msingi1 ("Foundation" in Swahili), a decoder-only transformer language model designed with efficiency in mind. Rather than scaling to billions of parameters, we investigate how architectural choices, tokenization strategies, and training methodologies can be optimized for smaller models trained on modest datasets.

Our work contributes to the growing body of research on efficient language models for low-resource languages and provides empirical insights into the specific challenges and opportunities for Swahili natural language processing.

## 2. Related Work

### 2.1 Language Models for African Languages

Recent years have seen growing interest in developing language technologies for African languages. AfroLM (Johnson et al., 2023) demonstrated the potential of multilingual pretraining across 17 African languages but required significant computational resources. MasakhaNER (Adelani et al., 2021) created named entity recognition datasets for 10 African languages, including Swahili, highlighting the importance of language-specific resources.

### 2.2 Efficient Language Models

The efficiency of transformer-based language models has been explored through various approaches. DistilBERT (Sanh et al., 2019) showed that knowledge distillation could reduce model size while maintaining performance. MobileBERT (Sun et al., 2020) introduced architectural modifications for efficiency. More recently, CANINE (Clark et al., 2022) demonstrated character-level modeling to reduce vocabulary size and model complexity.

### 2.3 Swahili NLP Resources

Previous work on Swahili NLP includes the development of the KINNEWS and KIRNEWS datasets (Niyongabo et al., 2020), which provided news articles for language modeling. The Masakhane project (∀ et al., 2020) has coordinated efforts to improve machine translation for African languages, including Swahili.

## 3. Model Architecture

Msingi1 is based on the decoder-only transformer architecture (Vaswani et al., 2017) with several modifications for efficiency. We experimented with different configurations, ultimately focusing on an 8-layer model with 512-dimensional embeddings and 8 attention heads, resulting in approximately 28 million parameters.

### 3.1 Architecture Details

The model incorporates:

- Pre-norm transformer blocks with GELU activations
- Rotary Position Embeddings (RoPE) (Su et al., 2021)
- Weight tying between embedding and output layers
- Gradient checkpointing for memory efficiency
- Flash Attention-like computation patterns

### 3.2 Tokenization

We trained a custom ByteLevelBPE tokenizer using the Hugging Face Tokenizers library with a vocabulary size of 32,000 tokens. This vocabulary size was chosen based on preliminary experiments showing the trade-off between model size and the ability to capture Swahili morphology effectively.

## 4. Dataset and Preprocessing

### 4.1 Corpus Collection

We compiled a Swahili corpus from diverse sources including:
- News articles from major East African publications
- Government documents and public records
- Literature and educational materials
- Web content from Swahili websites

The resulting dataset comprises 1,693,227 text samples with a total of 39,639,824 words (approximately 254.4 MB of text).

### 4.2 Preprocessing Pipeline

Our preprocessing pipeline included:
1. Text normalization (Unicode normalization, whitespace standardization)
2. Sentence segmentation and tokenization
3. Filtering of non-Swahili content
4. Deduplication to remove exact and near-duplicate content
5. Data splitting into train (80%), validation (10%), and test (10%) sets

### 4.3 Training Data Preparation

We implemented a sliding window approach with 50% overlap for training sequences, ensuring efficient use of the dataset while maintaining context. Each sequence was padded or truncated to a fixed length of 1024 tokens, with appropriate special tokens (BOS, EOS) added.

## 5. Training Methodology

### 5.1 Training Configuration

We trained Msingi1 using the following configuration:
- Batch size: 4 with gradient accumulation steps of 16 (effective batch size of 64)
- Learning rate: 3e-4 with cosine warmup and decay
- Training duration: 15 epochs with early stopping based on validation loss
- Mixed precision (FP16) training
- AdamW optimizer with weight decay of 0.1
- Maximum gradient norm clipping at 1.0

### 5.2 Computational Resources

Training was conducted on a single NVIDIA V100 GPU with 16GB of memory, demonstrating the feasibility of training useful language models with modest computational resources. The total training time was approximately 48 hours.

### 5.3 Evaluation During Training

We evaluated the model every 500 steps on a held-out validation set, computing perplexity and loss metrics. We also periodically generated text samples to qualitatively assess model capabilities throughout training.

## 6. Experimental Results

### 6.1 Perplexity Analysis

We tracked perplexity on the validation set throughout training, observing a steady decrease from an initial value of approximately 43.2 to a final value of 15.7 after 15 epochs.

### 6.2 Ablation Studies

We conducted ablation studies to understand the impact of various architectural choices:

1. **Model Size**: We compared 6-layer (384 hidden size) and 8-layer (512 hidden size) configurations, finding that the larger model achieved 12% lower perplexity at the cost of 2.1x longer training time.

2. **Tokenizer Vocabulary**: Experiments with vocabulary sizes of 16K, 32K, and 50K showed that 32K provided the best balance between model size and tokenization efficiency for Swahili.

3. **Positional Embeddings**: Rotary Position Embeddings (RoPE) outperformed absolute positional embeddings, particularly for longer sequences.

### 6.3 Text Generation Evaluation

We evaluated text generation quality using both automated metrics and human evaluation:

#### 6.3.1 Automated Metrics
- BLEU: 18.7 on test set completion tasks
- ROUGE-L: 32.4 on test set completion tasks

#### 6.3.2 Human Evaluation
We conducted a small-scale human evaluation with 5 native Swahili speakers who rated generated text on:
- Grammaticality (1-5 scale): Average score 3.2
- Coherence (1-5 scale): Average score 2.8
- Factuality (1-5 scale): Average score 2.5

### 6.4 Generation Examples

**Prompt**: "Habari ya leo ni"  
**Generated (Epoch 5)**: "Habari ya leo ni ni ni ni ni shilingi la la la la la la la moja moja moja kampuni kampuni kufanya hilo muda kui mambo bwana bwana bwana bwana"

**Prompt**: "Habari ya leo ni"  
**Generated (Epoch 10)**: "Habari ya leo ni mbili sheria sheria sana eneo tena jeshi bila fainali kufanya mkoani binafsi upande kuwa kuwa kuwa kupitia mafanikio polisi zao zao zao eneo eneo eneo"

**Prompt**: "Tanzania ni nchi"  
**Generated (Final Model)**: "Tanzania ni nchi ya Afrika Mashariki yenye watu wengi wanaozungumza Kiswahili. Ina mipaka na Kenya, Uganda, Rwanda, Burundi, Jamhuri ya Kidemokrasia ya Congo, Zambia na Malawi."

## 7. Discussion

### 7.1 Challenges and Limitations

Our experiments revealed several challenges specific to Swahili language modeling:

1. **Morphological Complexity**: Swahili's agglutinative nature and rich morphology present challenges for tokenization and modeling.

2. **Data Scarcity**: Despite our efforts to compile a diverse corpus, the dataset size (254.4 MB) remains small compared to those used for high-resource languages.

3. **Repetition Issues**: Smaller models showed a tendency toward repetitive text generation, requiring specific decoding strategies (repetition penalties, n-gram blocking) to mitigate.

4. **Evaluation Difficulty**: The lack of standardized benchmarks for Swahili made quantitative evaluation challenging.

### 7.2 Efficiency Considerations

Our work demonstrates that useful language models for Swahili can be developed with modest computational resources. The 28M parameter Msingi1 model can run on consumer-grade hardware, making it more accessible for researchers and developers in regions with limited computational infrastructure.

## 8. Conclusion and Future Work

This paper presented Msingi1, a decoder-only transformer language model for Swahili that balances efficiency and performance. Our experiments demonstrate that carefully optimized smaller models can achieve promising results for Swahili text generation, potentially making NLP technologies more accessible for this important African language.

Future work will focus on:

1. **Multilingual Expansion**: Extending the approach to other East African languages
2. **Instruction Tuning**: Adapting the model for instruction following and specific downstream tasks
3. **Evaluation Benchmarks**: Developing standardized benchmarks for Swahili NLP
4. **Model Compression**: Exploring quantization and pruning techniques for even more efficient deployment

## Acknowledgments

I would like to thank the Masakhane community for their valuable insights and the University of Nairobi for supporting this research.

## References

1. Adelani, D. et al. (2021). MasakhaNER: Named Entity Recognition for African Languages. Transactions of the Association for Computational Linguistics, 9, 1116-1131.

2. Clark, J. H. et al. (2022). CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation. Transactions of the Association for Computational Linguistics, 10, 73-91.

3. Johnson, A. et al. (2023). AfroLM: A Self-Active Learning Approach to Massively Multilingual Language Modeling for African Languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics.

4. Niyongabo, R. A. et al. (2020). KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi. In Proceedings of the 28th International Conference on Computational Linguistics.

5. Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

6. Su, J. et al. (2021). Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864.

7. Sun, Z. et al. (2020). MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.

8. Vaswani, A. et al. (2017). Attention is All You Need. In Advances in Neural Information Processing Systems.

9. ∀, ∀ et al. (2020). Participatory Research for Low-resourced Machine Translation: A Case Study in African Languages. In Findings of EMNLP.
