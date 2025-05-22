import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
import random

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def sample_text_for_training(
    input_file: str,
    output_file: str,
    sample_size: int = 1000000,
    seed: int = 42
) -> None:
    """
    Sample lines from a large text file for tokenizer training.
    
    Args:
        input_file: Path to input text file
        output_file: Path to output sampled file
        sample_size: Number of lines to sample
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Count total lines
    print(f"Counting lines in {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total lines: {total_lines:,}")
    
    # Determine sampling rate
    sample_size = min(sample_size, total_lines)
    sampling_rate = sample_size / total_lines
    
    print(f"Sampling {sample_size:,} lines (rate: {sampling_rate:.4f})...")
    
    # Sample lines
    sampled_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if random.random() < sampling_rate and line.strip():
                sampled_lines.append(line)
                if len(sampled_lines) >= sample_size:
                    break
    
    # Write sampled lines
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
    
    print(f"Wrote {len(sampled_lines):,} lines to {output_file}")

def train_sentencepiece_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    model_type: str = "unigram",
    user_defined_symbols: Optional[List[str]] = None,
    max_sentence_length: int = 4192,
    input_sentence_size: int = 1000000,
    shuffle_input_sentence: bool = True,
    normalization_rule_name: str = "nmt_nfkc_cf",
    split_digits: bool = True,
    byte_fallback: bool = True
) -> None:
    """
    Train a SentencePiece tokenizer optimized for Swahili.
    
    Args:
        input_file: Path to input text file
        model_prefix: Prefix for output model files
        vocab_size: Size of vocabulary
        character_coverage: Character coverage (higher for languages with large character sets)
        model_type: Model type (unigram or bpe)
        user_defined_symbols: List of special tokens to add
        max_sentence_length: Maximum sentence length
        input_sentence_size: Number of sentences to use for training
        shuffle_input_sentence: Whether to shuffle input sentences
        normalization_rule_name: Normalization rule
        split_digits: Whether to split digits
        byte_fallback: Whether to use byte fallback for unknown characters
    """
    if user_defined_symbols is None:
        user_defined_symbols = ["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<sw>"]
    
    # Create training command
    cmd = f"--input={input_file} "
    cmd += f"--model_prefix={model_prefix} "
    cmd += f"--vocab_size={vocab_size} "
    cmd += f"--character_coverage={character_coverage} "
    cmd += f"--model_type={model_type} "
    cmd += f"--max_sentence_length={max_sentence_length} "
    cmd += f"--input_sentence_size={input_sentence_size} "
    cmd += f"--shuffle_input_sentence={'true' if shuffle_input_sentence else 'false'} "
    cmd += f"--normalization_rule_name={normalization_rule_name} "
    cmd += f"--split_digits={'true' if split_digits else 'false'} "
    cmd += f"--byte_fallback={'true' if byte_fallback else 'false'} "
    
    # Add special tokens
    if user_defined_symbols:
        cmd += f"--user_defined_symbols={','.join(user_defined_symbols)} "
    
    # Add Swahili-specific settings
    cmd += "--split_by_unicode_script=true "
    cmd += "--split_by_whitespace=true "
    cmd += "--treat_whitespace_as_suffix=false "
    
    # Train the model
    print(f"Training SentencePiece tokenizer with {model_type} model...")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {user_defined_symbols}")
    
    start_time = time.time()
    spm.SentencePieceTrainer.train(cmd)
    elapsed_time = time.time() - start_time
    
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Model saved to {model_prefix}.model and {model_prefix}.vocab")

def convert_to_huggingface(
    model_file: str,
    vocab_file: str,
    save_dir: str
) -> None:
    """
    Convert SentencePiece model to Hugging Face format.
    
    Args:
        model_file: Path to SentencePiece model file
        vocab_file: Path to SentencePiece vocabulary file
        save_dir: Directory to save Hugging Face tokenizer
    """
    print("Converting to Hugging Face format...")
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load SentencePiece model
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=model_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<sw>"]
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"Hugging Face tokenizer saved to {save_dir}")

def test_tokenizer(
    model_file: str,
    test_sentences: List[str] = None
) -> None:
    """
    Test the trained tokenizer on sample sentences.
    
    Args:
        model_file: Path to SentencePiece model file
        test_sentences: List of test sentences
    """
    if test_sentences is None:
        test_sentences = [
            "Jambo! Habari yako?",
            "Nina umri wa miaka 25 na ninaishi Nairobi.",
            "Elimu ni muhimu sana kwa maendeleo ya jamii.",
            "Kompyuta yangu mpya ina programu za kisasa.",
            "Haba na haba hujaza kibaba.",
            "Ninapenda kusoma vitabu vya Kiswahili na kusikiliza muziki.",
            "Serikali ya Kenya imeahidi kuboresha miundombinu ya nchi.",
            "Wanafunzi wanasoma kwa bidii ili kufaulu mtihani wao.",
            "Mkulima huyo analima mahindi, maharage na viazi.",
            "Safari ya kutoka Nairobi hadi Mombasa inachukua saa tano kwa gari."
        ]
    
    # Load the model
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    
    print("\nTokenizer tests:")
    for test in test_sentences:
        # Encode
        tokens = sp.encode_as_pieces(test)
        ids = sp.encode_as_ids(test)
        
        # Decode
        decoded = sp.decode_pieces(tokens)
        
        # Print results
        print(f"\nOriginal: {test}")
        print(f"Tokens: {' '.join(tokens)}")
        print(f"IDs: {ids}")
        print(f"Decoded: {decoded}")
        print(f"Number of tokens: {len(tokens)}")

def analyze_tokenizer(
    model_file: str,
    vocab_file: str,
    sample_text_file: str = None,
    sample_size: int = 1000
) -> None:
    """
    Analyze tokenizer performance on sample text.
    
    Args:
        model_file: Path to SentencePiece model file
        vocab_file: Path to SentencePiece vocabulary file
        sample_text_file: Path to sample text file
        sample_size: Number of lines to sample
    """
    # Load the model
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    
    # Load or create sample text
    if sample_text_file and os.path.exists(sample_text_file):
        with open(sample_text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()[:sample_size] if line.strip()]
    elif os.path.exists("data/valid.txt"):
        with open("data/valid.txt", 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            lines = [line.strip() for line in random.sample(all_lines, min(sample_size, len(all_lines))) if line.strip()]
    else:
        print("No sample text available for analysis")
        return
    
    # Analyze tokenization
    total_tokens = 0
    total_chars = 0
    token_lengths = []
    
    for line in lines:
        tokens = sp.encode_as_pieces(line)
        total_tokens += len(tokens)
        total_chars += len(line)
        token_lengths.extend([len(token) for token in tokens])
    
    # Calculate statistics
    avg_tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0
    avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    
    # Print analysis
    print("\nTokenizer Analysis:")
    print(f"Sample size: {len(lines)} lines")
    print(f"Total characters: {total_chars}")
    print(f"Total tokens: {total_tokens}")
    print(f"Compression ratio (chars/tokens): {total_chars/total_tokens:.2f}")
    print(f"Tokens per character: {avg_tokens_per_char:.2f}")
    print(f"Average token length: {avg_token_length:.2f} characters")
    
    # Vocabulary statistics
    print("\nVocabulary Statistics:")
    print(f"Vocabulary size: {sp.get_piece_size()}")
    
    # Count token types
    subword_count = 0
    word_count = 0
    special_count = 0
    
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece.startswith("▁"):  # Word-initial piece
            word_count += 1
        elif piece.startswith("<") and piece.endswith(">"):  # Special token
            special_count += 1
        else:  # Subword
            subword_count += 1
    
    print(f"Word-initial pieces: {word_count} ({word_count/sp.get_piece_size()*100:.1f}%)")
    print(f"Subword pieces: {subword_count} ({subword_count/sp.get_piece_size()*100:.1f}%)")
    print(f"Special tokens: {special_count} ({special_count/sp.get_piece_size()*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer for Swahili")
    parser.add_argument("--input", default="data/train.txt", help="Input text file")
    parser.add_argument("--sample", action="store_true", help="Sample input for faster training")
    parser.add_argument("--sample-size", type=int, default=1000000, help="Number of lines to sample")
    parser.add_argument("--model-dir", default="tokenizer", help="Directory to save model")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram", help="Model type")
    parser.add_argument("--character-coverage", type=float, default=0.9995, help="Character coverage")
    
    args = parser.parse_args()
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Sample input if requested
    input_file = args.input
    if args.sample:
        sample_file = os.path.join(args.model_dir, "sampled_input.txt")
        sample_text_for_training(input_file, sample_file, args.sample_size)
        input_file = sample_file
    
    # Set model prefix
    model_prefix = os.path.join(args.model_dir, f"swahili_{args.model_type}_{args.vocab_size}")
    
    # Train tokenizer
    train_sentencepiece_tokenizer(
        input_file=input_file,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type
    )
    
    # Convert to Hugging Face format
    convert_to_huggingface(
        model_file=f"{model_prefix}.model",
        vocab_file=f"{model_prefix}.vocab",
        save_dir=os.path.join(args.model_dir, "huggingface")
    )
    
    # Test tokenizer
    test_tokenizer(model_file=f"{model_prefix}.model")
    
    # Analyze tokenizer
    analyze_tokenizer(
        model_file=f"{model_prefix}.model",
        vocab_file=f"{model_prefix}.vocab",
        sample_text_file=input_file if args.sample else None
    )

if __name__ == "__main__":
    main()
