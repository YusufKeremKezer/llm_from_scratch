# LLM from Scratch

A from-scratch implementation of a decoder-only transformer language model using PyTorch, trained on Turkish literary text.

## Overview

- **Goal**: Build and understand every component of a modern LLM without relying on high-level libraries
- **Architecture**: GPT-style decoder-only transformer with causal self-attention
- **Training Data**: Yunus Emre's collected works (Turkish poetry/prose)
- **Python**: 3.12+ | **Dependencies**: PyTorch, NumPy

## Project Structure

```
llm_from_scratch/
├── main.py                  # Entry point: BPE training → model init → forward pass
├── tokenization/
│   └── bpe.py               # Byte-Pair Encoding (BPE) tokenizer from scratch
├── model/
│   ├── transformer.py       # Transformer decoder (MHA, FFN, decoder blocks)
│   └── embeddings.py        # Token embeddings with truncated normal init
├── data/
│   └── yunusemre.txt        # Turkish text corpus
└── training/                # Training loop (WIP)
```

## Components

### Tokenization (`tokenization/bpe.py`)
- **BPE Training**: Learns merge rules from corpus by iteratively merging most frequent character pairs
- **BPE Inference**: Tokenizes new text using learned merge rules
- **Vocabulary**: Character-level base vocab + learned subword merges + end-of-word token
- **Key functions**: `bpe_training()`, `bpe_inference()`, `count_merge_pairs()`, `apply_merge()`

### Model (`model/transformer.py`)
- **Multi-Head Attention**: Scaled dot-product attention with causal masking (upper triangular mask)
- **Feed-Forward Network**: Two-layer MLP with GELU activation and 4× expansion factor
- **Decoder Block**: Pre-norm → MHA → residual → post-norm → FFN → residual
- **Transformer Model**: Stacks N decoder layers + final LM head projecting to vocab size
- **Positional Encoding**: Stub (WIP)

### Embeddings (`model/embeddings.py`)
- Standard `nn.Embedding` layer with truncated normal initialization (`std=0.02`)
- Prevents outlier spikes in embedding weights at initialization

## Quick Start

```bash
# Install dependencies
uv sync

# Run the pipeline (BPE training + model forward pass)
python main.py
```

## Key Design Decisions

- **Decoder-only** architecture (like GPT) — no encoder, no cross-attention
- **Pre-norm + post-norm** LayerNorm placement in each decoder block
- **Causal masking** via upper triangular boolean mask for autoregressive generation
- **BPE tokenizer** built entirely from scratch (no HuggingFace/sentencepiece dependency)
- **Truncated normal** embedding initialization for training stability

## Status

- [x] BPE tokenizer (training + inference)
- [x] Transformer decoder architecture
- [x] Embeddings with proper initialization
- [ ] Training loop
- [ ] Positional encoding
- [ ] Text generation / inference
- [ ] Evaluation metrics