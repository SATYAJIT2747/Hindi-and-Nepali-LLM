# 02. Language Model Pretraining: Custom Decoder-Only Transformer (GPT)

This module contains the complete codebase, configurations, training pipelines, generation engines, and evaluation suites for pretraining custom **Decoder-Only Transformer Language Models (~24.9M parameters)** from scratch on Hindi and Nepali corpora.

---

## 📌 Architecture Specification

Both language models share a synchronized causal GPT architecture designed for Devanagari script processing:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Model Type** | Causal Decoder Transformer | Pre-LayerNorm GPT architecture |
| **Layers (`n_layers`)** | 6 | Transformer blocks |
| **Embedding Dimension (`d_model`)** | 448 | Hidden representation size |
| **Attention Heads (`n_heads`)** | 8 | Multi-head self-attention (`d_head = 56`) |
| **FFN Inner Dimension (`d_ff`)** | 1,024 | Feed-forward intermediate dimension |
| **Context Length (`max_seq_len`)** | 512 | Maximum sequence length |
| **Vocabulary Size** | 32,000 | SentencePiece BPE tokens |
| **Dropout** | 0.1 | Regularization |
| **Total Parameters** | **24,919,552** | ~24.9M parameters (tied embeddings) |

---

## 📦 Trained Checkpoint Downloads (Kaggle Models)

Pretrained model checkpoints can be downloaded from Kaggle:
- **Model H (Hindi GPT Checkpoint)**: [kaggle.com/models/satyagudu/hindi-gpt-model-h](https://www.kaggle.com/models/satyagudu/hindi-gpt-model-h)
- **Model L (Nepali GPT Checkpoint)**: [kaggle.com/models/satyagudu/neplai-gpt-model-l](https://www.kaggle.com/models/satyagudu/neplai-gpt-model-l)

---

## 📁 Directory Structure

```text
02_Pretraining_Transformer_LMs/
├── README.md                          # Module Overview & Quickstart Guide
├── hindi/                             # Model H (Hindi GPT) Pretraining Pipeline
│   ├── README.md                      # Hindi Training & Evaluation Documentation
│   ├── configs/config.yaml            # Hyperparameters and path configurations
│   ├── model/gpt.py                   # Decoder-only GPT architecture implementation
│   ├── train/train.py                 # Pretraining entry point (CUDA, BFloat16)
│   ├── train/data.py                  # Memory-mapped dataset pipeline
│   ├── train/checkpoint.py            # Checkpoint saving, loading, and resuming
│   ├── eval/generate.py               # Greedy, top-k/nucleus sampling, and beam search
│   ├── eval/evaluate.py               # Perplexity and generation quality evaluation
│   ├── eval/metrics.py                # chrF, chrF++, BLEU, ROUGE-L, diversity metrics
│   ├── eval/attention_heatmap.py      # Attention matrices, distance, and entropy analysis
│   └── requirements.txt               # PyTorch and dependencies
├── nepali/                            # Model L (Nepali GPT) Pretraining Pipeline
│   ├── README.md                      # Nepali Training & Evaluation Documentation
│   ├── configs/config.yaml            # Hyperparameters and path configurations
│   ├── model/gpt.py                   # Decoder-only GPT architecture implementation
│   ├── train/train.py                 # Pretraining entry point (CUDA, BFloat16)
│   ├── train/data.py                  # Memory-mapped dataset pipeline
│   ├── train/checkpoint.py            # Checkpoint saving, loading, and resuming
│   ├── eval/generate.py               # Decoding strategies (Greedy / Sampling / Beam)
│   ├── eval/evaluate.py               # Evaluation benchmark execution
│   ├── eval/metrics.py                # Evaluation metrics implementation
│   ├── eval/attention_heatmap.py      # Attention heatmap generation and analysis
│   └── requirements.txt               # Dependencies
└── report/
    └── Phase2_Report.pdf              # Comprehensive Pretraining Report & Analysis
```

---

## 🚀 Quickstart & Reproduction

### 1. Requirements & Setup
```bash
# Recommended: Python 3.10+, NVIDIA GPU with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pyyaml tqdm sentencepiece matplotlib numpy
```

### 2. Pretrain Hindi Model
```bash
cd hindi
python -m train.train --config configs/config.yaml
```

### 3. Pretrain Nepali Model
```bash
cd nepali
python -m train.train --config configs/config.yaml
```

### 4. Evaluate Generation & Compute Metrics
```bash
python -m eval.evaluate --config configs/config.yaml --checkpoint <checkpoint_path>.pt --n-examples 50 --out-json results/eval.json
```

### 5. Generate Attention Heatmaps & Entropy Analysis
```bash
python -m eval.attention_heatmap --config configs/config.yaml --checkpoint <checkpoint_path>.pt --out results/attention_heatmap.png
```
