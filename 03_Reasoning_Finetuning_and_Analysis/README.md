# Multi-Entity Reasoning Fine-Tuning Pipeline: Model H (Hindi) & Model N (Nepali) GPTs

This repository contains the complete dataset generation, fine-tuning pipeline, evaluation suites, attention diagnostics, and reproduction steps for two Causal GPT Language Models:
* **Model H (Hindi GPT)**: 6-Layer Transformer fine-tuned on Hindi multi-entity reasoning tasks.
* **Model N (Nepali GPT)**: 6-Layer Transformer fine-tuned on Nepali multi-entity reasoning tasks.
* **Bonus Ablation Study**: Model H trained with **Positional Embeddings explicitly disabled** (`use_pos_emb=False`).

---

## 📌 Repository Directory Structure

```text
phase3/
├── README.md                          # Master Project Overview & Reproduction Steps
├── Hindi/                             # Model H (Hindi GPT) Pipeline
│   ├── README.md                      # Detailed Hindi Model Report & Metrics
│   ├── datagen/
│   │   ├── soft_gen.py                # Stage 1 Soft Dataset Generator
│   │   └── hard_datagen.py            # Stage 2 Hard Dataset Generator
│   ├── finetune/
│   │   ├── soft_finetuned.py          # Stage 1 Soft Fine-Tuning Script
│   │   └── hard_after_soft.py         # Stage 2 Hard Fine-Tuning Script
│   ├── eval/
│   │   ├── final_eval_hard.py         # Full Test Evaluation Suite
│   │   └── attention_analysis.py      # Attention Matrix & Heatmap Analysis
│   └── logs/
│       ├── soft_log.txt               # Soft Fine-Tuning Logs
│       └── hard_final_res.txt         # Hard Fine-Tuning & Evaluation Log
├── Nepali/                            # Model N (Nepali GPT) Pipeline
│   ├── README.md                      # Detailed Nepali Model Report & Metrics
│   ├── datagen/
│   │   ├── soft_gen.py                # Stage 1 Soft Dataset Generator
│   │   └── hard_datagen.py            # Stage 2 Hard Dataset Generator
│   ├── finetune/                      # Stage 1 & 2 Fine-Tuning Scripts
│   │   ├── soft_finetuned.py          # Stage 1 Soft Fine-Tuning Script
│   │   └── hard_after_soft.py         # Stage 2 Hard Fine-Tuning Script
│   ├── model_finetuned/               # Fine-Tuning Scripts Alias Directory
│   │   ├── soft_finetuned.py
│   │   └── hard_after_soft.py
│   ├── eval/
│   │   ├── final_eval_hard.py         # Full Test Evaluation Suite
│   │   └── attention_analysis.py      # Attention Diagnostics & Visualizations
│   └── logs/
│       ├── soft_finetuned.log
│       ├── hard_after_soft.log
│       └── final_eval_hard.log
└── Bonus/                             # Ablation Study: Positional Embedding Disabled
    ├── README.md                      # Detailed Ablation Analysis Report
    ├── config/                        # Model Configuration
    ├── model/                         # PyTorch GPT (use_pos_emb=False)
    ├── train/                         # Pretraining Scripts
    ├── eval/                          # Generation & Attention Analysis Scripts
    └── graphs/                        # Attention Heatmaps (48 Heads)
```

---

## 🏗️ Model Architecture Specifications

Both Hindi and Nepali models share an identical 6-layer decoder-only Causal Transformer architecture built with PyTorch, tailored specifically for low-resource Devanagari reasoning:

| Parameter | Specification |
| :--- | :--- |
| **Model Type** | Causal Transformer Decoder |
| **Layers (`n_layers`)** | 6 |
| **Hidden Dimension (`d_model`)** | 448 |
| **Attention Heads (`n_heads`)** | 8 (`d_head` = 56) |
| **FFN Inner Dimension (`d_ff`)** | 1,024 |
| **Max Sequence Length** | 512 |
| **Vocabulary Size** | 32,000 (SentencePiece BPE) |
| **Total Parameters** | **24,919,552** (~24.9M) |

---

## 📊 Summary of Benchmark Evaluation Results

Evaluated on 3,000 synthetic multi-entity reasoning test prompts (`hard_test.jsonl`):

### Pretrained vs. Two-Stage Fine-Tuned Accuracy

| Evaluation Metric | Model H (Hindi) Pretrained | Model H (Hindi) Hard Finetuned | Model N (Nepali) Pretrained | Model N (Nepali) Hard Finetuned |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Match Accuracy** | 0.00% | **59.23%** (1777/3000) | 0.00% | **56.53%** (1696/3000) |
| **First-Token Accuracy** | 0.00% | **59.23%** (1777/3000) | 0.00% | **56.53%** (1696/3000) |

### Task-Wise Accuracy Breakdown

| Task Type | Description | Model H (Hindi) | Model N (Nepali) |
| :--- | :--- | :--- | :--- |
| `bind_entity_to_value` | Direct lookup: Entity $\rightarrow$ Value | **100.00%** (139/139) | **100.00%** (139/139) |
| `bind_value_to_entity` | Reverse lookup: Value $\rightarrow$ Entity | **42.36%** (61/144) | **42.36%** (61/144) |
| `compare_2way` | 2-Candidate Min/Max Comparison | **73.95%** (369/499) | **76.35%** (381/499) |
| `compare_3way` | 3-Candidate Min/Max Comparison | **62.79%** (670/1067) | **56.98%** (608/1067) |
| `compare_4way` | 4-Candidate Min/Max Comparison | **46.74%** (538/1151) | **44.05%** (507/1151) |

---

## 🎓 Two-Stage Curriculum Fine-Tuning Strategy

The fine-tuning pipeline employs a **Two-Stage Curriculum Learning** approach:

1. **Stage 1 (Soft Fine-Tuning)**:
   - **Goal**: Warm up the pretrained base checkpoint (`hindi_gpt.pt` / `nepali_gpt.pt`) on foundational key-value entity bindings and simpler comparative statements.
   - **Recipe**: AdamW optimizer (`lr=2e-5`, `weight_decay=0.01`), cosine annealing schedule with 6% warmup, batch size 32, 3 epochs.
   - **First-Token Weighted Loss**: Cross-entropy loss assigns a **3.0x multiplier** to the first target answer token to force sharp entity selection.

2. **Stage 2 (Hard Fine-Tuning)**:
   - **Goal**: Fine-tune starting from the Stage 1 soft checkpoint (`soft_best.pt`) on multi-candidate reasoning (3-way & 4-way comparisons with varied attributes, units, and name orders).
   - **Recipe**: AdamW optimizer (`lr=1e-5`, `weight_decay=0.01`), cosine annealing with warmup, batch size 32, 3 epochs.
   - Saves the final high-performing checkpoint `hard_best.pt`.

---

## 🚀 Reproduction Steps

### Prerequisites & Dependencies

Install required Python packages:

```bash
pip install torch sentencepiece tqdm matplotlib numpy
```

---

### Reproduction Step 1: Fine-Tuning Model H (Hindi GPT)

#### Step 1.1: Generate Datasets
Run dataset generation scripts from the `Hindi/` directory:

```bash
cd Hindi
python datagen/soft_gen.py
python datagen/hard_datagen.py
```
*Generates `soft_train.jsonl`, `soft_val.jsonl`, `hard_train.jsonl`, `hard_val.jsonl`, and `hard_test.jsonl` in the target data directory.*

#### Step 1.2: Run Stage 1 Soft Fine-Tuning
```bash
python finetune/soft_finetuned.py
```
*Trains on `soft_train.jsonl` and saves `soft_best.pt`.*

#### Step 1.3: Run Stage 2 Hard Fine-Tuning
```bash
python finetune/hard_after_soft.py
```
*Loads `soft_best.pt`, trains on `hard_train.jsonl`, and saves `hard_best.pt`.*

#### Step 1.4: Run Final Test Evaluation
```bash
python eval/final_eval_hard.py
```
*Evaluates `hard_best.pt` against `hard_test.jsonl` and outputs overall & per-task exact-match accuracy.*

#### Step 1.5: Run Attention Diagnostics
```bash
python eval/attention_analysis.py
```
*Extracts attention heatmaps across transformer layers/heads into `Hindi/attention_analysis/`.*

---

### Reproduction Step 2: Fine-Tuning Model N (Nepali GPT)

#### Step 2.1: Generate Datasets
Run dataset generation scripts from the `Nepali/` directory:

```bash
cd Nepali
python datagen/soft_gen.py
python datagen/hard_datagen.py
```

#### Step 2.2: Run Stage 1 Soft Fine-Tuning
```bash
python finetune/soft_finetuned.py
```

#### Step 2.3: Run Stage 2 Hard Fine-Tuning
```bash
python finetune/hard_after_soft.py
```

#### Step 2.4: Run Final Test Evaluation
```bash
python eval/final_eval_hard.py
```

#### Step 2.5: Run Attention Diagnostics
```bash
python eval/attention_analysis.py
```

---

### Reproduction Step 3: Bonus Ablation Study (Positional Embeddings Disabled)

To reproduce the positional embedding ablation experiment:

```bash
cd Bonus
python -m train.train
python -m eval.eval_generation
python -m eval.attention_analysis
```

---

## ⚙️ Environment Variable Overrides

All scripts automatically resolve local data paths, but you can override any input/output path using standard environment variables:

| Environment Variable | Description | Default Fallback |
| :--- | :--- | :--- |
| `TOKENIZER_PATH` | Path to SentencePiece `.model` file | `<model_dir>/data/<lang>_bpe_32000.model` |
| `BASE_CHECKPOINT` | Pretrained base model checkpoint | `<model_dir>/data/<lang>_gpt.pt` |
| `SOFT_CHECKPOINT` | Stage 1 soft checkpoint | `<model_dir>/data/soft_best.pt` |
| `CHECKPOINT_PATH` | Model checkpoint for evaluation | `<model_dir>/data/hard_best.pt` |
| `TRAIN_PATH` | Training JSONL dataset path | `<model_dir>/data/<stage>_train.jsonl` |
| `VAL_PATH` | Validation JSONL dataset path | `<model_dir>/data/<stage>_val.jsonl` |
| `TEST_PATH` | Test JSONL dataset path | `<model_dir>/data/hard_test.jsonl` |
| `OUT_DIR` | Output directory for dataset generator | `<model_dir>/data/` |
| `SAVE_PATH` | Output path for saved checkpoints | `<model_dir>/data/<stage>_best.pt` |
