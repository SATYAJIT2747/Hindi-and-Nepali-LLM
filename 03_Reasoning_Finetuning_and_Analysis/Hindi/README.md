# Model H (Hindi GPT) — Hard Reasoning Evaluation & Comparison Report

This directory contains the dataset generation, fine-tuning scripts, evaluation pipeline, logs, and attention analysis artifacts for **Model H (Hindi GPT)**.

---

## 📌 Executive Summary

* **Architecture**: 6-layer Causal Transformer (`d_model=448`, `nhead=8`, `ffn_dim=1024`, ~24.9M parameters).
* **Tokenizer**: Hindi BPE Tokenizer (`hindi_bpe_32000.model`), trained on Devanagari Hindi text.
* **Finetuned Checkpoint**: Kaggle / Local hard fine-tuned model checkpoint (`hard_final_best.pt`).
* **Kaggle Datasets & Links**:
  - 🔤 **Tokenizer Model & Vocab**: [hindi-bpe-32k (Kaggle)](https://www.kaggle.com/datasets/satyagudu/hindi-bpe-32k)
  - 📥 **Finetuning Dataset**: [hindi-finetune-data (Kaggle)](https://www.kaggle.com/datasets/satyagudu/hindi-finetune-data)
  - 💾 **Finetuned Model Checkpoint**: [hindi-finetuned-model (Kaggle)](https://www.kaggle.com/datasets/satyagudu/hindi-finetuned-model)

---

## 📊 Pretrained vs. Finetuned Accuracy (Synthetic Hard Test Set)

Evaluated on `hard_test.jsonl` (3,000 synthetic multi-entity comparative reasoning prompts):

| Evaluation Metric | Pretrained Model (`hindi_gpt.pt`) | Hard Finetuned Model  | Gain |
| :--- | :--- | :--- | :--- |
| **Validation Best Accuracy** | 0.00% | **60.50%** | **+60.50%** |
| **Test Exact Match Accuracy** | 0.00% | **59.23%** (1777/3000) | **+59.23%** |
| **First-Token Accuracy** | 0.00% | **59.23%** (1777/3000) | **+59.23%** |

### Task-Wise Accuracy Breakdown (Hard Test Set)

| Task Type | Description | Model H Accuracy | Correct / Total |
| :--- | :--- | :--- | :--- |
| `bind_entity_to_value` | Direct lookup: Entity $\rightarrow$ Value | **100.00%** | 139 / 139 |
| `bind_value_to_entity` | Reverse lookup: Value $\rightarrow$ Entity | **42.36%** | 61 / 144 |
| `compare_2way` | 2-Entity Min/Max Comparison | **73.95%** | 369 / 499 |
| `compare_3way` | 3-Entity Min/Max Comparison | **62.79%** | 670 / 1067 |
| `compare_4way` | 4-Entity Min/Max Comparison | **46.74%** | 538 / 1151 |

---

## 🎯 Qualitative Successes & Failures

### ✅ Qualitative Successes

1. **Direct Entity Binding (`100.00%`)**:
   ```text
   Prompt:
   दीपक का अंक = 86
   सुनीता का अंक = 88

   दीपक का अंक क्या है?
   उत्तर:
   Gold: 86
   Pred: 86  [CORRECT]
   ```

2. **3-Way Comparative Reasoning (`62.79%`)**:
   ```text
   Prompt:
   पूजा की उम्र = 65 वर्ष
   अमित की उम्र = 61 वर्ष
   सुरेश की उम्र = 58 वर्ष

   सबसे कम उम्र किसकी है?
   उत्तर:
   Gold: सुरेश
   Pred: सुरेश  [CORRECT]
   ```

---

### ❌ Qualitative Failures & Error Analysis

1. **4-Way Multi-Entity Comparison Bottleneck (`46.74%`)**:
   ```text
   Prompt:
   दीपक के अंक = 70
   नीलम के अंक = 79
   राजू के अंक = 76
   सुरेश के अंक = 77

   अधिकतम अंक किसके हैं?
   उत्तर:
   Gold: नीलम
   Pred: सुरेश  [FAIL]
   ```
   * *Root Cause*: As the comparison scales to 4 candidates, key-value cross-attention gets split across multiple entity lines.

2. **Reverse Binding Lookup (`42.36%`)**:
   ```text
   Prompt:
   राजू के मूल्य = 33 रुपये
   दीपक का मूल्य 65 रुपये है।
   संजय के मूल्य = 58 रुपये

   65 रुपये किसके हैं?
   उत्तर:
   Gold: दीपक
   Pred: संजय  [FAIL]
   ```

---
