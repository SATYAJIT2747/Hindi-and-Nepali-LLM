# Model L (Nepali GPT) — Hard Reasoning Evaluation & Comparison Report

This directory contains the dataset generation, fine-tuning scripts, evaluation pipeline, logs, and attention analysis artifacts for **Model L (Nepali GPT)**.

---

## 📌 Executive Summary

* **Architecture**: 6-layer Causal Transformer (`d_model=448`, `nhead=8`, `ffn_dim=1024`, ~24.9M parameters).
* **Tokenizer**: Nepali BPE Tokenizer (`nepali_bpe_32000.model`), trained on Nepali Devanagari script.
* **Numeral Tokenization Fix**: All numbers are formatted using **Devanagari numerals** (`०१२३४५६७८९`), eliminating `<unk>` token degradation.
* **Kaggle Datasets & Links**:
  - 🔤 **Tokenizer Model & Vocab**: [neplai-vocab-and-model (Kaggle)](https://www.kaggle.com/datasets/satyagudu/neplai-vocab-and-model)
  - 📥 **Finetuning Dataset**: [neplai-finetune-data (Kaggle)](https://www.kaggle.com/datasets/satyagudu/neplai-finetune-data)
  - 💾 **Finetuned Model Checkpoint**: [neplai-finetuned-model (Kaggle)](https://www.kaggle.com/datasets/satyagudu/neplai-finetuned-model)

---

## 📊 Pretrained vs. Finetuned Accuracy (Synthetic Hard Test Set)

Evaluated on `hard_test.jsonl` (3,000 synthetic multi-entity comparative reasoning prompts):

| Evaluation Metric | Pretrained Model (`nepali_gpt.pt`) | Hard Finetuned Model (`hard_best.pt`) | Gain |
| :--- | :--- | :--- | :--- |
| **Validation Best Accuracy** | 0.00% | **62.17%** | **+62.17%** |
| **Test Exact Match Accuracy** | 0.00% | **56.53%** (1696/3000) | **+56.53%** |
| **First-Token Accuracy** | 0.00% | **56.53%** (1696/3000) | **+56.53%** |

### Task-Wise Accuracy Breakdown (Hard Test Set)

| Task Type | Description | Model L Accuracy | Correct / Total |
| :--- | :--- | :--- | :--- |
| `bind_entity_to_value` | Direct lookup: Entity $\rightarrow$ Value | **100.00%** | 139 / 139 |
| `bind_value_to_entity` | Reverse lookup: Value $\rightarrow$ Entity | **42.36%** | 61 / 144 |
| `compare_2way` | 2-Entity Min/Max Comparison | **76.35%** | 381 / 499 |
| `compare_3way` | 3-Entity Min/Max Comparison | **56.98%** | 608 / 1067 |
| `compare_4way` | 4-Entity Min/Max Comparison | **44.05%** | 507 / 1151 |

---

## 🎯 Qualitative Successes & Failures

### ✅ Qualitative Successes

1. **Direct Entity Binding (`100.00%`)**:
   ```text
   Prompt:
   दीपकको मूल्य = ७० रुपैयाँ
   सुरेशको मूल्य = ७७ रुपैयाँ

   दीपकको मूल्य कति हो?
   उत्तर:
   Gold: ७० रुपैयाँ
   Pred: ७० रुपैयाँ  [CORRECT]
   ```

2. **2-Way Comparative Reasoning (`76.35%`)**:
   ```text
   Prompt:
   दीपकका अंक = ७०
   सुरेशका अंक = ७७

   सबैभन्दा बढी अंक कसको छ?
   उत्तर:
   Gold: सुरेश
   Pred: सुरेश  [CORRECT]
   ```

---

### ❌ Qualitative Failures & Error Analysis

1. **4-Way Comparative Capacity Decay (`44.05%`)**:
   ```text
   Prompt:
   राजू: ३३ रुपैयाँ
   दीपक: ३४ रुपैयाँ
   रोहित: ३१ रुपैयाँ
   पूजा: ३५ रुपैयाँ

   सबैभन्दा कम मूल्य
   उत्तर:
   Gold: रोहित
   Pred: पूजा  [FAIL]
   ```
   * *Root Cause*: In 4-way comparison, small ~25M parameter models struggle to maintain 4 active entity-value bindings in working memory and rank them accurately.

2. **Reverse Binding Asymmetry (`42.36%`)**:
   ```text
   Prompt:
   नीलमको उमेर ९५ वर्ष छ।
   रोहितको उमेर २५ वर्ष छ।
   राहुल: ६५ वर्ष

   ९५ वर्ष कसको हो?
   उत्तर:
   Gold: नीलम
   Pred: राहुल  [FAIL]
   ```
   * *Root Cause*: Autoregressive causal masking makes retrieving a prior entity given a later value much harder than forward retrieval.

---


