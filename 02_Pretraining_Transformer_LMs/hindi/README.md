# Model H: Hindi Decoder-only Transformer LM

This directory contains the complete Hindi training and evaluation code, the
SentencePiece tokenizer, raw text data, and a trained checkpoint. The binary
training files are created locally by the training code and are not uploaded.

## Model download

The trained Hindi checkpoint is also available here:

<https://www.kaggle.com/models/satyagudu/hindi-gpt-model-h>

## Requirements

- Python 3.10 or newer
- NVIDIA GPU with a compatible driver for the CUDA PyTorch build
- About 6 GB of GPU memory for the provided model configuration

Install the dependencies from this directory:

```powershell
cd C:\Users\satya\Desktop\PHASE2\submission\hindi_project
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch. For CUDA 12.8:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Confirm the runtime before training or evaluation:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Files

```text
configs/config.yaml             hyperparameters and paths
model/gpt.py                    decoder-only GPT architecture
train/train.py                  pretraining entry point
train/checkpoint.py             checkpoint save/load/resume
eval/generate.py                greedy, sampling, and beam decoding
eval/evaluate.py                generation-quality evaluation
eval/metrics.py                 BLEU, chrF, chrF++, ROUGE-L, diversity metrics
eval/attention_heatmap.py       attention maps, entropy, and distance analysis
```

## Reproduce training

Run commands from this directory. The config uses CUDA, BFloat16, and the raw
files `hindi_train.txt` and `hindi_val.txt`. The first startup step tokenizes
the raw corpus and creates the required `.bin` files in `data/`. These generated
files are local and do not need to be uploaded.

```powershell
python -m train.train --config configs/config.yaml
```

Training checkpoints are written to `checkpoints/hindi/`. To resume from a
checkpoint found elsewhere, pass an additional search root:

```powershell
python -m train.train --config configs/config.yaml --resume-search-root C:\path\to\previous\output
```

## Evaluate the packaged checkpoint

The included checkpoint is `hindi_gpt.pt`. Use it directly for an immediate
evaluation, or replace it with `checkpoints/hindi/best.pt` after training.

```powershell
New-Item -ItemType Directory -Force results
python -m eval.evaluate --config configs/config.yaml --checkpoint hindi_gpt.pt --n-examples 50 --out-json results/hindi_eval.json
```

## Attention analysis

```powershell
python -m eval.attention_heatmap --config configs/config.yaml --checkpoint hindi_gpt.pt --out results/hindi_attention_heatmap.png
```

This also writes entropy and mean-distance plots beside the heatmap.

## Architecture and training configuration

32,000-token vocabulary, `d_model=448`, 6 layers, 8 attention heads,
`d_ff=1024`, sequence length 512, dropout 0.1, and approximately 24.9M
parameters. The output projection is tied to the token embedding weights.
