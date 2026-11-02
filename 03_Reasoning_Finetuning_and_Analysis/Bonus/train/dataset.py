import numpy as np
import torch
import sentencepiece as spm

def tokenize_corpus_to_bin(text_path, sp_model_path, out_path, dtype=np.uint16):
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    eos = sp.eos_id() if sp.eos_id() >= 0 else 0
    ids = []
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.extend(sp.encode(line, out_type=int))
            ids.append(eos)
    arr = np.array(ids, dtype=dtype)
    arr.tofile(out_path)
    print(f"Created {out_path}: {len(arr):,} tokens")
    return len(arr)

def compute_bytes_per_token(text_path, sp_model_path):
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    n_bytes, n_tokens = 0, 0
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_bytes += len(line.encode("utf-8"))
            n_tokens += len(sp.encode(line, out_type=int))
    return n_bytes / n_tokens

class MemmapDataset:
    def __init__(self, bin_path, seq_len, dtype=np.uint16):
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.seq_len = seq_len

    def get_batch(self, batch_size, device):
        ix = np.random.randint(0, len(self.data) - self.seq_len - 1, size=batch_size)
        x = np.stack([self.data[i:i + self.seq_len].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.seq_len].astype(np.int64) for i in ix])
        return (
            torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True),
        )
