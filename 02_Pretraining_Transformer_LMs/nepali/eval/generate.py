"""Decoding strategies used for the generation-quality evaluation:
greedy, temperature sampling, top-k, nucleus (top-p), and beam search."""
import random
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, prefix_ids, max_new_tokens=40, temperature=1.0, greedy=False, device="cuda"):
    """Greedy decoding or plain temperature sampling."""
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if greedy:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()


@torch.no_grad()
def generate_topk_topp(model, prefix_ids, max_new_tokens=40, temperature=1.0,
                        top_k=None, top_p=None, device="cuda"):
    """Top-k and/or nucleus (top-p) sampling."""
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # ---------- TOP-K ----------
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value, torch.full_like(logits, float("-inf")), logits)

        # ---------- TOP-P / NUCLEUS ----------
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()


@torch.no_grad()
def generate_beam_search(model, prefix_ids, max_new_tokens=40, num_beams=4, device="cuda"):
    """Simple log-prob beam search (no length penalty)."""
    beams = [(prefix_ids[:], 0.0)]
    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in beams:
            idx = torch.tensor([seq], dtype=torch.long, device=device)[:, -model.max_seq_len:]
            logits, _ = model(idx)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            topk_logp, topk_idx = torch.topk(log_probs, num_beams, dim=-1)
            for lp, tok in zip(topk_logp[0].tolist(), topk_idx[0].tolist()):
                candidates.append((seq + [tok], score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = candidates[:num_beams]
    best_seq, _ = max(beams, key=lambda c: c[1])
    return best_seq


def build_holdout_examples(val_txt_path, sp, n_examples=50, prefix_tokens=20, cont_tokens=40, seed=0):
    """Build (prefix, reference) pairs from held-out validation text for generation eval."""
    random.seed(seed)
    lines = [l.strip() for l in open(val_txt_path, encoding="utf-8") if l.strip()]
    examples = []
    for line in lines:
        ids = sp.encode(line, out_type=int)
        if len(ids) >= prefix_tokens + cont_tokens:
            examples.append(ids)
    random.shuffle(examples)
    examples = examples[:n_examples]
    data = []
    for ids in examples:
        prefix_ids, ref_ids = ids[:prefix_tokens], ids[prefix_tokens:prefix_tokens + cont_tokens]
        data.append({
            "prefix_ids": prefix_ids,
            "ref_ids": ref_ids,
            "prefix_text": sp.decode(prefix_ids),
            "ref_text": sp.decode(ref_ids),
        })
    print(f"Built {len(data)} held-out prefix/reference pairs")
    return data
