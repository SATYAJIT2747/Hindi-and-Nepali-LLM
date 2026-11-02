# =============================================================================
# V7 FINAL EVALUATION — EXACT V7 ARCHITECTURE
# =============================================================================

import os
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_path(env_key, default_name, kaggle_path=None):
    env_val = os.getenv(env_key)
    if env_val and os.path.exists(env_val):
        return env_val
    if kaggle_path and os.path.exists(kaggle_path):
        return kaggle_path
    local_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data", default_name))
    return local_p

CHECKPOINT_PATH = resolve_path("CHECKPOINT_PATH", "hard_best.pt", "/kaggle/working/v7_best.pt")
TOKENIZER_PATH = resolve_path(
    "TOKENIZER_PATH", 
    "hindi_bpe_32000.model",
    "/kaggle/input/datasets/satyagudu/hindi-bpe-32k/hindi_bpe_32000.model"
)
TEST_PATH = resolve_path("TEST_PATH", "hard_test.jsonl", "/kaggle/working/hard_test.jsonl")


# =============================================================================
# MODEL CONFIG — EXACTLY MATCHES V7 TRAINING
# =============================================================================

MAX_SEQ_LEN = 512
D_MODEL = 448
NHEAD = 8
DIM_FEEDFORWARD = 1024
NUM_LAYERS = 6
DROPOUT = 0.1


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("V7 FINAL EVALUATION")
print("=" * 80)

print(f"Device     : {DEVICE}")
print(f"Checkpoint : {CHECKPOINT_PATH}")
print(f"Test file  : {TEST_PATH}")
print()


# =============================================================================
# CHECK FILES
# =============================================================================

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{CHECKPOINT_PATH}"
    )

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"\nTokenizer not found:\n{TOKENIZER_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"\nTest file not found:\n{TEST_PATH}"
    )


# =============================================================================
# TOKENIZER
# =============================================================================

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

VOCAB_SIZE = sp.get_piece_size()
EOS_ID = sp.eos_id()
PAD_ID = sp.pad_id()

print("Tokenizer loaded.")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"EOS ID: {EOS_ID}")
print(f"PAD ID: {PAD_ID}")
print()


# =============================================================================
# MODEL
# =============================================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        max_seq_len
    ):
        super().__init__()

        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(
            d_model,
            3 * d_model
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model
        )

        # EXACT checkpoint shape:
        # [1, 1, 512, 512]

        mask = torch.triu(
            torch.ones(
                max_seq_len,
                max_seq_len
            ),
            diagonal=1
        ).bool()

        self.register_buffer(
            "causal_mask",
            mask.view(
                1,
                1,
                max_seq_len,
                max_seq_len
            ),
            persistent=True
        )


    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )

        q = q.view(
            B,
            T,
            self.nhead,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.nhead,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.nhead,
            self.head_dim
        ).transpose(1, 2)

        attn_mask = self.causal_mask[
            :,
            :,
            :T,
            :T
        ]

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=~attn_mask,
            dropout_p=0.0,
            is_causal=False
        )

        y = y.transpose(
            1,
            2
        ).contiguous()

        y = y.view(
            B,
            T,
            C
        )

        return self.out_proj(y)


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model=448,
        nhead=8,
        dim_feedforward=1024,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(
            d_model
        )

        self.attn = CausalSelfAttention(
            d_model,
            nhead,
            max_seq_len
        )

        self.ln2 = nn.LayerNorm(
            d_model
        )

        # IMPORTANT:
        # Must be fc1 / fc2
        # NOT nn.Sequential()

        self.ffn = nn.ModuleDict({

            "fc1": nn.Linear(
                d_model,
                dim_feedforward
            ),

            "fc2": nn.Linear(
                dim_feedforward,
                d_model
            )

        })

        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        x = x + self.dropout(
            self.attn(
                self.ln1(x)
            )
        )

        x = x + self.dropout(
            self.ffn["fc2"](
                F.gelu(
                    self.ffn["fc1"](
                        self.ln2(x)
                    )
                )
            )
        )

        return x


# =============================================================================
# HINDI GPT
# =============================================================================

class HindiGPT(nn.Module):

    def __init__(self):

        super().__init__()

        self.vocab_size = VOCAB_SIZE
        self.d_model = D_MODEL
        self.max_seq_len = MAX_SEQ_LEN

        self.tok_emb = nn.Embedding(
            self.vocab_size,
            self.d_model
        )

        self.pos_emb = nn.Embedding(
            self.max_seq_len,
            self.d_model
        )

        self.drop = nn.Dropout(
            DROPOUT
        )

        self.blocks = nn.ModuleList([

            TransformerBlock(
                d_model=D_MODEL,
                nhead=NHEAD,
                dim_feedforward=DIM_FEEDFORWARD,
                max_seq_len=MAX_SEQ_LEN,
                dropout=DROPOUT
            )

            for _ in range(NUM_LAYERS)

        ])

        self.ln_f = nn.LayerNorm(
            self.d_model
        )

        self.lm_head = nn.Linear(
            self.d_model,
            self.vocab_size,
            bias=False
        )

        # EXACT V7 / V6 weight tying
        self.lm_head.weight = self.tok_emb.weight


    def forward(self, idx):

        B, T = idx.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )

        pos = torch.arange(
            T,
            device=idx.device
        )

        x = (
            self.tok_emb(idx)
            +
            self.pos_emb(pos)
        )

        x = self.drop(x)

        for block in self.blocks:

            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits


# =============================================================================
# LOAD MODEL
# =============================================================================

print("=" * 80)
print("LOADING V7 MODEL")
print("=" * 80)

model = HindiGPT().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

print(
    f"Checkpoint type: {type(checkpoint)}"
)


# =============================================================================
# EXTRACT STATE DICT
# =============================================================================

if isinstance(checkpoint, dict) and "model" in checkpoint:

    state_dict = checkpoint["model"]

    print(
        "Checkpoint dictionary detected."
    )

else:

    state_dict = checkpoint

    print(
        "Raw state_dict detected."
    )


# IMPORTANT:
# Make a copy so we don't modify checkpoint unexpectedly

state_dict = state_dict.copy()


# =============================================================================
# CHECK ARCHITECTURE
# =============================================================================

print()
print("=" * 80)
print("ARCHITECTURE CHECK")
print("=" * 80)

model_state = model.state_dict()

missing = []
unexpected = []
shape_errors = []


# Check checkpoint keys

for key, value in state_dict.items():

    if "causal_mask" in key:

        continue

    if key not in model_state:

        unexpected.append(
            key
        )

    else:

        if tuple(value.shape) != tuple(
            model_state[key].shape
        ):

            shape_errors.append(
                (
                    key,
                    tuple(value.shape),
                    tuple(model_state[key].shape)
                )
            )


# Check model keys

for key in model_state.keys():

    if "causal_mask" in key:

        continue

    if key not in state_dict:

        missing.append(
            key
        )


print()

print(
    f"Model D_MODEL        : {D_MODEL}"
)

print(
    f"Model NHEAD          : {NHEAD}"
)

print(
    f"Model FFN dimension  : {DIM_FEEDFORWARD}"
)

print(
    f"Model layers         : {NUM_LAYERS}"
)

print(
    f"Model max sequence   : {MAX_SEQ_LEN}"
)

print(
    f"Model vocabulary     : {VOCAB_SIZE}"
)

print()


# =============================================================================
# STOP IF REAL ARCHITECTURE MISMATCH
# =============================================================================

if missing:

    print(
        "MISSING KEYS:"
    )

    for key in missing:

        print(
            " ",
            key
        )


if unexpected:

    print(
        "UNEXPECTED KEYS:"
    )

    for key in unexpected:

        print(
            " ",
            key
        )


if shape_errors:

    print(
        "SHAPE MISMATCHES:"
    )

    for (
        key,
        ckpt_shape,
        model_shape
    ) in shape_errors:

        print(
            f"{key}: "
            f"checkpoint={ckpt_shape}, "
            f"model={model_shape}"
        )


if missing or unexpected or shape_errors:

    raise RuntimeError(
        "\n❌ REAL ARCHITECTURE MISMATCH.\n"
        "Evaluation stopped."
    )


print(
    "✓ Architecture matches V7 checkpoint."
)


# =============================================================================
# REMOVE CAUSAL MASKS BEFORE LOADING
# =============================================================================

for key in list(state_dict.keys()):

    if "causal_mask" in key:

        del state_dict[key]

        print(
            f"Removed checkpoint buffer: {key}"
        )


# =============================================================================
# LOAD CHECKPOINT
# =============================================================================

missing_after, unexpected_after = model.load_state_dict(
    state_dict,
    strict=False
)


# Only causal masks are allowed to be missing

real_missing = [

    key

    for key in missing_after

    if "causal_mask" not in key

]


if real_missing:

    print(
        "\nERROR: Real missing parameters:"
    )

    for key in real_missing:

        print(
            " ",
            key
        )

    raise RuntimeError(
        "Checkpoint loading failed."
    )


if unexpected_after:

    print(
        "\nERROR: Unexpected parameters:"
    )

    for key in unexpected_after:

        print(
            " ",
            key
        )

    raise RuntimeError(
        "Checkpoint loading failed."
    )


model.eval()


print()
print(
    "✓ V7 checkpoint loaded successfully."
)

print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


if isinstance(checkpoint, dict):

    if "best_accuracy" in checkpoint:

        print(
            f"Training best accuracy: "
            f"{checkpoint['best_accuracy']:.2f}%"
        )

    if "best_step" in checkpoint:

        print(
            f"Best checkpoint step: "
            f"{checkpoint['best_step']}"
        )


print()


# =============================================================================
# LOAD TEST DATA
# =============================================================================

test_data = []

with open(
    TEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:

            continue

        test_data.append(
            json.loads(line)
        )


print(
    f"Test examples loaded: "
    f"{len(test_data):,}"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_prompt(
    example
):

    return example["prompt"]


def get_target(
    example
):

    return example["target"]


def get_task(
    example
):

    return example.get(
        "task",
        "unknown"
    )


# =============================================================================
# FIRST TOKEN PREDICTION
# =============================================================================

@torch.no_grad()
def predict_first_token(
    prompt_text
):

    prompt_ids = sp.encode_as_ids(
        prompt_text
    )

    if len(prompt_ids) == 0:

        raise ValueError(
            "Prompt produced no tokens."
        )

    if len(prompt_ids) > MAX_SEQ_LEN:

        prompt_ids = prompt_ids[
            -MAX_SEQ_LEN:
        ]

    x = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE
    )

    logits = model(x)

    pred_id = logits[
        0,
        -1
    ].argmax().item()

    return pred_id


# =============================================================================
# GREEDY GENERATION
# =============================================================================

@torch.no_grad()
def generate_answer(
    prompt_text,
    max_new_tokens=12
):

    prompt_ids = sp.encode_as_ids(
        prompt_text
    )

    if len(prompt_ids) == 0:

        raise ValueError(
            "Prompt produced no tokens."
        )

    generated = list(
        prompt_ids
    )

    for _ in range(
        max_new_tokens
    ):

        context = generated[
            -MAX_SEQ_LEN:
        ]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=DEVICE
        )

        logits = model(x)

        next_id = logits[
            0,
            -1
        ].argmax().item()

        generated.append(
            next_id
        )

        if next_id == EOS_ID:

            break


    answer_ids = generated[
        len(prompt_ids):
    ]

    # Remove EOS

    if EOS_ID in answer_ids:

        eos_index = answer_ids.index(
            EOS_ID
        )

        answer_ids = answer_ids[
            :eos_index
        ]


    answer = sp.decode_ids(
        answer_ids
    ).strip()

    return answer


# =============================================================================
# DIAGNOSTIC FUNCTION
# =============================================================================

@torch.no_grad()
def print_diagnostic(
    example,
    index
):

    prompt = get_prompt(
        example
    )

    gold = get_target(
        example
    )

    prompt_ids = sp.encode_as_ids(
        prompt
    )

    x = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE
    )

    logits = model(x)

    probs = torch.softmax(
        logits[0, -1],
        dim=-1
    )

    top_probs, top_ids = torch.topk(
        probs,
        8
    )

    pred = generate_answer(
        prompt
    )

    print()
    print(
        f"--- Diagnostic {index} ---"
    )

    print(
        f"Gold: {gold}"
    )

    print(
        f"Pred: {pred}"
    )

    print(
        "Top-8 next-token predictions:"
    )

    for token_id, prob in zip(
        top_ids.tolist(),
        top_probs.tolist()
    ):

        token = sp.id_to_piece(
            token_id
        )

        token = token.replace(
            "▁",
            ""
        )

        print(
            f"  {repr(token):<25} "
            f"{prob:.4f}"
        )


# =============================================================================
# CRITICAL DIAGNOSTICS
# =============================================================================

print()
print("=" * 80)
print("CRITICAL DIAGNOSTICS")
print("=" * 80)


NUM_DIAGNOSTICS = min(
    10,
    len(test_data)
)


for i in range(
    NUM_DIAGNOSTICS
):

    print_diagnostic(
        test_data[i],
        i + 1
    )


# =============================================================================
# FULL TEST SET EVALUATION
# =============================================================================

print()
print("=" * 80)
print("FULL TEST SET EVALUATION")
print("=" * 80)


first_token_correct = 0
exact_match_correct = 0
total = 0


task_correct = defaultdict(
    int
)

task_total = defaultdict(
    int
)


entity_correct = defaultdict(
    int
)

entity_total = defaultdict(
    int
)


position_correct = defaultdict(
    int
)

position_total = defaultdict(
    int
)


failures = []


# =============================================================================
# GET ENTITY COUNT
# =============================================================================

def get_entity_count(
    example
):

    task = get_task(
        example
    )

    if "2way" in task:

        return 2

    if "3way" in task:

        return 3

    if "4way" in task:

        return 4

    return None


# =============================================================================
# GET ANSWER POSITION
# =============================================================================

def get_answer_position(
    example
):

    if "answer_position" in example:

        return example[
            "answer_position"
        ]

    return None


# =============================================================================
# EVALUATE
# =============================================================================

for example in tqdm(
    test_data,
    desc="Evaluating"
):

    prompt = get_prompt(
        example
    )

    gold = get_target(
        example
    ).strip()

    task = get_task(
        example
    )


    # ---------------------------------------------------------
    # First token
    # ---------------------------------------------------------

    gold_ids = sp.encode_as_ids(
        gold
    )

    if len(gold_ids) == 0:

        continue


    pred_first_id = predict_first_token(
        prompt
    )

    first_correct = (
        pred_first_id == gold_ids[0]
    )


    # ---------------------------------------------------------
    # Full generation
    # ---------------------------------------------------------

    pred = generate_answer(
        prompt
    )

    exact_correct = (
        pred.strip() == gold.strip()
    )


    # ---------------------------------------------------------
    # Global
    # ---------------------------------------------------------

    total += 1

    first_token_correct += int(
        first_correct
    )

    exact_match_correct += int(
        exact_correct
    )


    # ---------------------------------------------------------
    # Task statistics
    # ---------------------------------------------------------

    task_total[task] += 1

    task_correct[task] += int(
        exact_correct
    )


    # ---------------------------------------------------------
    # Entity statistics
    # ---------------------------------------------------------

    entity_count = get_entity_count(
        example
    )

    if entity_count is not None:

        key = (
            f"{entity_count}-entity"
        )

        entity_total[key] += 1

        entity_correct[key] += int(
            exact_correct
        )


    # ---------------------------------------------------------
    # Answer position
    # ---------------------------------------------------------

    position = get_answer_position(
        example
    )

    if position is not None:

        key = f"pos{position}"

        position_total[key] += 1

        position_correct[key] += int(
            exact_correct
        )


    # ---------------------------------------------------------
    # Save failures
    # ---------------------------------------------------------

    if not exact_correct:

        if len(failures) < 30:

            failures.append({

                "task": task,

                "position": position,

                "prompt": prompt,

                "gold": gold,

                "pred": pred

            })


# =============================================================================
# RESULTS
# =============================================================================

first_token_accuracy = (
    100.0
    * first_token_correct
    / total
)

exact_match_accuracy = (
    100.0
    * exact_match_correct
    / total
)


print()
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(
    f"First-Token Accuracy : "
    f"{first_token_correct}/{total} "
    f"({first_token_accuracy:.2f}%)"
)

print(
    f"Exact Match Accuracy : "
    f"{exact_match_correct}/{total} "
    f"({exact_match_accuracy:.2f}%)"
)


# =============================================================================
# TASK RESULTS
# =============================================================================

print()
print("--- Accuracy by Task ---")

for task in sorted(
    task_total
):

    correct = task_correct[task]

    count = task_total[task]

    acc = (
        100.0
        * correct
        / count
    )

    print(
        f"{task:<30}: "
        f"{correct}/{count} "
        f"({acc:.2f}%)"
    )


# =============================================================================
# ENTITY RESULTS
# =============================================================================

if entity_total:

    print()
    print(
        "--- Accuracy by Number of Entities ---"
    )

    for key in sorted(
        entity_total
    ):

        correct = entity_correct[key]

        count = entity_total[key]

        acc = (
            100.0
            * correct
            / count
        )

        print(
            f"{key:<25}: "
            f"{correct}/{count} "
            f"({acc:.2f}%)"
        )


# =============================================================================
# POSITION RESULTS
# =============================================================================

if position_total:

    print()
    print(
        "--- Accuracy by Answer Position ---"
    )

    def position_sort_key(key):

        try:

            return int(
                key.replace(
                    "pos",
                    ""
                )
            )

        except:

            return 999


    for key in sorted(
        position_total,
        key=position_sort_key
    ):

        correct = position_correct[key]

        count = position_total[key]

        acc = (
            100.0
            * correct
            / count
        )

        print(
            f"{key:<8}: "
            f"{correct}/{count} "
            f"({acc:.2f}%)"
        )


# =============================================================================
# SAMPLE FAILURES
# =============================================================================

print()
print("=" * 80)
print("SAMPLE FAILURES")
print("=" * 80)


if len(failures) == 0:

    print(
        "No failures found! 🎉"
    )

else:

    for i, failure in enumerate(
        failures[:20],
        1
    ):

        print()

        print(
            f"[FAIL {i}] "
            f"Task: {failure['task']}"
        )

        if failure["position"] is not None:

            print(
                f"Answer position: "
                f"{failure['position']}"
            )

        print(
            "Prompt:"
        )

        print(
            failure["prompt"]
        )

        print(
            f"Gold: {failure['gold']}"
        )

        print(
            f"Pred: {failure['pred']}"
        )


# =============================================================================
# SANITY CHECK
# =============================================================================

print()
print("=" * 80)
print("SANITY CHECK")
print("=" * 80)

task_sum = sum(
    task_total.values()
)

task_correct_sum = sum(
    task_correct.values()
)


print(
    f"Overall total   : {total}"
)

print(
    f"Task total      : {task_sum}"
)

print(
    f"Overall correct : "
    f"{exact_match_correct}"
)

print(
    f"Task correct    : "
    f"{task_correct_sum}"
)


if total == task_sum:

    print(
        "✓ Counts are internally consistent."
    )

else:

    print(
        "❌ Count mismatch."
    )


if exact_match_correct == task_correct_sum:

    print(
        "✓ Correct counts are consistent."
    )

else:

    print(
        "❌ Correct count mismatch."
    )


if exact_match_correct <= total:

    print(
        "✓ Accuracy is valid."
    )

else:

    print(
        "❌ Invalid accuracy."
    )


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(
    f"Overall exact-match accuracy: "
    f"{exact_match_accuracy:.2f}%"
)

print()

if "compare_2way" in task_total:

    print(
        "The important numbers are:"
    )

    print(
        "  1. compare_2way"
    )

    print(
        "  2. compare_3way"
    )

    print(
        "  3. compare_4way"
    )

    print(
        "  4. Accuracy by answer position"
    )

    print()

    print(
        "If 3-way/4-way accuracy remains much lower "
        "than 2-way accuracy, the model still has a "
        "multi-entity comparison limitation."
    )


print()
print("=" * 80)
print("V7 EVALUATION COMPLETE")
print("=" * 80)