# ==============================================================================
# SOFT DATASET GENERATION (NEPALI)
# ==============================================================================

import json
import random
import os
import sys
import hashlib
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)

# ==============================================================================
# PATHS
# ==============================================================================

OUT_DIR = os.getenv("OUT_DIR")
if not OUT_DIR:
    if os.name != "nt" and os.path.exists("/kaggle/working"):
        OUT_DIR = "/kaggle/working"
    else:
        OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(OUT_DIR, "soft_train.jsonl")
VAL_PATH   = os.path.join(OUT_DIR, "soft_val.jsonl")
TEST_PATH  = os.path.join(OUT_DIR, "soft_test.jsonl")

TRAIN_SIZE = 30000
VAL_SIZE   = 3000
TEST_SIZE  = 3000

TOTAL_SIZE = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

# ==============================================================================
# TASK DISTRIBUTION
# ==============================================================================

TASK_RATIOS = {
    "bind_entity_to_value": 0.20,
    "bind_value_to_entity": 0.15,
    "compare_2way":         0.25,
    "compare_3way":         0.20,
    "compare_4way":         0.10,
    "equality":             0.10,
}

# ==============================================================================
# VOCABULARY (NEPALI)
# ==============================================================================

NAMES = [
    "अमित",
    "राहुल",
    "विकास",
    "सुरेश",
    "मोहन",
    "रोहित",
    "दीपक",
    "संजय",
    "कविता",
    "अशोक",
    "पूजा",
    "सुनीता",
    "रमेश",
    "राजू",
    "नीलम",
]

ATTRIBUTES = [
    {
        "name": "अंक",
        "unit": "",
        "min_word": "सबैभन्दा कम",
        "max_word": "सबैभन्दा बढी",
    },
    {
        "name": "तौल",
        "unit": " किलोग्राम",
        "min_word": "सबैभन्दा हलुका",
        "max_word": "सबैभन्दा गह्रौँ",
    },
    {
        "name": "उमेर",
        "unit": " वर्ष",
        "min_word": "सबैभन्दा सानो",
        "max_word": "सबैभन्दा ठूलो",
    },
    {
        "name": "मूल्य",
        "unit": " रुपैयाँ",
        "min_word": "सबैभन्दा सस्तो",
        "max_word": "सबैभन्दा महँगो",
    },
]


def make_exact_quotas(total_size, ratios):
    raw = {k: total_size * v for k, v in ratios.items()}
    quotas = {k: int(v) for k, v in raw.items()}
    remainder = total_size - sum(quotas.values())
    fractional = sorted(
        ratios.keys(),
        key=lambda k: raw[k] - int(raw[k]),
        reverse=True
    )
    for i in range(remainder):
        quotas[fractional[i % len(fractional)]] += 1
    return quotas


quotas = make_exact_quotas(TOTAL_SIZE, TASK_RATIOS)

print("Target task quotas:")
for task, count in quotas.items():
    print(f"  {task:<24}: {count}")

# ==============================================================================
# TRACKERS
# ==============================================================================

seen_prompts = set()
name_tracker = defaultdict(int)
position_tracker = defaultdict(lambda: defaultdict(int))


def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def choose_attribute():
    return random.choice(ATTRIBUTES)


def choose_names(n):
    return random.sample(NAMES, n)


def choose_balanced_position(task_type, n):
    counts = [position_tracker[task_type][i] for i in range(n)]
    minimum = min(counts)
    candidates = [i for i, c in enumerate(counts) if c == minimum]
    return random.choice(candidates)


def generate_distinct_values(n):
    return random.sample(range(10, 100), n)


def construct_values_with_winner(n, target_idx, is_max):
    values = [None] * n
    if is_max:
        winner = random.randint(70, 99)
        values[target_idx] = winner
        for i in range(n):
            if i == target_idx:
                continue
            values[i] = random.randint(10, winner - 1)
    else:
        winner = random.randint(10, 30)
        values[target_idx] = winner
        for i in range(n):
            if i == target_idx:
                continue
            values[i] = random.randint(winner + 1, 99)
    return values


DEVANAGARI_TRANS = str.maketrans("0123456789", "०१२३४५६७८९")

def to_nepali_num(val):
    return str(val).translate(DEVANAGARI_TRANS)


# ==============================================================================
# TEMPLATE GENERATORS (NEPALI)
# ==============================================================================

def make_bind_entity_to_value():
    attr = choose_attribute()
    name = random.choice(NAMES)
    val = random.randint(10, 99)
    val_str = to_nepali_num(val)

    templates = [
        (
            f"{name}को {attr['name']} {val_str}{attr['unit']} छ।\n"
            f"{name}को {attr['name']} कति छ?\n"
            f"उत्तर:"
        ),
        (
            f"{name}को {attr['name']} {val_str}{attr['unit']} छ।\n"
            f"बताउनुहोस् {name}को {attr['name']} कति छ?\n"
            f"उत्तर:"
        ),
        (
            f"{name}सँग {attr['name']} {val_str}{attr['unit']} छ।\n"
            f"{name}को मान कति छ?\n"
            f"उत्तर:"
        ),
    ]

    prompt = random.choice(templates)
    return {
        "prompt": prompt,
        "target": f" {val_str}",
        "task": "bind_entity_to_value",
        "answer_pos": 0,
    }


def make_bind_value_to_entity():
    attr = choose_attribute()
    names = choose_names(2)
    v1, v2 = random.sample(range(10, 100), 2)
    v1_str = to_nepali_num(v1)
    v2_str = to_nepali_num(v2)

    templates = [
        (
            f"{names[0]}को {attr['name']} {v1_str}{attr['unit']} छ "
            f"र {names[1]}को {attr['name']} {v2_str}{attr['unit']} छ।\n"
            f"{v2_str}{attr['unit']} {attr['name']} कसको छ?\n"
            f"उत्तर:"
        ),
        (
            f"{names[0]}को {attr['name']} {v1_str}{attr['unit']} छ, "
            f"जबकि {names[1]}को {attr['name']} {v2_str}{attr['unit']} छ।\n"
            f"{v2_str}{attr['unit']} कसको हो?\n"
            f"उत्तर:"
        ),
        (
            f"{names[0]}: {v1_str}{attr['unit']}\n"
            f"{names[1]}: {v2_str}{attr['unit']}\n\n"
            f"{v2_str}{attr['unit']} कसको छ?\n"
            f"उत्तर:"
        ),
    ]

    prompt = random.choice(templates)
    return {
        "prompt": prompt,
        "target": f" {names[1]}",
        "task": "bind_value_to_entity",
        "answer_pos": 1,
    }


def make_comparison(task_type):
    n = int(task_type.split("_")[1][0])
    attr = choose_attribute()
    names = choose_names(n)

    target_idx = choose_balanced_position(task_type, n)
    is_max = random.choice([True, False])

    values = construct_values_with_winner(
        n=n,
        target_idx=target_idx,
        is_max=is_max
    )

    target_name = names[target_idx]
    position_tracker[task_type][target_idx] += 1
    name_tracker[target_name] += 1

    lines = []
    template_style = random.randint(0, 2)

    if template_style == 0:
        for name, value in zip(names, values):
            lines.append(f"{name}: {to_nepali_num(value)}{attr['unit']}")

        if attr["name"] == "अंक":
            if is_max:
                question = "सबैभन्दा बढी अंक कसको छ?"
            else:
                question = "सबैभन्दा कम अंक कसको छ?"
        else:
            if is_max:
                question = f"को {attr['max_word']} छ?"
            else:
                question = f"को {attr['min_word']} छ?"

        prompt = "\n".join(lines) + f"\n\n{question}\nउत्तर:"

    elif template_style == 1:
        for name, value in zip(names, values):
            lines.append(f"{name}को {attr['name']} {to_nepali_num(value)}{attr['unit']} छ।")

        if is_max:
            question = f"यिनीहरूमध्ये {attr['max_word']} कसको छ?"
        else:
            question = f"यिनीहरूमध्ये {attr['min_word']} कसको छ?"

        prompt = " ".join(lines) + f"\n{question}\nउत्तर:"

    else:
        for name, value in zip(names, values):
            lines.append(f"{name}को {attr['name']} {to_nepali_num(value)}{attr['unit']} छ।")

        if is_max:
            question = f"सबैभन्दा बढी {attr['name']} कसको छ?"
        else:
            question = f"सबैभन्दा कम {attr['name']} कसको छ?"

        prompt = " ".join(lines) + f"\n{question}\nउत्तर:"

    return {
        "prompt": prompt,
        "target": f" {target_name}",
        "task": task_type,
        "answer_pos": target_idx,
    }


def make_equality():
    attr = choose_attribute()
    names = choose_names(2)
    value = random.randint(10, 99)
    val_str = to_nepali_num(value)

    templates = [
        (
            f"{names[0]}: {val_str}{attr['unit']}\n"
            f"{names[1]}: {val_str}{attr['unit']}\n\n"
            f"कसको बढी {attr['name']} छ?\n"
            f"उत्तर:"
        ),
        (
            f"{names[0]}को {attr['name']} {val_str}{attr['unit']} छ र "
            f"{names[1]}को {attr['name']} पनि {val_str}{attr['unit']} छ।\n"
            f"कसको {attr['name']} बढी छ?\n"
            f"उत्तर:"
        ),
        (
            f"{names[0]}को {attr['name']} {val_str}{attr['unit']} छ। "
            f"{names[1]}को {attr['name']} पनि {val_str}{attr['unit']} छ।\n"
            f"को बढी छ?\n"
            f"उत्तर:"
        ),
    ]

    prompt = random.choice(templates)
    return {
        "prompt": prompt,
        "target": " दुवै समान छन्",
        "task": "equality",
        "answer_pos": -1,
    }


# ==============================================================================
# MAIN GENERATOR
# ==============================================================================

GENERATORS = {
    "bind_entity_to_value": make_bind_entity_to_value,
    "bind_value_to_entity": make_bind_value_to_entity,
    "compare_2way": lambda: make_comparison("compare_2way"),
    "compare_3way": lambda: make_comparison("compare_3way"),
    "compare_4way": lambda: make_comparison("compare_4way"),
    "equality": make_equality,
}

generated_data = []

print("\nGenerating NEPALI SOFT dataset...")
print("No rejection-based generation is used.\n")

for task_type, count in quotas.items():
    generated_for_task = 0
    attempts = 0

    while generated_for_task < count:
        attempts += 1
        if attempts > count * 20:
            raise RuntimeError(
                f"Too many duplicate prompts for {task_type}. "
                f"Generated {generated_for_task}/{count}."
            )

        example = GENERATORS[task_type]()
        prompt = example["prompt"]
        prompt_hash = get_hash(prompt)

        if prompt_hash in seen_prompts:
            continue

        seen_prompts.add(prompt_hash)
        generated_data.append(example)
        generated_for_task += 1

    print(f"{task_type:<24}: {generated_for_task:>6}")

assert len(generated_data) == TOTAL_SIZE

print(f"\nTotal generated: {len(generated_data):,}")

random.shuffle(generated_data)

train_data = generated_data[:TRAIN_SIZE]
val_data = generated_data[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
test_data = generated_data[TRAIN_SIZE + VAL_SIZE:]

assert len(train_data) == TRAIN_SIZE
assert len(val_data) == VAL_SIZE
assert len(test_data) == TEST_SIZE


def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


save_jsonl(train_data, TRAIN_PATH)
save_jsonl(val_data, VAL_PATH)
save_jsonl(test_data, TEST_PATH)


def get_stats(data):
    task_counts = defaultdict(int)
    position_counts = defaultdict(lambda: defaultdict(int))

    for rec in data:
        task_counts[rec["task"]] += 1
        if rec["task"].startswith("compare"):
            position_counts[rec["task"]][rec["answer_pos"]] += 1

    return task_counts, position_counts


train_tasks, train_positions = get_stats(train_data)
val_tasks, val_positions = get_stats(val_data)
test_tasks, test_positions = get_stats(test_data)

print("\n" + "=" * 80)
print("NEPALI SOFT DATASET CREATED")
print("=" * 80)
print(f"Train : {len(train_data):,} -> {TRAIN_PATH}")
print(f"Val   : {len(val_data):,} -> {VAL_PATH}")
print(f"Test  : {len(test_data):,} -> {TEST_PATH}")

print("\n--- TRAIN TASK DISTRIBUTION ---")
for task in TASK_RATIOS:
    count = train_tasks[task]
    print(f"{task:<24}: {count:>6} ({count / len(train_data) * 100:>5.1f}%)")

print("\n--- VALIDATION TASK DISTRIBUTION ---")
for task in TASK_RATIOS:
    count = val_tasks[task]
    print(f"{task:<24}: {count:>6} ({count / len(val_data) * 100:>5.1f}%)")

print("\n--- TEST TASK DISTRIBUTION ---")
for task in TASK_RATIOS:
    count = test_tasks[task]
    print(f"{task:<24}: {count:>6} ({count / len(test_data) * 100:>5.1f}%)")

print("\n--- TRAIN ANSWER POSITION BALANCE ---")
for task in ["compare_2way", "compare_3way", "compare_4way"]:
    print(f"\n{task}:")
    for pos in sorted(train_positions[task]):
        print(f"  pos{pos}: {train_positions[task][pos]}")

print("\n--- TOP ANSWER NAMES ---")
for name, count in sorted(name_tracker.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"{name:<10}: {count}")

print("\n" + "=" * 80)
print("10 SAMPLE EXAMPLES")
print("=" * 80)
for i, example in enumerate(random.sample(generated_data, min(10, len(generated_data))), 1):
    print(f"\nExample {i}")
    print("-" * 60)
    print(example["prompt"])
    print(f"GOLD: {example['target']}")
    print(f"TASK: {example['task']}")
    print(f"ANSWER POSITION: {example['answer_pos']}")

print("\n" + "=" * 80)
print("SANITY CHECKS")
print("=" * 80)

assert len(seen_prompts) == len(generated_data)
print("[OK] No duplicate prompts")

assert len(train_data) == 30000
assert len(val_data) == 3000
assert len(test_data) == 3000
print("[OK] Train/Val/Test sizes correct")

for rec in generated_data:
    if rec["task"] == "compare_2way":
        assert rec["answer_pos"] in [0, 1]
    elif rec["task"] == "compare_3way":
        assert rec["answer_pos"] in [0, 1, 2]
    elif rec["task"] == "compare_4way":
        assert rec["answer_pos"] in [0, 1, 2, 3]

print("[OK] All comparison answer positions valid")
print("\nNEPALI SOFT generation COMPLETE.")
