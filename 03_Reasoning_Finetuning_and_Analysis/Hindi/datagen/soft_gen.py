# ==============================================================================
# SOFT DATASET GENERATION
# Fixes:
#   1. No rejection/infinite while-loop
#   2. Explicit answer-position balancing
#   3. Balanced names
#   4. More numerical entity-binding examples
#   5. Equality examples
#   6. Exact train/val/test sizes
#   7. No duplicate prompts
# ==============================================================================


import json
import random
import os
import hashlib
from collections import defaultdict

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

# Focus mainly on the problem we actually have:
# numerical entity <-> value binding.
#
# Transitive/symbolic reasoning is already relatively easy for the model,
# so we do not waste most of the dataset on it.

TASK_RATIOS = {
    "bind_entity_to_value": 0.20,
    "bind_value_to_entity": 0.15,
    "compare_2way":         0.25,
    "compare_3way":         0.20,
    "compare_4way":         0.10,
    "equality":             0.10,
}


# ==============================================================================
# VOCABULARY
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
        "min_word": "सबसे कम",
        "max_word": "सबसे अधिक",
    },
    {
        "name": "वजन",
        "unit": " किलोग्राम",
        "min_word": "सबसे हल्का",
        "max_word": "सबसे भारी",
    },
    {
        "name": "उम्र",
        "unit": " वर्ष",
        "min_word": "सबसे छोटा",
        "max_word": "सबसे बड़ा",
    },
    {
        "name": "मूल्य",
        "unit": " रुपये",
        "min_word": "सबसे सस्ता",
        "max_word": "सबसे महंगा",
    },
]


# ==============================================================================
# EXACT TASK COUNTS
# ==============================================================================

def make_exact_quotas(total_size, ratios):
    """
    Convert ratios into exact integer counts whose sum is total_size.
    """
    raw = {
        k: total_size * v
        for k, v in ratios.items()
    }

    quotas = {
        k: int(v)
        for k, v in raw.items()
    }

    remainder = total_size - sum(quotas.values())

    # Distribute rounding remainder deterministically
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

# Track how often each name is used as answer.
name_tracker = defaultdict(int)

# Track answer positions for comparison tasks.
position_tracker = defaultdict(lambda: defaultdict(int))


# ==============================================================================
# HELPERS
# ==============================================================================

def get_hash(text):
    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


def choose_attribute():
    return random.choice(ATTRIBUTES)


def choose_names(n):
    """
    Random unique names.
    """
    return random.sample(NAMES, n)


def choose_balanced_position(task_type, n):
    """
    Always choose the currently least-used answer position.

    This is the important fix.

    OLD:
        generate random values
        discover target position
        reject if position is full

    NEW:
        choose target position first
        construct values so that target position wins
    """

    counts = [
        position_tracker[task_type][i]
        for i in range(n)
    ]

    minimum = min(counts)

    candidates = [
        i for i, c in enumerate(counts)
        if c == minimum
    ]

    return random.choice(candidates)


def generate_distinct_values(n):
    """
    Generate n distinct values.
    """
    return random.sample(range(10, 100), n)


def construct_values_with_winner(n, target_idx, is_max):
    """
    Construct values where target_idx is guaranteed to be:
        maximum, if is_max=True
        minimum, if is_max=False
    """

    values = [None] * n

    if is_max:

        # Winner is deliberately large.
        winner = random.randint(70, 99)
        values[target_idx] = winner

        for i in range(n):
            if i == target_idx:
                continue

            values[i] = random.randint(
                10,
                winner - 1
            )

    else:

        # Winner is deliberately small.
        winner = random.randint(10, 30)
        values[target_idx] = winner

        for i in range(n):
            if i == target_idx:
                continue

            values[i] = random.randint(
                winner + 1,
                99
            )

    return values


# ==============================================================================
# TEMPLATE GENERATORS
# ==============================================================================

def make_bind_entity_to_value():
    """
    Example:

    अमित के अंक 72 हैं।
    अमित के अंक कितने हैं?
    उत्तर:

    Gold = 72
    """

    attr = choose_attribute()
    name = random.choice(NAMES)

    val = random.randint(10, 99)

    templates = [

        (
            f"{name} के {attr['name']} {val}{attr['unit']} हैं।\n"
            f"{name} के {attr['name']} कितने हैं?\n"
            f"उत्तर:"
        ),

        (
            f"{name} का {attr['name']} {val}{attr['unit']} है।\n"
            f"बताइए {name} के {attr['name']} कितने हैं?\n"
            f"उत्तर:"
        ),

        (
            f"{name} के पास {attr['name']} {val}{attr['unit']} हैं।\n"
            f"{name} का मान कितना है?\n"
            f"उत्तर:"
        ),
    ]

    prompt = random.choice(templates)

    return {
        "prompt": prompt,
        "target": f" {val}",
        "task": "bind_entity_to_value",
        "answer_pos": 0,
    }


def make_bind_value_to_entity():
    """
    Example:

    अमित के अंक 35 हैं और राहुल के अंक 82 हैं।
    82 अंक किसके हैं?
    उत्तर:

    Gold = राहुल
    """

    attr = choose_attribute()

    names = choose_names(2)

    v1, v2 = random.sample(
        range(10, 100),
        2
    )

    templates = [

        (
            f"{names[0]} के {attr['name']} {v1}{attr['unit']} हैं "
            f"और {names[1]} के {attr['name']} {v2}{attr['unit']} हैं।\n"
            f"{v2}{attr['unit']} {attr['name']} किसका है?\n"
            f"उत्तर:"
        ),

        (
            f"{names[0]} का {attr['name']} {v1}{attr['unit']} है, "
            f"जबकि {names[1]} का {attr['name']} {v2}{attr['unit']} है।\n"
            f"{v2}{attr['unit']} किसका है?\n"
            f"उत्तर:"
        ),

        (
            f"{names[0]}: {v1}{attr['unit']}\n"
            f"{names[1]}: {v2}{attr['unit']}\n\n"
            f"{v2}{attr['unit']} किसके हैं?\n"
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
    """
    Comparison task with GUARANTEED answer position.

    This is the key V5 fix.
    """

    n = int(task_type.split("_")[1][0])

    attr = choose_attribute()

    names = choose_names(n)

    # First decide where the answer should be.
    target_idx = choose_balanced_position(
        task_type,
        n
    )

    # Decide max or min.
    is_max = random.choice([True, False])

    # Construct numbers so target_idx is guaranteed correct.
    values = construct_values_with_winner(
        n=n,
        target_idx=target_idx,
        is_max=is_max
    )

    target_name = names[target_idx]

    # Track position (soft dataset fix for balancing)
    position_tracker[task_type][target_idx] += 1

    # Track answer name.
    name_tracker[target_name] += 1

    lines = []

    # Several surface forms to prevent memorization.
    template_style = random.randint(0, 2)

    if template_style == 0:

        for name, value in zip(names, values):
            lines.append(
                f"{name}: {value}{attr['unit']}"
            )

        if attr["name"] == "अंक":
            if is_max:
                question = "सबसे अधिक अंक किसके हैं?"
            else:
                question = "सबसे कम अंक किसके हैं?"
        else:
            if is_max:
                question = f"कौन {attr['max_word']} है?"
            else:
                question = f"कौन {attr['min_word']} है?"

        prompt = (
            "\n".join(lines)
            + f"\n\n{question}\nउत्तर:"
        )

    elif template_style == 1:

        for name, value in zip(names, values):
            lines.append(
                f"{name} के {attr['name']} "
                f"{value}{attr['unit']} हैं।"
            )

        if is_max:
            question = (
                f"इनमें से {attr['max_word']} "
                f"किसके हैं?"
            )
        else:
            question = (
                f"इनमें से {attr['min_word']} "
                f"किसके हैं?"
            )

        prompt = (
            " ".join(lines)
            + f"\n{question}\nउत्तर:"
        )

    else:

        for name, value in zip(names, values):
            lines.append(
                f"{name} का {attr['name']} "
                f"{value}{attr['unit']} है।"
            )

        if is_max:
            question = f"सबसे अधिक {attr['name']} किसका है?"
        else:
            question = f"सबसे कम {attr['name']} किसका है?"

        prompt = (
            " ".join(lines)
            + f"\n{question}\nउत्तर:"
        )

    return {
        "prompt": prompt,
        "target": f" {target_name}",
        "task": task_type,
        "answer_pos": target_idx,
    }


def make_equality():
    """
    Equality is explicitly represented.

    Example:

    राहुल: 50
    अमित: 50

    Gold:
    दोनों समान हैं
    """

    attr = choose_attribute()

    names = choose_names(2)

    value = random.randint(10, 99)

    templates = [

        (
            f"{names[0]}: {value}{attr['unit']}\n"
            f"{names[1]}: {value}{attr['unit']}\n\n"
            f"किसके अधिक {attr['name']} हैं?\n"
            f"उत्तर:"
        ),

        (
            f"{names[0]} के {attr['name']} "
            f"{value}{attr['unit']} हैं और "
            f"{names[1]} के {attr['name']} भी "
            f"{value}{attr['unit']} हैं।\n"
            f"किसका {attr['name']} अधिक है?\n"
            f"उत्तर:"
        ),

        (
            f"{names[0]} का {attr['name']} "
            f"{value}{attr['unit']} है। "
            f"{names[1]} का {attr['name']} भी "
            f"{value}{attr['unit']} है।\n"
            f"कौन अधिक है?\n"
            f"उत्तर:"
        ),
    ]

    prompt = random.choice(templates)

    return {
        "prompt": prompt,
        "target": " दोनों समान हैं",
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

print("\nGenerating SOFT dataset...")
print("No rejection-based generation is used.\n")


# ==============================================================================
# GENERATE EXACT NUMBER FOR EACH TASK
# ==============================================================================

for task_type, count in quotas.items():

    generated_for_task = 0

    attempts = 0

    while generated_for_task < count:

        attempts += 1

        # Safety check.
        if attempts > count * 20:
            raise RuntimeError(
                f"Too many duplicate prompts for {task_type}. "
                f"Generated {generated_for_task}/{count}."
            )

        example = GENERATORS[task_type]()

        prompt = example["prompt"]

        prompt_hash = get_hash(prompt)

        # Avoid duplicate prompts.
        if prompt_hash in seen_prompts:
            continue

        seen_prompts.add(prompt_hash)

        generated_data.append(example)

        generated_for_task += 1

    print(
        f"{task_type:<24}: "
        f"{generated_for_task:>6}"
    )


# ==============================================================================
# VERIFY TOTAL
# ==============================================================================

assert len(generated_data) == TOTAL_SIZE

print(
    f"\nTotal generated: "
    f"{len(generated_data):,}"
)


# ==============================================================================
# SHUFFLE BEFORE SPLIT
# ==============================================================================

random.shuffle(generated_data)


# ==============================================================================
# TRAIN / VAL / TEST SPLIT
# ==============================================================================

train_data = generated_data[
    :TRAIN_SIZE
]

val_data = generated_data[
    TRAIN_SIZE:
    TRAIN_SIZE + VAL_SIZE
]

test_data = generated_data[
    TRAIN_SIZE + VAL_SIZE:
]


assert len(train_data) == TRAIN_SIZE
assert len(val_data) == VAL_SIZE
assert len(test_data) == TEST_SIZE


# ==============================================================================
# SAVE JSONL
# ==============================================================================

def save_jsonl(data, filename):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        for record in data:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


save_jsonl(
    train_data,
    TRAIN_PATH
)

save_jsonl(
    val_data,
    VAL_PATH
)

save_jsonl(
    test_data,
    TEST_PATH
)


# ==============================================================================
# STATISTICS
# ==============================================================================

def get_stats(data):

    task_counts = defaultdict(int)

    position_counts = defaultdict(lambda: defaultdict(int))

    for rec in data:

        task_counts[
            rec["task"]
        ] += 1

        if rec["task"].startswith("compare"):

            position_counts[
                rec["task"]
            ][rec["answer_pos"]] += 1

    return task_counts, position_counts


train_tasks, train_positions = get_stats(train_data)
val_tasks, val_positions = get_stats(val_data)
test_tasks, test_positions = get_stats(test_data)


# ==============================================================================
# PRINT RESULTS
# ==============================================================================

print("\n" + "=" * 80)
print("SOFT DATASET CREATED")
print("=" * 80)

print(
    f"Train : {len(train_data):,} -> {TRAIN_PATH}"
)

print(
    f"Val   : {len(val_data):,} -> {VAL_PATH}"
)

print(
    f"Test  : {len(test_data):,} -> {TEST_PATH}"
)


print("\n--- TRAIN TASK DISTRIBUTION ---")

for task in TASK_RATIOS:

    count = train_tasks[task]

    print(
        f"{task:<24}: "
        f"{count:>6} "
        f"({count / len(train_data) * 100:>5.1f}%)"
    )


print("\n--- VALIDATION TASK DISTRIBUTION ---")

for task in TASK_RATIOS:

    count = val_tasks[task]

    print(
        f"{task:<24}: "
        f"{count:>6} "
        f"({count / len(val_data) * 100:>5.1f}%)"
    )


print("\n--- TEST TASK DISTRIBUTION ---")

for task in TASK_RATIOS:

    count = test_tasks[task]

    print(
        f"{task:<24}: "
        f"{count:>6} "
        f"({count / len(test_data) * 100:>5.1f}%)"
    )


# ==============================================================================
# POSITION BALANCE
# ==============================================================================

print("\n--- TRAIN ANSWER POSITION BALANCE ---")

for task in [
    "compare_2way",
    "compare_3way",
    "compare_4way",
]:

    print(f"\n{task}:")

    for pos in sorted(train_positions[task]):

        print(
            f"  pos{pos}: "
            f"{train_positions[task][pos]}"
        )


# ==============================================================================
# NAME BALANCE
# ==============================================================================

print("\n--- TOP ANSWER NAMES ---")

for name, count in sorted(
    name_tracker.items(),
    key=lambda x: x[1],
    reverse=True
)[:15]:

    print(
        f"{name:<10}: {count}"
    )


# ==============================================================================
# SAMPLE EXAMPLES
# ==============================================================================

print("\n" + "=" * 80)
print("10 SAMPLE EXAMPLES")
print("=" * 80)

for i, example in enumerate(
    random.sample(
        generated_data,
        min(10, len(generated_data))
    ),
    1
):

    print(f"\nExample {i}")
    print("-" * 60)

    print(
        example["prompt"]
    )

    print(
        f"GOLD: {example['target']}"
    )

    print(
        f"TASK: {example['task']}"
    )

    print(
        f"ANSWER POSITION: {example['answer_pos']}"
    )


# ==============================================================================
# FINAL SANITY CHECKS
# ==============================================================================

print("\n" + "=" * 80)
print("SANITY CHECKS")
print("=" * 80)

# No duplicate prompts.
assert len(seen_prompts) == len(generated_data)

print("✓ No duplicate prompts")

# Exact sizes.
assert len(train_data) == 30000
assert len(val_data) == 3000
assert len(test_data) == 3000

print("✓ Train/Val/Test sizes correct")

# Check every comparison has valid answer position.
for rec in generated_data:

    if rec["task"] == "compare_2way":
        assert rec["answer_pos"] in [0, 1]

    elif rec["task"] == "compare_3way":
        assert rec["answer_pos"] in [0, 1, 2]

    elif rec["task"] == "compare_4way":
        assert rec["answer_pos"] in [0, 1, 2, 3]

print("✓ All comparison answer positions valid")

print("\nSOFT generation COMPLETE.")
print("You can now start SOFT training.")