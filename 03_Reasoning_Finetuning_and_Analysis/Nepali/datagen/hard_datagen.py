import json
import random
import os
import sys
import hashlib
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# HARD DATASET GENERATOR (NEPALI)
# =============================================================================

random.seed(42)

OUT_DIR = os.getenv("OUT_DIR")
if not OUT_DIR:
    if os.name != "nt" and os.path.exists("/kaggle/working"):
        OUT_DIR = "/kaggle/working"
    else:
        OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(OUT_DIR, "hard_train.jsonl")
VAL_PATH   = os.path.join(OUT_DIR, "hard_val.jsonl")
TEST_PATH  = os.path.join(OUT_DIR, "hard_test.jsonl")

TRAIN_SIZE = 30000
VAL_SIZE   = 3000
TEST_SIZE  = 3000

TOTAL_SIZE = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

# =============================================================================
# NAMES (NEPALI)
# =============================================================================

NAMES = [
    "अमित", "राहुल", "विकास", "सुरेश", "मोहन",
    "रोहित", "दीपक", "संजय", "कविता", "अशोक",
    "पूजा", "सुनीता", "रमेश", "राजू", "नीलम"
]

# =============================================================================
# ATTRIBUTES (NEPALI)
# =============================================================================

ATTRIBUTES = [
    {
        "name": "अंक",
        "unit": "",
        "min_words": [
            "सबैभन्दा कम अंक",
            "सबैभन्दा कम",
            "न्यूनतम अंक",
            "सबैभन्दा कम अंक कसको छ?"
        ],
        "max_words": [
            "सबैभन्दा बढी अंक",
            "सबैभन्दा बढी",
            "अधिकतम अंक",
            "सबैभन्दा बढी अंक कसको छ?"
        ]
    },
    {
        "name": "मूल्य",
        "unit": " रुपैयाँ",
        "min_words": [
            "सबैभन्दा सस्तो",
            "सबैभन्दा कम मूल्य",
            "न्यूनतम मूल्य",
            "सबैभन्दा कम मूल्य कसको छ?"
        ],
        "max_words": [
            "सबैभन्दा महँगो",
            "सबैभन्दा बढी मूल्य",
            "अधिकतम मूल्य",
            "सबैभन्दा बढी मूल्य कसको छ?"
        ]
    },
    {
        "name": "तौल",
        "unit": " किलोग्राम",
        "min_words": [
            "सबैभन्दा हलुका",
            "सबैभन्दा कम तौल",
            "न्यूनतम तौल",
            "सबैभन्दा कम तौल कसको छ?"
        ],
        "max_words": [
            "सबैभन्दा गह्रौँ",
            "सबैभन्दा बढी तौल",
            "अधिकतम तौल",
            "सबैभन्दा बढी तौल कसको छ?"
        ]
    },
    {
        "name": "उमेर",
        "unit": " वर्ष",
        "min_words": [
            "सबैभन्दा कान्छा",
            "सबैभन्दा कम उमेर",
            "न्यूनतम उमेर",
            "सबैभन्दा कम उमेर कसको छ?"
        ],
        "max_words": [
            "सबैभन्दा जेठा",
            "सबैभन्दा बढी उमेर",
            "अधिकतम उमेर",
            "सबैभन्दा बढी उमेर कसको छ?"
        ]
    }
]

# =============================================================================
# HARD TASK DISTRIBUTION
# =============================================================================

TASK_RATIOS = {
    "compare_2way": 0.15,
    "compare_3way": 0.35,
    "compare_4way": 0.40,
    "bind_entity_to_value": 0.05,
    "bind_value_to_entity": 0.05
}

quotas = {
    task: int(TOTAL_SIZE * ratio)
    for task, ratio in TASK_RATIOS.items()
}

difference = TOTAL_SIZE - sum(quotas.values())
quotas["compare_4way"] += difference

# =============================================================================
# TRACKERS
# =============================================================================

generated_data = []
seen_prompts = set()
position_tracker = defaultdict(lambda: defaultdict(int))


def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def generate_close_values(n):
    start = random.randint(10, 90)
    values = set()
    while len(values) < n:
        offset = random.randint(0, min(12, 99 - start))
        values.add(start + offset)
    return list(values)


DEVANAGARI_TRANS = str.maketrans("0123456789", "०१२३४५६७८९")

def to_nepali_num(val):
    return str(val).translate(DEVANAGARI_TRANS)


# =============================================================================
# FORMAT VARIATION (NEPALI)
# =============================================================================

def format_entity_line(name, value, attr, style):
    unit = attr["unit"]
    attribute = attr["name"]
    val_str = to_nepali_num(value)

    if style == 0:
        return f"{name}: {val_str}{unit}"
    elif style == 1:
        return f"{name}को {attribute} {val_str}{unit} छ।"
    elif style == 2:
        return f"{name}को {attribute} {val_str}{unit} छ।"
    elif style == 3:
        return f"{name}को {attribute} = {val_str}{unit}"
    else:
        return f"{name}सँग {attribute} {val_str}{unit} छ।"


# =============================================================================
# GENERATE COMPARISON
# =============================================================================

def generate_comparison(task_type):
    num_entities = int(task_type.split("_")[1][0])

    names = random.sample(NAMES, num_entities)
    attr = random.choice(ATTRIBUTES)
    values = generate_close_values(num_entities)

    combined = list(zip(names, values))
    random.shuffle(combined)

    names = [x[0] for x in combined]
    values = [x[1] for x in combined]

    is_max = random.choice([True, False])
    target_value = max(values) if is_max else min(values)
    target_idx = values.index(target_value)
    target_name = names[target_idx]

    original_quota = int(TOTAL_SIZE * TASK_RATIOS[task_type])
    max_per_position = max(1, original_quota // num_entities)

    if position_tracker[task_type][target_idx] >= max_per_position:
        return None

    style = random.randint(0, 4)

    lines = [
        format_entity_line(n, v, attr, style)
        for n, v in zip(names, values)
    ]

    if is_max:
        q = random.choice(attr["max_words"])
    else:
        q = random.choice(attr["min_words"])

    if q in ["सबैभन्दा बढी", "सबैभन्दा कम", "न्यूनतम अंक", "अधिकतम अंक"]:
        q = q + " " + attr["name"] + " कसको छ?"

    prompt = "\n".join(lines) + "\n\n" + q + "\nउत्तर:"
    target = " " + target_name

    return {
        "prompt": prompt,
        "target": target,
        "task": task_type,
        "answer_pos": target_idx
    }


# =============================================================================
# BINDING TASKS (NEPALI)
# =============================================================================

def generate_binding(task_type):
    attr = random.choice(ATTRIBUTES)

    if task_type == "bind_entity_to_value":
        name = random.choice(NAMES)
        value = random.randint(10, 99)
        val_str = to_nepali_num(value)

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
                f"{name}को {attr['name']} = {val_str}{attr['unit']}\n"
                f"{name}को {attr['name']} कति छ?\n"
                f"उत्तर:"
            )
        ]

        prompt = random.choice(templates)

        return {
            "prompt": prompt,
            "target": f" {val_str}",
            "task": task_type,
            "answer_pos": 0
        }
    else:
        names = random.sample(NAMES, 3)
        values = random.sample(range(10, 99), 3)
        target_idx = random.randint(0, 2)

        lines = []
        for name, value in zip(names, values):
            style = random.randint(0, 3)
            lines.append(format_entity_line(name, value, attr, style))

        target_name = names[target_idx]
        target_value = values[target_idx]
        target_value_str = to_nepali_num(target_value)

        question_templates = [
            f"{target_value_str}{attr['unit']} कसको छ?",
            f"{target_value_str}{attr['unit']} कसको हो?",
            f"{target_value_str}{attr['unit']} {attr['name']} कसको छ?"
        ]

        prompt = (
            "\n".join(lines)
            + "\n\n"
            + random.choice(question_templates)
            + "\nउत्तर:"
        )

        return {
            "prompt": prompt,
            "target": f" {target_name}",
            "task": task_type,
            "answer_pos": target_idx
        }


# =============================================================================
# GENERATE
# =============================================================================

print("=" * 80)
print("GENERATING NEPALI V7 HARD DATASET")
print("=" * 80)

while sum(quotas.values()) > 0:
    available = [k for k, v in quotas.items() if v > 0]
    if not available:
        break

    weights = [quotas[k] for k in available]
    task = random.choices(available, weights=weights, k=1)[0]

    if task.startswith("compare"):
        example = generate_comparison(task)
    else:
        example = generate_binding(task)

    if example is None:
        continue

    h = get_hash(example["prompt"])
    if h in seen_prompts:
        continue

    seen_prompts.add(h)
    quotas[task] -= 1

    if task.startswith("compare"):
        position_tracker[task][example["answer_pos"]] += 1

    generated_data.append(example)

    if len(generated_data) % 5000 == 0:
        print(f"Generated {len(generated_data):,}/{TOTAL_SIZE:,}")

random.shuffle(generated_data)

train_data = generated_data[:TRAIN_SIZE]
val_data = generated_data[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
test_data = generated_data[TRAIN_SIZE + VAL_SIZE:]


def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


save_jsonl(train_data, TRAIN_PATH)
save_jsonl(val_data, VAL_PATH)
save_jsonl(test_data, TEST_PATH)


def prompt_set(data):
    return set(x["prompt"] for x in data)


train_prompts = prompt_set(train_data)
val_prompts = prompt_set(val_data)
test_prompts = prompt_set(test_data)

print()
print("=" * 80)
print("NEPALI HARD DATASET CREATED")
print("=" * 80)
print(f"Train: {len(train_data):,}")
print(f"Val  : {len(val_data):,}")
print(f"Test : {len(test_data):,}")
print()
print("Train AND Val :", len(train_prompts & val_prompts))
print("Train AND Test:", len(train_prompts & test_prompts))
print("Val AND Test  :", len(val_prompts & test_prompts))

counts = defaultdict(int)
for item in generated_data:
    counts[item["task"]] += 1

print()
print("TASK DISTRIBUTION")
print("-" * 50)
for task in sorted(counts):
    pct = (counts[task] / len(generated_data) * 100)
    print(f"{task:<28} {counts[task]:>6} ({pct:5.1f}%)")

print()
print("ANSWER POSITION DISTRIBUTION")
print("-" * 50)
for task in ["compare_2way", "compare_3way", "compare_4way"]:
    print(f"\n{task}")
    total = sum(position_tracker[task].values())
    for pos in sorted(position_tracker[task]):
        count = position_tracker[task][pos]
        print(f"  pos{pos}: {count}")

print()
print("=" * 80)
print("SAMPLE HARD EXAMPLES")
print("=" * 80)

comparison_examples = [
    x for x in generated_data
    if x["task"] in ["compare_3way", "compare_4way"]
]

for item in comparison_examples[:10]:
    print()
    print(f"Task: {item['task']}")
    print(item["prompt"])
    print("Gold:", item["target"])

print()
print("=" * 80)
print("FILES")
print("=" * 80)
print(TRAIN_PATH)
print(VAL_PATH)
print(TEST_PATH)

print()
print("[OK] NEPALI HARD DATASET READY")
