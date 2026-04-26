"""Build an aegis_v2 dataset from nvidia/Aegis-AI-Content-Safety-Dataset-2.0.

Label space (4-class):
    0 = (safe,   safe)    — benign prompt, safe response
    1 = (safe,   unsafe)  — benign prompt, harmful response
    2 = (unsafe, safe)    — harmful prompt, safe/refused response
    3 = (unsafe, unsafe)  — harmful prompt, harmful response

Rows where response_label is None (prompt-only annotations) are dropped.
Use the dataset's native train/validation/test splits directly — no carving.

Text format: "User: {prompt}\n\nAssistant: {response}"
(same as obfus; adversarially controllable region is before the boundary)

Usage:
    python harm_classifiers/build_aegis_v2_dataset.py \\
        --output harm_classifiers/datasets/aegis_v2
"""

import argparse
import os

from datasets import Dataset, DatasetDict, load_dataset


LABEL_MAP = {
    ("safe",   "safe"):   0,
    ("safe",   "unsafe"): 1,
    ("unsafe", "safe"):   2,
    ("unsafe", "unsafe"): 3,
}

LABEL_NAMES = ["safe_safe", "safe_unsafe", "unsafe_safe", "unsafe_unsafe"]


def process_split(split):
    rows = {"text": [], "label": []}
    for ex in split:
        if ex["response_label"] is None:
            continue
        key = (ex["prompt_label"], ex["response_label"])
        if key not in LABEL_MAP:
            continue
        text = f"User: {ex['prompt']}\n\nAssistant: {ex['response']}"
        rows["text"].append(text)
        rows["label"].append(LABEL_MAP[key])
    return Dataset.from_dict(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    print("Downloading nvidia/Aegis-AI-Content-Safety-Dataset-2.0 ...")
    raw = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")

    ds = DatasetDict({
        "train":      process_split(raw["train"]),
        "validation": process_split(raw["validation"]),
        "test":       process_split(raw["test"]),
    })

    print(ds)
    for split_name, split in ds.items():
        from collections import Counter
        counts = Counter(split["label"])
        print(f"  {split_name}: n={len(split)}")
        for label_id, name in enumerate(LABEL_NAMES):
            print(f"    {name}: {counts.get(label_id, 0)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ds.save_to_disk(args.output)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
