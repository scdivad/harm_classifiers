"""Build an aegis_v2_prompt dataset from nvidia/Aegis-AI-Content-Safety-Dataset-2.0.

Label space (binary):
    0 = safe prompt
    1 = unsafe prompt

Uses all rows (including prompt-only rows where response_label is None).
Uses the dataset's native train/validation/test splits directly.

Text format: "User: {prompt}"
(prompt only — no assistant section)

Usage:
    python harm_classifiers/build_aegis_v2_prompt_dataset.py \\
        --output harm_classifiers/datasets/aegis_v2_prompt
"""

import argparse
import os

from datasets import Dataset, DatasetDict, load_dataset


LABEL_MAP = {"safe": 0, "unsafe": 1}
LABEL_NAMES = ["safe", "unsafe"]


def process_split(split):
    rows = {"text": [], "label": []}
    for ex in split:
        label = LABEL_MAP.get(ex["prompt_label"])
        if label is None:
            continue
        rows["text"].append(f"User: {ex['prompt']}")
        rows["label"].append(label)
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
        print(f"  {split_name}: n={len(split)}, "
              f"safe={counts.get(0, 0)}, unsafe={counts.get(1, 0)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ds.save_to_disk(args.output)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
