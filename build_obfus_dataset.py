"""Build a unified obfus dataset with train/attack_val/test splits.

Source: /home/davidsc2/FOCAL/ctlm/pulled/obfuscated-activations/
        inference_time_experiments/datasets/harmful_dataset/

  {benign,harmful}_{train,val,test}_no_spec_tokens.csv

Output: a HuggingFace DatasetDict with columns ['text', 'label']:
  train       = harmful_train  + benign_train   (800 + 800 = 1600, label=1/0)
  attack_val  = harmful_val                     (100, all label=1)
  test        = harmful_test   + benign_test    (100 + 100 = 200)

Labels are binary (1 = harmful, 0 = benign). Unlike aegis, obfus does NOT
require per-model carving — the source already ships a val split, so all 4
classifiers share the same obfus dataset.

The 'text' field comes from the CSV 'prompt' column (not 'response'), matching
how aegis trained on prompts only.

Usage:
    python harm_classifiers/build_obfus_dataset.py \\
        --output harm_classifiers/datasets/obfus_v2
"""

import argparse
import os

import pandas as pd
from datasets import Dataset, DatasetDict


SRC = ("/home/davidsc2/FOCAL/ctlm/pulled/obfuscated-activations/"
       "inference_time_experiments/datasets/harmful_dataset")


def load_split(label_name: str, split: str) -> pd.DataFrame:
    path = os.path.join(SRC, f"{label_name}_{split}_no_spec_tokens.csv")
    df = pd.read_csv(path)
    df = df[["prompt"]].rename(columns={"prompt": "text"})
    df["label"] = 1 if label_name == "harmful" else 0
    return df


def build():
    train = pd.concat([load_split("harmful", "train"),
                       load_split("benign", "train")], ignore_index=True)
    attack_val = load_split("harmful", "val")  # harmful-only, like aegis
    test = pd.concat([load_split("harmful", "test"),
                      load_split("benign", "test")], ignore_index=True)

    train = train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return DatasetDict({
        "train": Dataset.from_pandas(train, preserve_index=False),
        "attack_val": Dataset.from_pandas(attack_val, preserve_index=False),
        "test": Dataset.from_pandas(test, preserve_index=False),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, required=True,
                    help="Output dir for the DatasetDict")
    args = ap.parse_args()

    ds = build()
    print(ds)
    for split in ds:
        labels = ds[split]["label"]
        print(f"  {split}: n={len(ds[split])}, "
              f"harmful={sum(labels)}, benign={len(labels) - sum(labels)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ds.save_to_disk(args.output)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
