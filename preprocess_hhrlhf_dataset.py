"""
Preprocess the Anthropic HH-RLHF harmless-base dataset into HuggingFace Dataset format,
matching the AEGIS schema (columns: text, label) for use with train_harm_classifier.py.

Source: /home/davidsc2/FOCAL/ctlm/pulled/hh-rlhf/harmless-base/
Each record has a 'chosen' (safe) and 'rejected' (harmful) conversation.
We extract both as separate samples:
  label 0 = safe (chosen response)
  label 1 = harmful (rejected response)
"""

import json
import gzip
import random
import pandas as pd
from datasets import Dataset, DatasetDict

SRC_DIR = "/home/davidsc2/FOCAL/ctlm/pulled/hh-rlhf/harmless-base"
SAVE_PATH = "/share/CT_LM_models/sbert/hhrlhf_harm_preprocessed"

MAX_TRAIN_PROMPTS = 25000
SEED = 42


def load_jsonl_gz(path):
    records = []
    with gzip.open(path, "rt") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def records_to_dataset(records):
    texts = []
    labels = []
    for rec in records:
        # chosen = safe response (label 0)
        texts.append(rec["chosen"].strip())
        labels.append(0)
        # rejected = harmful response (label 1)
        texts.append(rec["rejected"].strip())
        labels.append(1)
    return Dataset.from_dict({"text": texts, "label": labels})


train_records = load_jsonl_gz(f"{SRC_DIR}/train.jsonl.gz")
test_records = load_jsonl_gz(f"{SRC_DIR}/test.jsonl.gz")

# Downsample training prompts
if len(train_records) > MAX_TRAIN_PROMPTS:
    random.seed(SEED)
    train_records = random.sample(train_records, MAX_TRAIN_PROMPTS)
    print(f"Downsampled train to {MAX_TRAIN_PROMPTS} prompts ({MAX_TRAIN_PROMPTS * 2} samples)")

train_ds = records_to_dataset(train_records)
test_ds = records_to_dataset(test_records)

dataset = DatasetDict({"train": train_ds, "test": test_ds})

print(dataset)
print(f"\nTrain label distribution: {pd.Series(dataset['train']['label']).value_counts().to_dict()}")
print(f"Test label distribution:  {pd.Series(dataset['test']['label']).value_counts().to_dict()}")
print(f"\nSample (safe):    {dataset['train'][0]['text'][:150]}...")
print(f"Sample (harmful): {dataset['train'][1]['text'][:150]}...")

dataset.save_to_disk(SAVE_PATH)
print(f"\nSaved to {SAVE_PATH}")
