"""
Preprocess the obfuscated-activations harmful dataset into HuggingFace Dataset format,
matching the AEGIS schema (columns: text, label) for use with train_harm_classifier.py.

Source: /home/davidsc2/FOCAL/ctlm/pulled/obfuscated-activations/inference_time_experiments/datasets/harmful_dataset/
Output: HuggingFace DatasetDict with train/test splits, each row having 'text' and 'label'.
  label 0 = safe, label 1 = harmful
"""

import pandas as pd
from datasets import Dataset, DatasetDict

SRC_DIR = "/home/davidsc2/FOCAL/ctlm/pulled/obfuscated-activations/inference_time_experiments/datasets/harmful_dataset"
SAVE_PATH = "/share/CT_LM_models/sbert/obfus_harm_preprocessed"


def load_split(split_name):
    harmful = pd.read_csv(f"{SRC_DIR}/harmful_{split_name}_no_spec_tokens.csv")
    benign = pd.read_csv(f"{SRC_DIR}/benign_{split_name}_no_spec_tokens.csv")

    texts = []
    labels = []

    for _, row in benign.iterrows():
        texts.append(f"{row['prompt']}\n{row['response']}")
        labels.append(0)

    for _, row in harmful.iterrows():
        texts.append(f"{row['prompt']}\n{row['response']}")
        labels.append(1)

    return Dataset.from_dict({"text": texts, "label": labels})


# Merge val into test since val is small (100+100) and train_harm_classifier
# evaluates on the "test" split.
train_ds = load_split("train")
val_ds = load_split("val")
test_ds = load_split("test")

# Concatenate val and test into a single test split
from datasets import concatenate_datasets
test_combined = concatenate_datasets([val_ds, test_ds])

dataset = DatasetDict({"train": train_ds, "test": test_combined})

print(dataset)
print(f"\nTrain label distribution: {pd.Series(dataset['train']['label']).value_counts().to_dict()}")
print(f"Test label distribution:  {pd.Series(dataset['test']['label']).value_counts().to_dict()}")
print(f"\nSample (safe):    {dataset['train'][0]['text'][:120]}...")
print(f"Sample (harmful): {dataset['train'][800]['text'][:120]}...")

dataset.save_to_disk(SAVE_PATH)
print(f"\nSaved to {SAVE_PATH}")
