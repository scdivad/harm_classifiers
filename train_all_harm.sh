#!/bin/bash
# Train BERT, RoBERTa, and DeBERTa on all 3 harmfulness datasets.
# Usage: bash train_all_harm.sh

set -e

SCRIPT="train_harm_classifier.py"
AEGIS="datasets/aegis"
OBFUS="datasets/obfus"
HHRLHF="datasets/hhrlhf"

MODELS=(
    "bert-base-uncased"
    "roberta-base"
    "microsoft/deberta-v3-base"
)
SHORT_NAMES=(
    "bert"
    "roberta"
    "deberta"
)

DATASETS=("$AEGIS" "$OBFUS" "$HHRLHF")
DATASET_NAMES=("aegis" "obfus" "hhrlhf")

for i in "${!MODELS[@]}"; do
    for j in "${!DATASETS[@]}"; do
        echo "===== Training ${SHORT_NAMES[$i]} on ${DATASET_NAMES[$j]} ====="
        python "$SCRIPT" \
            --model "${MODELS[$i]}" \
            --dataset "${DATASETS[$j]}" \
            --save_dir "models/${SHORT_NAMES[$i]}_${DATASET_NAMES[$j]}_binary"
    done
done

echo "===== All training complete ====="
