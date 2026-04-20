#!/bin/bash
# Submit SLURM jobs to train SBERT, BERT, RoBERTa, and DeBERTa on the obfus
# dataset (binary harmful/benign classification).
#
# The obfus dataset has a shared attack_val split (harmful_val from the source)
# so all 4 models use the SAME dataset — no per-model carving needed.
#
# Outputs models to models/{short_name}_obfus_retrained/ to avoid clobbering
# existing {short_name}_obfus_binary checkpoints in the same tree.
#
# Usage:
#   bash harm_classifiers/train_all_obfus.sh                  # all 4
#   MODELS="sbert bert" bash harm_classifiers/train_all_obfus.sh  # subset
#
# PREREQ: run build_obfus_dataset.py first to create datasets/obfus_v2.

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${WORKDIR}/train_harm_classifier.py"
DATASET="${WORKDIR}/datasets/obfus_v2"

# Short-name → (HF model id, model_type) pairs.
declare -A HF_MODEL=(
    [sbert]="sentence-transformers/all-MiniLM-L6-v2"
    [bert]="bert-base-uncased"
    [roberta]="roberta-base"
    [deberta]="microsoft/deberta-v3-base"
)
declare -A MODEL_TYPE=(
    [sbert]="sbert"
    [bert]="auto"
    [roberta]="auto"
    [deberta]="auto"
)

MODELS="${MODELS:-sbert bert roberta deberta}"

mkdir -p "${WORKDIR}/logs"

for SHORT in ${MODELS}; do
    if [[ -z "${HF_MODEL[$SHORT]:-}" ]]; then
        echo "ERROR: unknown model '${SHORT}'. Options: ${!HF_MODEL[*]}" >&2
        exit 1
    fi
    NAME="${SHORT}_obfus_retrained"
    echo "Submitting ${NAME}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH --job-name=${NAME}
#SBATCH --output=${WORKDIR}/logs/${NAME}_%j.out

module load cpu/0.15.4
module load anaconda3/2020.11
eval "\$(conda shell.bash hook)"
conda activate focal
cd ${WORKDIR}

python ${SCRIPT} \\
    --model "${HF_MODEL[$SHORT]}" \\
    --model_type "${MODEL_TYPE[$SHORT]}" \\
    --dataset "${DATASET}" \\
    --save_dir "${WORKDIR}/models/${NAME}" \\
    --truncate_left false
EOF
done
