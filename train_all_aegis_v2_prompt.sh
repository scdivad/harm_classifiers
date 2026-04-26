#!/bin/bash
# Submit SLURM jobs to train SBERT, BERT, RoBERTa, and DeBERTa on the Aegis 2.0
# prompt-only binary safety dataset (safe / unsafe).
#
# Uses all rows including prompt-only annotations — full 30k train set.
#
# PREREQ: run build_aegis_v2_prompt_dataset.py first to create datasets/aegis_v2_prompt.
#
# Usage:
#   bash harm_classifiers/train_all_aegis_v2_prompt.sh                  # all 4
#   MODELS="bert roberta" bash harm_classifiers/train_all_aegis_v2_prompt.sh  # subset

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${WORKDIR}/train_aegis_v2_prompt_classifier.py"
DATASET="${WORKDIR}/datasets/aegis_v2_prompt"

declare -A HF_MODEL=(
    [sbert]="sentence-transformers/all-MiniLM-L6-v2"
    [bert]="bert-base-uncased"
    [roberta]="roberta-base"
    [deberta]="microsoft/deberta-v3-base"
    [deberta_large]="microsoft/deberta-v3-large"
)
declare -A MODEL_TYPE=(
    [sbert]="sbert"
    [bert]="auto"
    [roberta]="auto"
    [deberta]="auto"
    [deberta_large]="auto"
)
# Per-device batch size + grad accum (effective batch = 32 for all)
declare -A BATCH_SIZE=(
    [sbert]=32
    [bert]=32
    [roberta]=32
    [deberta]=32
    [deberta_large]=8
)
declare -A GRAD_ACCUM=(
    [sbert]=1
    [bert]=1
    [roberta]=1
    [deberta]=1
    [deberta_large]=4
)
declare -A MEM=(
    [sbert]=32G
    [bert]=32G
    [roberta]=32G
    [deberta]=32G
    [deberta_large]=64G
)

MODELS="${MODELS:-sbert bert roberta deberta}"

mkdir -p "${WORKDIR}/logs"

for SHORT in ${MODELS}; do
    if [[ -z "${HF_MODEL[$SHORT]:-}" ]]; then
        echo "ERROR: unknown model '${SHORT}'. Options: ${!HF_MODEL[*]}" >&2
        exit 1
    fi
    NAME="${SHORT}_aegis_v2_prompt"
    echo "Submitting ${NAME}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=${MEM[$SHORT]}
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
    --batch_size ${BATCH_SIZE[$SHORT]} \\
    --gradient_accumulation_steps ${GRAD_ACCUM[$SHORT]} \\
    --truncate_left false
EOF
done
