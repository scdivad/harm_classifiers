#!/bin/bash
# Submit SLURM jobs to train BERT, RoBERTa, and DeBERTa on hhrlhf.
# Usage: bash train_all_harm.sh

SCRIPT="train_harm_classifier.py"
DATASET="datasets/hhrlhf"
WORKDIR="$(pwd)"

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

mkdir -p logs

for i in "${!MODELS[@]}"; do
    NAME="${SHORT_NAMES[$i]}_hhrlhf_binary"
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
#SBATCH --output=logs/${NAME}_%j.out

module load cpu/0.15.4
module load anaconda3/2020.11
eval "\$(conda shell.bash hook)"
conda activate focal
cd ${WORKDIR}

python ${SCRIPT} \\
    --model "${MODELS[$i]}" \\
    --dataset "${DATASET}" \\
    --save_dir "models/${NAME}"
EOF
done
