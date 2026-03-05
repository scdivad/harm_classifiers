#!/bin/bash
# Submit SLURM jobs to run GCG attacks on all 6 harm classifier models.
# Assumes download_models.py has been run (models in models/).
# Usage: bash run_gcg_attacks.sh

SCRIPT="gcg.py"
WORKDIR="$(pwd)"

# Model dir, model_type, dataset, output file
JOBS=(
    "models/bert_harm_binary       bert    datasets/aegis   gcg_results/bert_harm.pt"
    "models/bert_obfus_binary      bert    datasets/obfus   gcg_results/bert_obfus.pt"
    "models/roberta_harm_binary    roberta datasets/aegis   gcg_results/roberta_harm.pt"
    "models/roberta_obfus_binary   roberta datasets/obfus   gcg_results/roberta_obfus.pt"
    "models/deberta_harm_binary    deberta datasets/aegis   gcg_results/deberta_harm.pt"
    "models/deberta_obfus_binary   deberta datasets/obfus   gcg_results/deberta_obfus.pt"
)

mkdir -p logs gcg_results

for entry in "${JOBS[@]}"; do
    read -r MODEL_DIR MODEL_TYPE DATASET OUTPUT <<< "${entry}"
    NAME="gcg_$(basename ${MODEL_DIR})"
    BATCH_SIZE=8
    echo "Submitting ${NAME} (batch_size=${BATCH_SIZE})"
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH --job-name=${NAME}
#SBATCH --output=logs/${NAME}_%j.out

module load cpu/0.15.4
module load anaconda3/2020.11
eval "\$(conda shell.bash hook)"
conda activate focal
cd ${WORKDIR}

python ${SCRIPT} \\
    --base_model_dir "${MODEL_DIR}" \\
    --model_type "${MODEL_TYPE}" \\
    --dataset_path "${DATASET}" \\
    --output "${OUTPUT}" \\
    --num_examples 100 \\
    --batch_size ${BATCH_SIZE}
EOF
done
