#!/bin/bash
# Submit SLURM jobs for GCG attacks with 5 random restarts on aegis dataset.
# Then submit an aggregation job to analyze per-restart ASR.
# set -x

SCRIPT="gcg.py"
WORKDIR="$(pwd)"
DATASET="datasets/aegis"
NUM_RESTARTS=3
BATCH_SIZE=8
SEARCH_WIDTH=2048
NUM_EXAMPLES=100

mkdir -p logs gcg_results

# Define the 4 models (model_dir  model_type  output_name)
declare -a MODELS=(
    "models/bert_harm_binary|bert|bert_harm_r5"
    "models/roberta_harm_binary|roberta|roberta_harm_r5"
    "models/deberta_harm_binary|deberta|deberta_harm_r5"
    "models/sbert_combined_binary|sbert|sbert_harm_r5"
)

JOB_IDS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_DIR MODEL_TYPE OUTPUT_NAME <<< "${entry}"
    OUTPUT="gcg_results/${OUTPUT_NAME}.pt"
    NAME="gcg_${OUTPUT_NAME}"

    echo "Submitting ${NAME}: ${MODEL_DIR} (${MODEL_TYPE})"

    JOB_ID=$(sbatch --parsable <<EOF
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

python -u ${SCRIPT} \
    --base_model_dir "${MODEL_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --dataset_path "${DATASET}" \
    --output "${OUTPUT}" \
    --num_examples ${NUM_EXAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --search_width ${SEARCH_WIDTH} \
    --num_restarts ${NUM_RESTARTS} \
    --resume
EOF
    )

    echo "  -> Job ID: ${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")
done

# Submit aggregation job after all attacks finish
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
    echo ""
    echo "Submitting aggregation job (depends on: ${DEP_STR})"

    sbatch --parsable --dependency=afterany:${DEP_STR} <<EOF
#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --gpus=1
#SBATCH -t 00:05:00
#SBATCH --job-name=gcg_restart_analysis
#SBATCH --output=logs/gcg_restart_analysis_%j.out

module load cpu/0.15.4
module load anaconda3/2020.11
eval "\$(conda shell.bash hook)"
conda activate focal
cd ${WORKDIR}

python -u analyze_restarts.py --pattern "gcg_results/*_r5.pt" --num_restarts ${NUM_RESTARTS}
EOF
    echo "All jobs submitted."
else
    echo "No jobs submitted."
fi
