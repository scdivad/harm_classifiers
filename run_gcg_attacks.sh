#!/bin/bash
# Submit SLURM jobs to run GCG attacks, then aggregate results.
# Reads job definitions from a config file (one job per line).
#
# Usage:
#   bash run_gcg_attacks.sh                  # uses gcg_jobs.txt
#   bash run_gcg_attacks.sh my_jobs.txt      # uses custom file

JOBS_FILE="${1:-gcg_jobs.txt}"
SCRIPT="gcg.py"
WORKDIR="$(pwd)"

if [ ! -f "${JOBS_FILE}" ]; then
    echo "Error: jobs file '${JOBS_FILE}' not found"
    echo "Usage: bash run_gcg_attacks.sh [jobs_file]"
    exit 1
fi

mkdir -p logs gcg_results

JOB_IDS=()

while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and blank lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue

    read -r MODEL_DIR MODEL_TYPE DATASET OUTPUT BATCH_SIZE SEARCH_WIDTH <<< "${line}"

    # Defaults
    BATCH_SIZE="${BATCH_SIZE:-8}"
    SEARCH_WIDTH="${SEARCH_WIDTH:-2048}"

    NAME="gcg_$(basename ${MODEL_DIR})"
    echo "Submitting ${NAME} (batch_size=${BATCH_SIZE}, search_width=${SEARCH_WIDTH})"

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH --job-name=${NAME}
#SBATCH --output=logs/${NAME}_%j.out

module load cpu/0.15.4
module load anaconda3/2020.11
eval "\$(conda shell.bash hook)"
conda activate focal
cd ${WORKDIR}

python -u ${SCRIPT} \\
    --base_model_dir "${MODEL_DIR}" \\
    --model_type "${MODEL_TYPE}" \\
    --dataset_path "${DATASET}" \\
    --output "${OUTPUT}" \\
    --num_examples 100 \\
    --batch_size ${BATCH_SIZE} \\
    --search_width ${SEARCH_WIDTH}
EOF
    )

    echo "  -> Job ID: ${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")

done < "${JOBS_FILE}"

# Submit aggregation job that runs after all attack jobs finish
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
#SBATCH -t 00:05:00
#SBATCH --job-name=gcg_aggregate
#SBATCH --output=logs/gcg_aggregate_%j.out

module load cpu/0.15.4
module load anaconda3/2020.11
eval "\$(conda shell.bash hook)"
conda activate focal
cd ${WORKDIR}

python -u inspect_gcg_results.py
EOF
    echo "All jobs submitted."
else
    echo "No jobs submitted."
fi
