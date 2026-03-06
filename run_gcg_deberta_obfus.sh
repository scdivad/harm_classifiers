#!/bin/bash
# Run GCG attack on both deberta models.
# Usage: bash run_gcg_deberta_obfus.sh

WORKDIR="$(pwd)"

# deberta + obfus
sbatch <<EOF
#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH --job-name=gcg_deberta_obfus
#SBATCH --output=logs/gcg_deberta_obfus_%j.out

module load cpu/0.15.4
module load anaconda3/2020.11
eval "\$(conda shell.bash hook)"
conda activate focal
cd ${WORKDIR}

python -u gcg.py \\
    --base_model_dir "models/deberta_obfus_binary" \\
    --model_type "deberta" \\
    --dataset_path "datasets/obfus" \\
    --output "gcg_results/deberta_obfus.pt" \\
    --num_examples 100 \\
    --batch_size 8 \\
    --search_width 2048
EOF

# deberta + harm (aegis)
sbatch <<EOF
#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH --job-name=gcg_deberta_harm
#SBATCH --output=logs/gcg_deberta_harm_%j.out

module load cpu/0.15.4
module load anaconda3/2020.11
eval "\$(conda shell.bash hook)"
conda activate focal
cd ${WORKDIR}

python -u gcg.py \\
    --base_model_dir "models/deberta_harm_binary" \\
    --model_type "deberta" \\
    --dataset_path "datasets/aegis" \\
    --output "gcg_results/deberta_harm.pt" \\
    --num_examples 100 \\
    --batch_size 8 \\
    --search_width 2048
EOF
