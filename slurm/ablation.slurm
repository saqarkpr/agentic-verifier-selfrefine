#!/bin/bash
#SBATCH --job-name=avs-ablation
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4            # one task per feedback condition
set -euo pipefail

# NOT YET RUN ON A REAL CLUSTER. Written against Slurm conventions and
# exercised locally via scripts/run_ablation.sh; no claim is made that this
# has executed on Alps or any other HPC system. See README.
CONDITIONS=(retry generic diagnostic actionable constrained)
COND=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

export AVS_CONFIG=configs/agent_ablation.yaml
export AVS_N_SEEDS=${AVS_N_SEEDS:-3}
mkdir -p logs results

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} condition=${COND}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The control is always included so each array task is self-contained and
# interpretable on its own, rather than needing a cross-task join.
srun python src/feedback_protocol_ablation.py retry "${COND}"
