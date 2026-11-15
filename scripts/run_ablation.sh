#!/usr/bin/env bash
# Single-GPU equivalent of slurm/ablation.slurm. Same entrypoint, same config,
# so what runs on a workstation or in Colab is what a cluster would run.
set -euo pipefail
cd "$(dirname "$0")/.."
export AVS_CONFIG=${AVS_CONFIG:-configs/agent_ablation.yaml}
mkdir -p results logs
cd src
python feedback_protocol_ablation.py "$@"
