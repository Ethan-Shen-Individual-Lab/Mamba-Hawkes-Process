#!/usr/bin/env bash
# Run MHP on all five datasets with default batch sizes (40 epochs).

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/mhp_all_gpu1_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log file: ${LOG_FILE}"

export CUDA_VISIBLE_DEVICES=1

echo "===== $(date -Iseconds) START MHP all datasets ====="
python3 Main.py --model mhp
echo "===== $(date -Iseconds) ALL MHP RUNS COMPLETE ====="
