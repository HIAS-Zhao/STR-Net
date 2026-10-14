#!/usr/bin/env bash
# Stage 1: train the 209-param spectral gate (SG) from scratch (12 epochs).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=code
python code/train/train_spectral_mlp_gate.py --config code/train/configs/gate_765_train.json
