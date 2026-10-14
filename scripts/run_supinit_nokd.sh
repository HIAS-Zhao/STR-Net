#!/usr/bin/env bash
# Stage 4: SupInit No-KD refinement - 40-epoch supervised continuation
# initialized from the candidate-pretraining checkpoint. For the random-init
# ablation set training.initialization="random" and
# sources.initialization_checkpoint=null in the config first.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=code
python code/train/train_candidate_nokd.py --config code/train/configs/supinit_nokd_40e.json
