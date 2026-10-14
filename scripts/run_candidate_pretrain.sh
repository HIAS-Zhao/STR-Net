#!/usr/bin/env bash
# Stages 2+3: mine candidate tiles with the frozen gate + frozen full-PGSS
# scanner, then supervised candidate pretraining of the 4,401-param verifier.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=code
python code/train/build_candidate_pgss_tiles.py --config code/train/configs/candidate_765.json
python code/train/train_candidate_pgss.py --config code/train/configs/candidate_765.json
