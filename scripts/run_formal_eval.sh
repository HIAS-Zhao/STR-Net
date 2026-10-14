#!/usr/bin/env bash
# Formal frozen-threshold evaluation: verifier threshold re-selected on
# ActiveFire-Val (exact max-F1), then ActiveFire-Test (source-frozen),
# Land8Fire-Test zero-shot and Land8Fire-Test target-calibrated.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=code
python code/eval/evaluate_candidate_pgss_formal_test.py --config code/eval/configs/formal_eval_765.json --device cuda
