#!/usr/bin/env python3
"""Create the independent ActiveFire-Train candidate-tile training set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# code/ root on sys.path so the core/models/train/eval packages resolve
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.audit_spectral_mlp_gate import load_gate, load_pgss
from core.candidate_pgss_data import build_candidate_manifest, materialize_candidate_tiles


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/train/configs/candidate_765.json")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for candidate-tile generation")
    device = torch.device("cuda")

    gate, _gate_checkpoint = load_gate(Path(config["gate"]["checkpoint"]).resolve(), device)
    pgss, _pgss_checkpoint = load_pgss(Path(config["pgss"]["checkpoint"]).resolve(), device)
    bands = tuple(int(value) for value in config["input"]["bands"])
    tile = config["tile"]
    paths = config["outputs"]
    manifest = Path(paths["train_manifest"]).resolve()
    tensor_path = Path(paths["train_tensor"]).resolve()
    scan = build_candidate_manifest(
        csv_path=Path(config["data"]["activefire_train"]).resolve(),
        output_manifest=manifest,
        gate=gate,
        pgss=pgss,
        bands=bands,
        gate_logit_threshold=float(config["gate"]["activefire_frozen_logit_threshold"]),
        pgss_logit_threshold=float(config["pgss"]["activefire_frozen_logit_threshold"]),
        tile_size=int(tile["size"]),
        halo=int(tile["halo"]),
        batch_size=int(config["scan"]["batch_size"]),
        workers=int(config["scan"]["workers"]),
        device=device,
        seed=int(config["sampling"]["seed"]),
        caps={str(key): int(value) for key, value in config["sampling"]["caps"].items()},
    )
    materialized = materialize_candidate_tiles(
        manifest,
        tensor_path,
        bands,
        int(tile["size"]),
        int(tile["halo"]),
    )
    result = {
        "experiment": "candidate_region_pgss_training_data",
        "config": str(config_path),
        "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
        "scan": scan,
        "materialized": materialized,
        "source_data_modified": False,
        "test_data_read": False,
        "gpu_state": config.get("runtime", {}).get("gpu_state", "UNKNOWN"),
    }
    result_path = Path(paths["build_result"]).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

