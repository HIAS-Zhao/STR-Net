#!/usr/bin/env python3
"""Train Candidate-PGSS on the same Gate-selected tensor, random or SupInit."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# code/ root on sys.path so the core/models/train/eval packages resolve
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.audit_spectral_mlp_gate import load_gate, load_pgss
from core.candidate_pgss_data import CandidateTensorDataset
from eval.evaluate_candidate_pgss_formal_test import collect_split, exact_f1_threshold, summarize_protocol
from core.pgss_common import make_loss, sha256_file, write_json
from models.pgss_model import PGSSTinyV1, count_parameters, estimate_macs_per_pixel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def priority(validation: dict[str, Any]) -> tuple[float, float, float, float]:
    events = validation.get("event_metrics", {})
    bins = {row["event_size_bin"]: row for row in validation.get("event_size_summary", [])}
    total = sum(int(bins.get(key, {}).get("events", 0)) for key in ("1", "2-4"))
    hit = sum(int(bins.get(key, {}).get("recalled", 0)) for key in ("1", "2-4"))
    small = hit / total if total else 0.0
    return (
        float(events.get("final_event_recall", 0.0)),
        float(small),
        float(validation["metrics"]["f1"]),
        float(validation["pr_auc"]),
    )


def load_initialization(model: PGSSTinyV1, path: Path, config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "bands": [7, 6, 5],
        "width": int(config["training"]["width"]),
        "head_width": int(config["training"]["head_width"]),
        "tile_size": int(config["tile"]["size"]),
        "halo": int(config["tile"]["halo"]),
    }
    for key, value in expected.items():
        actual = checkpoint.get(key)
        if key == "bands":
            mismatch = actual is not None and list(actual) != value
        else:
            mismatch = actual is not None and actual != value
        if mismatch:
            raise RuntimeError(f"Initialization metadata mismatch for {key}: {actual!r} != {value!r}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"No model_state_dict in initialization checkpoint: {path}")
    model.load_state_dict(state, strict=True)
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["outputs"]["model_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "TRAIN_REPORT.json"
    if report_path.exists() and not args.force:
        print(json.dumps({"status": "SKIPPED", "report": str(report_path)}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this script through the GPU runner")
    seed = int(config["training"]["seed"])
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    tensor_path = Path(config["sources"]["candidate_tensor"]).resolve()
    dataset = CandidateTensorDataset(tensor_path)
    workers = int(config["training"]["workers"] if args.workers < 0 else args.workers)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    bands = tuple(int(value) for value in config["input"]["bands"])
    width = int(config["training"]["width"])
    head_width = int(config["training"]["head_width"])
    tile_size = int(config["tile"]["size"])
    halo = int(config["tile"]["halo"])
    model = PGSSTinyV1(in_channels=len(bands), width=width, head_width=head_width).to(device)
    init_mode = str(config["training"]["initialization"])
    init_path = config["sources"].get("initialization_checkpoint")
    init_checkpoint = None
    if init_mode == "kd_free_candidate_pgss":
        if not init_path:
            raise RuntimeError("SupInit requires sources.initialization_checkpoint")
        init_path = str(Path(init_path).resolve())
        if not Path(init_path).exists():
            raise FileNotFoundError(init_path)
        init_checkpoint = load_initialization(model, Path(init_path), config)
    elif init_mode != "random":
        raise ValueError(f"Unsupported initialization: {init_mode}")
    criterion = make_loss(config["training"]["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    gate, _ = load_gate(Path(config["sources"]["gate_checkpoint"]).resolve(), device)
    scanner, _ = load_pgss(Path(config["sources"]["scanner_checkpoint"]).resolve(), device)
    gate.eval()
    scanner.eval()
    gate_threshold = float(config["thresholds"]["gate_logit_threshold"])
    scanner_threshold = float(config["thresholds"]["scanner_logit_threshold"])
    val_csv = Path(config["data"]["activefire_val"]).resolve()
    started = time.perf_counter()
    history = []
    best_record = None
    best_key = None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, targets, valid, _category_ids in tqdm(loader, desc=f"{init_mode} seed={seed} train {epoch}/{epochs}", unit="batch"):
            images = images.float().to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)
            valid = valid.bool().to(device, non_blocking=True)
            logits = model(images)[:, :, halo:halo + tile_size, halo:halo + tile_size]
            loss = criterion(logits, targets, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()
        torch.cuda.synchronize(device)
        model.eval()
        collected = collect_split(
            gate, model, scanner, val_csv, "ActiveFire-Val",
            gate_threshold, scanner_threshold, bands,
            int(config["evaluation"]["batch_size"]),
            int(config["evaluation"]["workers"]),
            tile_size, halo,
            int(config["evaluation"]["candidate_tile_batch_size"]), device,
        )
        threshold = exact_f1_threshold(
            collected["scores"], collected["labels"],
            collected["positive_pixels"], collected["valid_pixels"],
        )
        validation = summarize_protocol(collected, threshold, "ActiveFire-Val-source-frozen", compute_events=True)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": validation,
            "threshold": threshold,
            "priority": list(priority(validation)),
        }
        history.append(record)
        payload = {
            "model_state_dict": model.state_dict(),
            "architecture": "PGSSTinyV1_Gate765_NoKD",
            "bands": list(bands), "width": width, "head_width": head_width,
            "dilations": [1, 2], "epoch": epoch, "tile_size": tile_size, "halo": halo,
            "loss": config["training"]["loss"], "seed": seed,
            "initialization": init_mode, "teacher_used": False, "kd_used": False,
            "candidate_tensor": str(tensor_path),
            "candidate_tensor_sha256": sha256_file(tensor_path),
            "threshold": threshold, "validation": validation, "test_used": False,
        }
        epoch_path = output / f"model_epoch_{epoch:02d}.pth"
        torch.save(payload, epoch_path)
        write_json(output / f"epoch_{epoch:02d}_activefire_val.json", record)
        key = tuple(record["priority"])
        if best_key is None or key > best_key:
            best_key = key
            best_record = record
            shutil.copy2(epoch_path, output / "model_best.pth")
        print(
            f"Epoch {epoch}: loss={record['train_loss']:.6f} "
            f"valF1={validation['metrics']['f1']:.6f} "
            f"EventR={validation.get('event_metrics', {}).get('final_event_recall', float('nan')):.6f}",
            flush=True,
        )
    report = {
        "algorithm": "Gate-PGSS-765 No-KD on Gate-selected Candidate tensor",
        "status": "trained_from_scratch" if init_mode == "random" else "trained_from_kd_free_candidate_init",
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "seed": seed, "bands": list(bands), "normalization": "/65535",
        "input_shape": [3, 48, 48], "loss_shape": [1, 32, 32],
        "tile_size": tile_size, "halo": halo, "epochs": epochs,
        "batch_size": int(config["training"]["batch_size"]), "workers": workers,
        "loss": config["training"]["loss"], "optimizer": "AdamW",
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "initialization": init_mode, "teacher_used": False, "kd_used": False,
        "pretrain_checkpoint": str(Path(init_path).resolve()) if init_path else None,
        "pretrain_checkpoint_sha256": sha256_file(Path(init_path).resolve()) if init_path else None,
        "gate_checkpoint": config["sources"]["gate_checkpoint"],
        "gate_checkpoint_sha256": sha256_file(Path(config["sources"]["gate_checkpoint"]).resolve()),
        "gate_logit_threshold": gate_threshold, "scanner_logit_threshold": scanner_threshold,
        "candidate_tensor": str(tensor_path), "candidate_tensor_sha256": sha256_file(tensor_path),
        "parameter_info": count_parameters(model),
        "macs_per_pixel": estimate_macs_per_pixel(len(bands), width, head_width),
        "selected_epoch": best_record["epoch"] if best_record else None,
        "official_checkpoint": str(output / "model_best.pth"),
        "official_checkpoint_sha256": sha256_file(output / "model_best.pth"),
        "checkpoint_selection": "ActiveFire-Val event recall, small-event recall, pixel F1, then PR-AUC",
        "test_used": False, "wall_time_s": time.perf_counter() - started,
        "history": history,
    }
    write_json(report_path, report)
    print(json.dumps({"status": "COMPLETE", "initialization": init_mode, "selected_epoch": report["selected_epoch"], "official_checkpoint": report["official_checkpoint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
