#!/usr/bin/env python3
"""Train PGSS-W32 from scratch on Gate-selected ActiveFire-Train tiles."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# code/ root on sys.path so the core/models/train/eval packages resolve
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.audit_spectral_mlp_gate import load_gate, load_pgss
from core.candidate_pgss_data import CandidateTensorDataset, CATEGORIES
from core.candidate_pgss_eval import evaluate_candidate_pipeline
from core.pgss_common import make_loss, sha256_file, write_json
from models.pgss_model import PGSSTinyV1, count_parameters, estimate_macs_per_pixel


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def probability_from_logit(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + np.exp(-value))
    exp_value = np.exp(value)
    return float(exp_value / (1.0 + exp_value))


def make_sampler(dataset: CandidateTensorDataset, fractions: dict[str, float]) -> WeightedRandomSampler:
    counts = torch.bincount(dataset.category_ids, minlength=len(dataset.categories)).float()
    target = torch.tensor(
        [float(fractions.get(category, 0.0)) for category in dataset.categories],
        dtype=torch.float32,
    )
    if float(target.sum()) <= 0:
        raise ValueError("Category sampling fractions must have a positive sum")
    target = target / target.sum()
    weights_by_category = torch.zeros_like(target)
    for index in range(len(target)):
        if counts[index] > 0:
            weights_by_category[index] = target[index] / counts[index]
    weights = weights_by_category[dataset.category_ids]
    return WeightedRandomSampler(weights.double(), num_samples=len(dataset), replacement=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/train/configs/candidate_765.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for candidate PGSS training")
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    tensor_path = Path(config["outputs"]["train_tensor"]).resolve()
    train_dataset = CandidateTensorDataset(tensor_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        sampler=make_sampler(train_dataset, config["training"]["category_sampling_fractions"]),
        num_workers=int(config["training"]["workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(config["training"]["workers"]) > 0,
    )

    bands = tuple(int(value) for value in config["input"]["bands"])
    width = int(config["training"]["width"])
    head_width = int(config["training"]["head_width"])
    tile_size = int(config["tile"]["size"])
    halo = int(config["tile"]["halo"])
    model = PGSSTinyV1(in_channels=len(bands), width=width, head_width=head_width).to(device)
    criterion = make_loss(config["training"]["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    parameter_info = count_parameters(model)
    macs_per_pixel = estimate_macs_per_pixel(len(bands), width, head_width)

    gate, _gate_checkpoint = load_gate(Path(config["gate"]["checkpoint"]).resolve(), device)
    pgss, _pgss_checkpoint = load_pgss(Path(config["pgss"]["checkpoint"]).resolve(), device)
    val_protocol = {
        "name": "activefire_val_source_frozen_gate",
        "gate_selection_split": "ActiveFire-Val",
        "gate_logit_threshold": float(config["gate"]["activefire_frozen_logit_threshold"]),
        "pgss_logit_threshold": float(config["pgss"]["activefire_frozen_logit_threshold"]),
        "tile_size": tile_size,
        "halo": halo,
    }
    output_dir = Path(config["outputs"]["model_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_record = None
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, targets, valid, _category_ids in tqdm(
            train_loader, desc=f"candidate train {epoch}/{epochs}", unit="batch"
        ):
            images = images.float().to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)
            valid = valid.bool().to(device, non_blocking=True)
            logits = model(images)
            core_logits = logits[:, :, halo:halo + tile_size, halo:halo + tile_size]
            loss = criterion(core_logits, targets, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()
        torch.cuda.synchronize()
        validation = evaluate_candidate_pipeline(
            model,
            gate,
            pgss,
            Path(config["data"]["activefire_val"]).resolve(),
            "ActiveFire-Val",
            val_protocol,
            bands,
            int(config["evaluation"]["batch_size"]),
            int(config["evaluation"]["workers"]),
            device,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": validation,
        }
        history.append(record)
        candidate_f1 = float(validation["gate_candidate_pgss"]["f1"])
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "architecture": "PGSSTinyV1_candidate_region",
            "bands": bands,
            "width": width,
            "head_width": head_width,
            "loss": config["training"]["loss"],
            "epoch": epoch,
            "tile_size": tile_size,
            "halo": halo,
            "candidate_training_tensor": str(tensor_path),
            "candidate_training_tensor_sha256": sha256_file(tensor_path),
            "best_validation": validation["gate_candidate_pgss"],
            "candidate_threshold": validation["candidate_threshold"],
            "parameter_info": parameter_info,
            "macs_per_pixel": macs_per_pixel,
            "from_scratch": True,
        }
        torch.save(checkpoint_payload, output_dir / f"model_epoch_{epoch:02d}.pth")
        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best_record = {"epoch": epoch, **validation["candidate_threshold"], "f1": candidate_f1}
            torch.save(checkpoint_payload, output_dir / "model_best.pth")
        print(
            f"Epoch {epoch}: loss={record['train_loss']:.6f} "
            f"candidate_val_f1={candidate_f1:.6f} "
            f"candidate_logit_threshold={validation['candidate_threshold']['logit_threshold']:.6f}"
        )
        write_json(output_dir / f"epoch_{epoch:02d}_activefire_val.json", validation)

    report = {
        "algorithm": "Gate-selected candidate-region PGSS-W32",
        "status": "trained_from_scratch",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "training_tensor": str(tensor_path),
        "training_tensor_sha256": sha256_file(tensor_path),
        "bands": list(bands),
        "normalization": "/65535",
        "tile_size": tile_size,
        "halo": halo,
        "loss": config["training"]["loss"],
        "parameter_info": parameter_info,
        "macs_per_pixel": macs_per_pixel,
        "estimated_macs_48x48": macs_per_pixel * (tile_size + 2 * halo) ** 2,
        "epochs": epochs,
        "batch_size": int(config["training"]["batch_size"]),
        "workers": int(config["training"]["workers"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "category_sampling_fractions": config["training"]["category_sampling_fractions"],
        "from_scratch": True,
        "checkpoint_selection": "ActiveFire-Val candidate-pipeline F1 with exact logit threshold",
        "best": best_record,
        "best_model": str(output_dir / "model_best.pth"),
        "wall_time_s": time.perf_counter() - started,
        "gpu_state": config.get("runtime", {}).get("gpu_state", "UNKNOWN"),
        "power_energy_policy": config.get("runtime", {}).get("power_energy_policy", "not collected"),
        "test_used": False,
        "history": history,
    }
    write_json(output_dir / "train_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

