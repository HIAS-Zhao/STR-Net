#!/usr/bin/env python3
"""Train SpectralMLPGate from scratch and select checkpoints on ActiveFire-Val only."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# code/ root on sys.path so the core/models/train/eval packages resolve
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.audit_spectral_mlp_gate import validation_exact_audit
from core.pgss_common import FireSegmentationCsvDataset, make_loss, sha256_file, write_json
from models.spectral_mlp_gate import (
    SpectralMLPGate,
    count_parameters,
    estimate_macs_per_pixel,
    prepare_gate_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="code/train/configs/gate_765_train.json",
    )
    return parser.parse_args()


def checkpoint_payload(
    model: SpectralMLPGate,
    epoch: int,
    validation: dict,
    config: dict,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "architecture": "SpectralMLPGate_3_16_8_1",
        "in_channels": 3,
        "hidden1": 16,
        "hidden2": 8,
        "bands": tuple(config["input"]["bands"]),
        "normalization": config["input"]["normalization"],
        "invalid_policy": config["input"]["invalid_policy"],
        "loss": config["training"]["loss"],
        "epoch": epoch,
        "parameter_info": count_parameters(model),
        "macs_per_pixel": estimate_macs_per_pixel(),
        "checkpoint_selection_validation": validation,
        "seed": config["training"]["seed"],
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SpectralMLPGate training")

    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    bands = tuple(int(value) for value in config["input"]["bands"])
    run_dir = Path(config["outputs"]["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_csv = Path(config["data"]["activefire_train"]).resolve()
    val_csv = Path(config["data"]["activefire_val"]).resolve()
    train_dataset = FireSegmentationCsvDataset(train_csv, bands=bands)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(training["workers"]) > 0,
    )

    model = SpectralMLPGate().to(device)
    parameter_info = count_parameters(model)
    macs_per_pixel = estimate_macs_per_pixel()
    if parameter_info["total_parameters"] != int(config["architecture"]["expected_parameters"]):
        raise AssertionError(f"Unexpected parameter count: {parameter_info}")
    if macs_per_pixel != int(config["architecture"]["expected_macs_per_pixel"]):
        raise AssertionError(f"Unexpected MAC count: {macs_per_pixel}")
    criterion = make_loss(training["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    validation_tile = {
        "size": int(training["checkpoint_tile_size"]),
        "halo": 0,
    }
    recall_constraint = float(training["checkpoint_recall_constraint"])
    history = []
    best: dict | None = None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, masks, valid in tqdm(
            train_loader, desc=f"SpectralMLPGate train {epoch}/{epochs}", unit="batch"
        ):
            images = images.to(device, non_blocking=True)
            masks = (masks > 0).float().to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True).bool()
            logits = model(prepare_gate_input(images, valid))
            loss = criterion(logits, masks, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()
        torch.cuda.synchronize()

        validation = validation_exact_audit(
            model=model,
            csv_path=val_csv,
            split_name="ActiveFire-Val",
            bands=bands,
            constraints=[recall_constraint],
            tile_specs=[validation_tile],
            batch_size=int(training["batch_size"]),
            workers=max(1, int(training["workers"]) // 2),
            device=device,
            collect_curve=False,
            collect_events=False,
        )
        tile_result = validation["tile_rows"][0]
        selection = validation["selections"][0]
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_pr_auc": validation["pr_auc"],
            "validation_selection": selection,
            "validation_tile": tile_result,
            "validation_audit_json": str(run_dir / f"epoch_{epoch:02d}_activefire_val_audit.json"),
            "checkpoint": str(run_dir / f"SpectralMLPGate_epoch_{epoch:02d}.pth"),
        }
        history.append(epoch_record)
        write_json(run_dir / f"epoch_{epoch:02d}_activefire_val_audit.json", validation)
        payload = checkpoint_payload(model, epoch, epoch_record, config)
        torch.save(payload, run_dir / f"SpectralMLPGate_epoch_{epoch:02d}.pth")

        satisfies = float(selection["pixel_gate_recall"]) >= recall_constraint
        score = (float(tile_result["candidate_tile_ratio"]), -float(validation["pr_auc"]))
        if satisfies and (best is None or score < tuple(best["selection_score"])):
            best = {
                **epoch_record,
                "selection_score": list(score),
                "selection_reason": "minimum ActiveFire-Val 32x32 candidate tile ratio at pixel Recall >=99.5%; exact PR-AUC tie-break",
            }
            torch.save(payload, Path(config["outputs"]["best_checkpoint"]).resolve())
        print(
            f"Epoch {epoch}: loss={mean_loss:.6f} val_recall={selection['pixel_gate_recall']:.7f} "
            f"tile32={tile_result['candidate_tile_ratio']:.7f} AP={validation['pr_auc']:.7f}"
        )

    status = "trained" if best is not None else "NO-GO_NO_VALID_CHECKPOINT"
    report = {
        "algorithm": "SpectralMLPGate_3_16_8_1",
        "status": status,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "train_csv": str(train_csv),
        "train_csv_sha256": sha256_file(train_csv),
        "val_csv": str(val_csv),
        "val_csv_sha256": sha256_file(val_csv),
        "test_data_used_for_selection": False,
        "bands": list(bands),
        "normalization": config["input"]["normalization"],
        "augmentation": training["augmentation"],
        "loss": training["loss"],
        "optimizer": training["optimizer"],
        "scheduler": training["scheduler"],
        "seed": seed,
        "epochs": epochs,
        "batch_size": training["batch_size"],
        "workers": training["workers"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "parameter_info": parameter_info,
        "macs_per_pixel": macs_per_pixel,
        "estimated_macs_256x256": macs_per_pixel * 256 * 256,
        "checkpoint_selection": training.get("checkpoint_tie_break", training.get("checkpoint_selection", "ActiveFire-Val minimum candidate tile ratio at recall constraint; PR-AUC tie-break")),
        "best": best,
        "best_checkpoint": config["outputs"]["best_checkpoint"] if best is not None else None,
        "history": history,
        "wall_time_s": time.perf_counter() - started,
        "gpu_state": config["runtime"]["gpu_state"],
        "power_energy_policy": config["runtime"]["power_energy_policy"],
    }
    write_json(Path(config["outputs"]["train_result_json"]).resolve(), report)
    print(json.dumps({"status": status, "best": best}, indent=2, ensure_ascii=False))
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
