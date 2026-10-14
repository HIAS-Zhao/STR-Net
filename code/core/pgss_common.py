#!/usr/bin/env python3
"""Shared helpers for PGSS-Tiny V1 training and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# Optional external power monitor (used only for optional energy logging).
# Point PGSS_POWER_MONITOR_PATH at a directory containing power_monitor.py to
# enable it; when unset, efficiency_summary() degrades to an empty dict.
POWER_ROOT = Path(os.environ["PGSS_POWER_MONITOR_PATH"]) if os.environ.get("PGSS_POWER_MONITOR_PATH") else None
if POWER_ROOT is not None and str(POWER_ROOT) not in sys.path:
    sys.path.insert(0, str(POWER_ROOT))

try:
    from power_monitor import NvidiaSmiPowerMonitor, efficiency_summary
except Exception:
    NvidiaSmiPowerMonitor = None

    def efficiency_summary(power: dict, processed_images: int, total_pixels: int) -> dict:
        return {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_rows(csv_path: str | Path, max_patches: int = 0) -> list[dict[str, str]]:
    path = Path(csv_path).resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    if "data_file" not in rows[0] or "mask_file" not in rows[0]:
        raise ValueError(f"CSV must contain data_file and mask_file: {path}")
    if max_patches and max_patches > 0:
        rows = rows[:max_patches]
    return rows


def read_mask(mask_path: str | Path) -> np.ndarray:
    path = Path(mask_path)
    try:
        with rasterio.open(path) as source:
            mask = source.read(1)
    except Exception:
        with Image.open(path) as image:
            mask = np.asarray(image).copy()
            if mask.ndim == 3:
                mask = mask[..., 0]
    return mask > 0


def scene_id_from_row(row: dict[str, str]) -> str:
    """Return the explicit scene id, or derive it from known patch suffixes."""
    if row.get("scene_id"):
        return row["scene_id"]
    stem = Path(row["data_file"]).stem
    stem = re.sub(r"_patch_\d+_\d+$", "", stem)
    stem = re.sub(r"_p\d+$", "", stem)
    return stem


class FireSegmentationCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        bands=(7, 6, 2),
        max_patches: int = 0,
        return_scene_id: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path).resolve()
        self.rows = read_rows(self.csv_path, max_patches=max_patches)
        self.bands = tuple(int(band) for band in bands)
        self.return_scene_id = bool(return_scene_id)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with rasterio.open(row["data_file"]) as source:
            image = source.read(self.bands).astype(np.float32) / 65535.0
            valid_source = source.read().astype(np.float32, copy=False)
        truth = read_mask(row["mask_file"])
        valid = np.any(valid_source != 0, axis=0)
        image[:, ~valid] = 0.0
        if image.shape[1:] != truth.shape:
            raise ValueError(
                f"Image/mask shape mismatch image={image.shape}, mask={truth.shape}, path={row['data_file']}"
            )
        sample = (
            torch.from_numpy(image),
            torch.from_numpy(truth[None].astype(np.float32)),
            torch.from_numpy(valid[None].astype(np.bool_)),
        )
        if self.return_scene_id:
            return (*sample, scene_id_from_row(row))
        return sample


def masked_bce_loss(logits: torch.Tensor, masks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, masks.float(), reduction="none")
    valid = valid.float()
    return torch.sum(loss * valid) / torch.clamp(torch.sum(valid), min=1.0)


def masked_focal_tversky_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    gamma: float = 2.0,
    focal_alpha: float = 0.85,
    tversky_fp_weight: float = 0.3,
    tversky_fn_weight: float = 0.7,
    smooth: float = 1e-6,
) -> torch.Tensor:
    masks = masks.float()
    valid_f = valid.float()
    bce = F.binary_cross_entropy_with_logits(logits, masks, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = torch.where(masks > 0.5, probs, 1.0 - probs)
    alpha_factor = torch.where(masks > 0.5, focal_alpha, 1.0 - focal_alpha)
    focal = alpha_factor * torch.pow(1.0 - p_t, gamma) * bce
    focal = torch.sum(focal * valid_f) / torch.clamp(torch.sum(valid_f), min=1.0)

    probs = probs * valid_f
    targets = masks * valid_f
    tp = torch.sum(probs * targets)
    fp = torch.sum(probs * (1.0 - targets) * valid_f)
    fn = torch.sum((1.0 - probs) * targets)
    tversky = (tp + smooth) / (tp + tversky_fp_weight * fp + tversky_fn_weight * fn + smooth)
    return 0.5 * focal + 0.5 * (1.0 - tversky)


def make_loss(name: str):
    if name == "bce":
        return masked_bce_loss
    if name == "focal_tversky":
        return masked_focal_tversky_loss
    raise ValueError(f"Unknown loss: {name}")


def update_counts(counts: dict[str, int], pred: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> None:
    pred = pred.bool() & valid.bool()
    truth = truth.bool() & valid.bool()
    valid = valid.bool()
    counts["tp"] += int(torch.count_nonzero(pred & truth).item())
    counts["fp"] += int(torch.count_nonzero(pred & ~truth).item())
    counts["fn"] += int(torch.count_nonzero(~pred & truth).item())
    counts["tn"] += int(torch.count_nonzero(~pred & ~truth & valid).item())


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (int(counts[key]) for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def scan_from_hist(pos_hist: torch.Tensor, neg_hist: torch.Tensor, bins: int) -> list[dict[str, Any]]:
    pos_hist = pos_hist.detach().cpu().long()
    neg_hist = neg_hist.detach().cpu().long()
    tp_cum = torch.flip(torch.cumsum(torch.flip(pos_hist, dims=[0]), dim=0), dims=[0])
    fp_cum = torch.flip(torch.cumsum(torch.flip(neg_hist, dims=[0]), dim=0), dims=[0])
    total_pos = int(pos_hist.sum().item())
    total_neg = int(neg_hist.sum().item())
    scan = []
    for index in range(1, bins):
        counts = {
            "tp": int(tp_cum[index].item()),
            "fp": int(fp_cum[index].item()),
            "fn": int(total_pos - tp_cum[index].item()),
            "tn": int(total_neg - fp_cum[index].item()),
        }
        scan.append({"threshold": index / bins, **metrics_from_counts(counts)})
    scan.sort(key=lambda item: (item["f1"], -item["threshold"]), reverse=True)
    return scan


def update_score_histograms(
    pos_hist: torch.Tensor,
    neg_hist: torch.Tensor,
    scores: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    bins: int,
) -> tuple[int, int]:
    scores = torch.clamp(scores[valid], 0.0, 1.0 - 1e-7)
    labels = masks[valid]
    bin_idx = torch.clamp((scores * bins).long(), 0, bins - 1)
    pos_idx = bin_idx[labels]
    neg_idx = bin_idx[~labels]
    if pos_idx.numel():
        pos_hist += torch.bincount(pos_idx, minlength=bins).to(pos_hist.dtype)
    if neg_idx.numel():
        neg_hist += torch.bincount(neg_idx, minlength=bins).to(neg_hist.dtype)
    return int(pos_idx.numel()), int(neg_idx.numel())


def gpu_snapshot(gpu_index: int) -> dict[str, Any]:
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=uuid,name,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=5,
    ).strip().splitlines()[0]
    values = [part.strip() for part in line.split(",")]
    uuid = values[0]
    process_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=5,
    ).strip()
    processes = []
    for process_line in process_text.splitlines():
        parts = [part.strip() for part in process_line.split(",")]
        if len(parts) == 4 and parts[0] == uuid:
            processes.append(
                {
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mib": float(parts[3]),
                    "is_current_process": int(parts[1]) == os.getpid(),
                }
            )
    return {
        "timestamp_unix_s": time.time(),
        "physical_gpu_index": int(gpu_index),
        "uuid": uuid,
        "name": values[1],
        "memory_used_mib": float(values[2]),
        "utilization_gpu_pct": float(values[3]),
        "power_w": float(values[4]),
        "compute_processes": processes,
    }


def contaminated(snapshot: dict[str, Any]) -> bool:
    return any(not process["is_current_process"] for process in snapshot["compute_processes"])
