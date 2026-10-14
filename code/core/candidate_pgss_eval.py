#!/usr/bin/env python3
"""Validation-only evaluation for the Gate-selected PGSS verifier."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.audit_spectral_mlp_gate import build_tile_masks, make_audit_loader, prediction_count_tensors
from core.pgss_common import metrics_from_counts, sha256_file
from models.pgss_model import (
    PGSSTinyV1,
    count_parameters,
    estimate_macs_per_pixel,
    forward_boundary_aligned_tile,
)
from models.spectral_mlp_gate import prepare_gate_input


COUNT_KEYS = ("tp", "fp", "fn", "tn")


def load_candidate_model(path: Path, device: torch.device) -> tuple[PGSSTinyV1, dict[str, Any]]:
    # Keep training/audit metadata on CPU and move only the instantiated model.
    # Some checkpoints contain validation tensors that should never consume GPU memory.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    bands = tuple(checkpoint.get("bands", (7, 6, 2)))
    model = PGSSTinyV1(
        in_channels=len(bands),
        width=int(checkpoint.get("width", 32)),
        head_width=int(checkpoint.get("head_width", 8)),
    )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()
    return model, checkpoint


def exact_candidate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    total_positive_pixels: int,
    total_valid_pixels: int | None = None,
) -> tuple[dict[str, Any], float]:
    """Find the exact threshold for a candidate-only pipeline.

    Scores contain valid pixels inside selected base tiles only. Positives
    outside those tiles remain unavoidable FN through total_positive_pixels.
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.bool_)
    if scores.size == 0 or not np.any(labels):
        raise RuntimeError("Candidate validation scores contain no positive pixel")
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )
    tp_cum = np.cumsum(sorted_labels, dtype=np.int64)
    fp_cum = np.cumsum(~sorted_labels, dtype=np.int64)
    thresholds = sorted_scores[boundaries]
    tp = tp_cum[boundaries]
    fp = fp_cum[boundaries]
    has_positive_group = np.zeros_like(thresholds, dtype=np.bool_)
    start = np.r_[0, boundaries[:-1] + 1]
    for index, (left, right) in enumerate(zip(start, boundaries)):
        has_positive_group[index] = bool(np.any(sorted_labels[left:right + 1]))
    thresholds = thresholds[has_positive_group]
    tp = tp[has_positive_group]
    fp = fp[has_positive_group]
    fn = total_positive_pixels - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    recall = tp.astype(np.float64) / max(1, total_positive_pixels)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_f1 = float(np.max(f1))
    best_indices = np.flatnonzero(f1 == best_f1)
    best_index = int(best_indices[0])  # thresholds are descending: first is highest
    best = {
        "logit_threshold": float(thresholds[best_index]),
        "tp": int(tp[best_index]),
        "fp": int(fp[best_index]),
        "fn": int(fn[best_index]),
        "tn": int(
            (total_valid_pixels if total_valid_pixels is not None else total_positive_pixels + (scores.size - int(labels.sum())))
            - tp[best_index]
            - fp[best_index]
            - fn[best_index]
        ),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": best_f1,
        "predicted_positive_pixels": int(tp[best_index] + fp[best_index]),
    }
    if best["logit_threshold"] >= 0:
        best["probability_threshold"] = float(1.0 / (1.0 + math.exp(-best["logit_threshold"])))
    else:
        exp_value = math.exp(best["logit_threshold"])
        best["probability_threshold"] = float(exp_value / (1.0 + exp_value))

    group_precision = precision
    group_recall = recall
    recall_increment = np.diff(np.r_[0.0, group_recall])
    average_precision = float(np.sum(group_precision * recall_increment))
    return best, average_precision


def _counts_from_numpy(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    prediction = prediction.astype(bool, copy=False) & valid
    truth = truth.astype(bool, copy=False) & valid
    return {
        "tp": int(np.count_nonzero(prediction & truth)),
        "fp": int(np.count_nonzero(prediction & ~truth)),
        "fn": int(np.count_nonzero(~prediction & truth & valid)),
        "tn": int(np.count_nonzero(~prediction & ~truth & valid)),
    }


def metrics_for_candidate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    total_positive_pixels: int,
    total_valid_pixels: int,
    threshold: float,
) -> dict[str, Any]:
    prediction = scores >= float(threshold)
    tp = int(np.count_nonzero(prediction & labels))
    fp = int(np.count_nonzero(prediction & ~labels))
    fn = int(total_positive_pixels - tp)
    metrics = metrics_from_counts(
        {"tp": tp, "fp": fp, "fn": fn, "tn": total_valid_pixels - tp - fp - fn}
    )
    return {
        "logit_threshold": float(threshold),
        "probability_threshold": probability_from_logit(float(threshold)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": total_valid_pixels - tp - fp - fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "predicted_positive_pixels": tp + fp,
    }


def probability_from_logit(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


@torch.inference_mode()
def candidate_core_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    selected: torch.Tensor,
    tile_size: int,
    halo: int,
    tile_batch_size: int = 256,
) -> torch.Tensor:
    """Run the candidate model on selected tile+halo windows and return core logits."""
    batch_size, _channels, height, width = images.shape
    output = torch.full(
        (batch_size, 1, height, width),
        -torch.inf,
        dtype=torch.float32,
        device=images.device,
    )
    padded = F.pad(images, (halo, halo, halo, halo), value=0.0)
    image_valid = torch.ones(
        (batch_size, 1, height, width), dtype=torch.bool, device=images.device
    )
    padded_valid = F.pad(image_valid, (halo, halo, halo, halo), value=False)
    windows: list[torch.Tensor] = []
    valid_windows: list[torch.Tensor] = []
    locations: list[tuple[int, int, int, int]] = []
    grid_coordinates = [torch.nonzero(selected[index, 0], as_tuple=False).cpu().tolist() for index in range(batch_size)]
    for batch_index, coordinates in enumerate(grid_coordinates):
        for grid_y, grid_x in coordinates:
            y0 = int(grid_y) * tile_size
            x0 = int(grid_x) * tile_size
            windows.append(
                padded[batch_index:batch_index + 1, :, y0:y0 + tile_size + 2 * halo, x0:x0 + tile_size + 2 * halo]
            )
            valid_windows.append(
                padded_valid[batch_index:batch_index + 1, :, y0:y0 + tile_size + 2 * halo, x0:x0 + tile_size + 2 * halo]
            )
            locations.append((batch_index, y0, x0, len(windows) - 1))
    if not windows:
        return output
    for start in range(0, len(windows), tile_batch_size):
        stop = min(len(windows), start + tile_batch_size)
        tile_inputs = torch.cat(windows[start:stop], dim=0)
        tile_valid = torch.cat(valid_windows[start:stop], dim=0)
        tile_outputs = forward_boundary_aligned_tile(model, tile_inputs, tile_valid)[
            :, :, halo:halo + tile_size, halo:halo + tile_size
        ]
        for local_index, (batch_index, y0, x0, global_index) in enumerate(locations[start:stop]):
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)
            output[batch_index, :, y0:y1, x0:x1] = tile_outputs[
                local_index, :, :y1 - y0, :x1 - x0
            ]
    return output


def _empty_scene() -> dict[str, int]:
    return {
        "patches": 0,
        "valid_pixels": 0,
        "positive_pixels": 0,
        "candidate_valid_pixels": 0,
        "candidate_positive_pixels": 0,
        "candidate_tiles": 0,
        "eligible_tiles": 0,
        "full_pgss_tp": 0,
        "full_pgss_fp": 0,
        "full_pgss_fn": 0,
        "full_pgss_tn": 0,
        "gate_pgss_tp": 0,
        "gate_pgss_fp": 0,
        "gate_pgss_fn": 0,
        "gate_pgss_tn": 0,
        "candidate_tp": 0,
        "candidate_fp": 0,
        "candidate_fn": 0,
        "candidate_tn": 0,
    }


@torch.inference_mode()
def evaluate_candidate_pipeline(
    candidate_model: torch.nn.Module,
    gate: torch.nn.Module,
    pgss: torch.nn.Module,
    csv_path: Path,
    split_name: str,
    protocol: dict[str, Any],
    bands: tuple[int, ...],
    batch_size: int,
    workers: int,
    device: torch.device,
    fixed_candidate_logit_threshold: float | None = None,
) -> dict[str, Any]:
    candidate_model.eval()
    gate.eval()
    pgss.eval()
    _dataset, loader = make_audit_loader(csv_path, bands, batch_size, workers)
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    scene_score_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    scene_label_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    scenes: dict[str, dict[str, int]] = defaultdict(_empty_scene)
    total_positive = 0
    total_valid = 0
    processed = 0

    for images, masks, valid, scene_ids_raw, *_rest in tqdm(loader, desc=f"{split_name} candidate PGSS", unit="batch"):
        scene_ids = list(scene_ids_raw)
        images = images.to(device, non_blocking=True)
        truth = (masks > 0).to(device, non_blocking=True).bool()
        valid = valid.to(device, non_blocking=True).bool()
        gate_logits = gate(prepare_gate_input(images, valid))
        pgss_logits = pgss(images)
        candidate_pixels = valid & (gate_logits >= float(protocol["gate_logit_threshold"]))
        selected, eligible, core, _unique_halo = build_tile_masks(
            candidate_pixels, valid, int(protocol["tile_size"]), int(protocol["halo"])
        )
        candidate_logits = candidate_core_logits(
            candidate_model,
            images,
            selected,
            int(protocol["tile_size"]),
            int(protocol["halo"]),
        )
        full_prediction = pgss_logits >= float(protocol["pgss_logit_threshold"])
        gate_prediction = full_prediction & core
        full_counts = prediction_count_tensors(full_prediction, truth, valid)
        gate_counts = prediction_count_tensors(gate_prediction, truth, valid)
        batch_candidate_valid = core & valid
        candidate_values = candidate_logits[batch_candidate_valid].float().detach().cpu().numpy()
        candidate_labels = truth[batch_candidate_valid].detach().cpu().numpy().astype(bool, copy=False)
        if candidate_values.size:
            score_chunks.append(candidate_values)
            label_chunks.append(candidate_labels)

        for batch_index, scene_id in enumerate(scene_ids):
            scene = scenes[scene_id]
            scene["patches"] += 1
            scene["valid_pixels"] += int(torch.count_nonzero(valid[batch_index]).item())
            scene["positive_pixels"] += int(torch.count_nonzero(truth[batch_index] & valid[batch_index]).item())
            scene["candidate_valid_pixels"] += int(torch.count_nonzero(batch_candidate_valid[batch_index]).item())
            scene["candidate_positive_pixels"] += int(torch.count_nonzero(batch_candidate_valid[batch_index] & truth[batch_index]).item())
            scene["candidate_tiles"] += int(torch.count_nonzero(selected[batch_index]).item())
            scene["eligible_tiles"] += int(torch.count_nonzero(eligible[batch_index]).item())
            for prefix, counts in (("full_pgss", full_counts), ("gate_pgss", gate_counts)):
                for key in COUNT_KEYS:
                    scene[f"{prefix}_{key}"] += int(counts[key][batch_index].item())
            scene_mask = batch_candidate_valid[batch_index, 0]
            scene_scores = candidate_logits[batch_index, 0][scene_mask].float().detach().cpu().numpy()
            scene_labels = truth[batch_index, 0][scene_mask].detach().cpu().numpy().astype(bool, copy=False)
            if scene_scores.size:
                scene_score_chunks[scene_id].append(scene_scores)
                scene_label_chunks[scene_id].append(scene_labels)
        total_positive += int(torch.count_nonzero(truth & valid).item())
        total_valid += int(torch.count_nonzero(valid).item())
        processed += int(images.shape[0])

    if not score_chunks:
        raise RuntimeError(f"No selected valid candidate pixels in {split_name}")
    all_scores = np.concatenate(score_chunks).astype(np.float32, copy=False)
    all_labels = np.concatenate(label_chunks).astype(bool, copy=False)
    exact_best, pr_auc = exact_candidate_threshold(all_scores, all_labels, total_positive, total_valid)
    best = (
        metrics_for_candidate_threshold(
            all_scores,
            all_labels,
            total_positive,
            total_valid,
            fixed_candidate_logit_threshold,
        )
        if fixed_candidate_logit_threshold is not None
        else exact_best
    )

    full_counts_aggregate = {key: sum(scene[f"full_pgss_{key}"] for scene in scenes.values()) for key in COUNT_KEYS}
    gate_counts_aggregate = {key: sum(scene[f"gate_pgss_{key}"] for scene in scenes.values()) for key in COUNT_KEYS}
    candidate_counts_aggregate = {key: 0 for key in COUNT_KEYS}
    scene_rows: list[dict[str, Any]] = []
    for scene_id in sorted(scenes):
        scene = scenes[scene_id]
        if scene_score_chunks[scene_id]:
            scene_scores = np.concatenate(scene_score_chunks[scene_id]).astype(np.float32, copy=False)
            scene_labels = np.concatenate(scene_label_chunks[scene_id]).astype(bool, copy=False)
            candidate_pred = scene_scores >= float(best["logit_threshold"])
            candidate_counts = {
                "tp": int(np.count_nonzero(candidate_pred & scene_labels)),
                "fp": int(np.count_nonzero(candidate_pred & ~scene_labels)),
                "fn": scene["positive_pixels"] - int(np.count_nonzero(candidate_pred & scene_labels)),
                "tn": scene["valid_pixels"] - scene["positive_pixels"] - int(np.count_nonzero(candidate_pred & ~scene_labels)),
            }
        else:
            candidate_counts = {"tp": 0, "fp": 0, "fn": scene["positive_pixels"], "tn": scene["valid_pixels"] - scene["positive_pixels"]}
        for key in COUNT_KEYS:
            candidate_counts_aggregate[key] += candidate_counts[key]
        full_metrics = metrics_from_counts({key: scene[f"full_pgss_{key}"] for key in COUNT_KEYS})
        gate_metrics = metrics_from_counts({key: scene[f"gate_pgss_{key}"] for key in COUNT_KEYS})
        candidate_metrics = metrics_from_counts(candidate_counts)
        scene_rows.append(
            {
                "dataset": split_name,
                "protocol": protocol["name"],
                "scene_id": scene_id,
                **scene,
                "candidate_pixel_ratio": scene["candidate_valid_pixels"] / scene["valid_pixels"] if scene["valid_pixels"] else 0.0,
                "candidate_positive_recall": scene["candidate_positive_pixels"] / scene["positive_pixels"] if scene["positive_pixels"] else float("nan"),
                "candidate_tile_ratio": scene["candidate_tiles"] / scene["eligible_tiles"] if scene["eligible_tiles"] else 0.0,
                "full_pgss_precision": full_metrics["precision"],
                "full_pgss_recall": full_metrics["recall"],
                "full_pgss_f1": full_metrics["f1"],
                "gate_pgss_precision": gate_metrics["precision"],
                "gate_pgss_recall": gate_metrics["recall"],
                "gate_pgss_f1": gate_metrics["f1"],
                "candidate_precision": candidate_metrics["precision"],
                "candidate_recall": candidate_metrics["recall"],
                "candidate_f1": candidate_metrics["f1"],
                **{f"candidate_{key}": candidate_counts[key] for key in COUNT_KEYS},
            }
        )

    full_metrics = metrics_from_counts(full_counts_aggregate)
    gate_metrics = metrics_from_counts(gate_counts_aggregate)
    candidate_metrics = metrics_from_counts(candidate_counts_aggregate)
    return {
        "split": split_name,
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path.resolve()),
        "protocol": protocol,
        "processed_patches": processed,
        "valid_pixels": sum(scene["valid_pixels"] for scene in scenes.values()),
        "positive_pixels": total_positive,
        "candidate_scored_valid_pixels": int(all_scores.size),
        "candidate_scored_positive_pixels": int(np.count_nonzero(all_labels)),
        "candidate_threshold_selection": "exact float32 logit scores inside Gate-selected core tiles; FN outside candidates included",
        "candidate_threshold_selection_is_fixed": fixed_candidate_logit_threshold is not None,
        "candidate_exact_oracle_threshold": exact_best,
        "candidate_threshold": best,
        "candidate_pr_auc": pr_auc,
        "full_pgss": {**full_counts_aggregate, **full_metrics},
        "gate_original_pgss": {**gate_counts_aggregate, **gate_metrics},
        "gate_candidate_pgss": {**candidate_counts_aggregate, **candidate_metrics},
        "scene_rows": scene_rows,
        "test_used": False,
    }
