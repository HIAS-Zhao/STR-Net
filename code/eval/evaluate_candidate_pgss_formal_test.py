#!/usr/bin/env python3
"""Formal Candidate-PGSS Test evaluation and verifier Recall curve.

The script deliberately lives beside, but does not overwrite, the original
Val-only Candidate-PGSS evaluator. It freezes Gate and verifier thresholds on
Val splits, then evaluates Test exactly once per frozen protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# code/ root on sys.path so the core/models/train/eval packages resolve
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.audit_spectral_mlp_gate import (
    build_tile_masks,
    load_gate,
    load_pgss,
    make_audit_loader,
)
from core.candidate_pgss_eval import candidate_core_logits, load_candidate_model
from core.pgss_common import metrics_from_counts, sha256_file, write_json
from models.spectral_mlp_gate import prepare_gate_input


COUNT_KEYS = ("tp", "fp", "fn", "tn")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def probability_from_logit(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


def metric_counts(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    prediction = prediction.astype(bool, copy=False) & valid
    truth = truth.astype(bool, copy=False) & valid
    return {
        "tp": int(np.count_nonzero(prediction & truth)),
        "fp": int(np.count_nonzero(prediction & ~truth)),
        "fn": int(np.count_nonzero(~prediction & truth & valid)),
        "tn": int(np.count_nonzero(~prediction & ~truth & valid)),
    }


def empty_scene() -> dict[str, int]:
    return {
        "patches": 0,
        "valid_pixels": 0,
        "positive_pixels": 0,
        "candidate_valid_pixels": 0,
        "candidate_positive_pixels": 0,
        "candidate_tiles": 0,
        "eligible_tiles": 0,
        "base_tile_area_pixels": 0,
        "base_tile_valid_pixels": 0,
        "tile_covered_positive_pixels": 0,
        "halo_repeated_input_pixels": 0,
        "halo_unique_input_pixels": 0,
        "all_patch_pixels": 0,
        "full_tp": 0,
        "full_fp": 0,
        "full_fn": 0,
        "full_tn": 0,
        "gate_tp": 0,
        "gate_fp": 0,
        "gate_fn": 0,
        "gate_tn": 0,
    }


def exact_average_precision(scores: np.ndarray, labels: np.ndarray, total_positive: int) -> float:
    """Exact AP with positives outside candidate tiles retained as FN."""
    if scores.size == 0 or total_positive <= 0:
        return float("nan")
    order = np.argsort(scores.astype(np.float32, copy=False), kind="stable")[::-1]
    sorted_labels = labels.astype(bool, copy=False)[order]
    sorted_scores = scores.astype(np.float32, copy=False)[order]
    boundaries = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    tp = np.cumsum(sorted_labels, dtype=np.int64)[boundaries]
    fp = np.cumsum(~sorted_labels, dtype=np.int64)[boundaries]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    recall = tp.astype(np.float64) / float(total_positive)
    return float(np.sum(precision * np.diff(np.r_[0.0, recall])))


def exact_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    total_positive: int,
    total_valid: int,
) -> dict[str, Any]:
    """Choose the exact score threshold with maximum valid-pixel F1."""
    if scores.size == 0 or not np.any(labels):
        raise RuntimeError("No candidate score contains a positive pixel")
    order = np.argsort(scores.astype(np.float32, copy=False), kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels.astype(bool, copy=False)[order]
    boundaries = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    tp = np.cumsum(sorted_labels, dtype=np.int64)[boundaries]
    fp = np.cumsum(~sorted_labels, dtype=np.int64)[boundaries]
    threshold = sorted_scores[boundaries]
    fn = int(total_positive) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    recall = tp.astype(np.float64) / float(max(1, total_positive))
    f1 = np.divide(2.0 * precision * recall, precision + recall,
                    out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = int(np.flatnonzero(f1 == np.max(f1))[0])
    return {
        "selection_rule": "maximum exact valid-pixel F1",
        "logit_threshold": float(threshold[best]),
        "probability_threshold": probability_from_logit(float(threshold[best])),
        "tp": int(tp[best]),
        "fp": int(fp[best]),
        "fn": int(fn[best]),
        "tn": int(total_valid - tp[best] - fp[best] - fn[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
        "predicted_positive_pixels": int(tp[best] + fp[best]),
    }


def exact_recall_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    total_positive: int,
    total_valid: int,
    constraint: float,
) -> dict[str, Any]:
    """Return the maximum verifier logit with Recall >= constraint."""
    positive_scores = np.sort(scores[labels].astype(np.float32, copy=False))
    if positive_scores.size == 0:
        raise RuntimeError("No positive candidate scores available")
    thresholds = np.unique(positive_scores)
    retained = positive_scores.size - np.searchsorted(positive_scores, thresholds, side="left")
    recall = retained.astype(np.float64) / float(total_positive)
    feasible = np.flatnonzero(recall >= float(constraint))
    if feasible.size == 0:
        return {
            "recall_constraint": float(constraint),
            "status": "NO-GO",
            "selection_rule": "maximum exact logit with overall Recall constraint",
        }
    index = int(feasible[-1])
    threshold = float(thresholds[index])
    prediction = scores >= threshold
    tp = int(np.count_nonzero(prediction & labels))
    fp = int(np.count_nonzero(prediction & ~labels))
    fn = int(total_positive - tp)
    tn = int(total_valid - tp - fp - fn)
    metrics = metrics_from_counts({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return {
        "recall_constraint": float(constraint),
        "status": "feasible",
        "selection_rule": "maximum exact logit with overall Recall constraint",
        "logit_threshold": threshold,
        "probability_threshold": probability_from_logit(threshold),
        "positive_pixels": int(total_positive),
        "retained_positive_pixels": tp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": metrics["precision"],
        "pixel_recall": metrics["recall"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "threshold_index": index,
        "candidate_pixel_ratio": float((tp + fp) / max(1, total_valid)),
    }


def candidate_metrics(
    scene_scores: list[np.ndarray],
    scene_labels: list[np.ndarray],
    scene_stats: dict[str, dict[str, int]],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    aggregate = {key: 0 for key in COUNT_KEYS}
    rows: dict[str, dict[str, Any]] = {}
    for scene_id in sorted(scene_stats):
        scores = np.concatenate(scene_scores[scene_id]).astype(np.float32, copy=False) if scene_scores[scene_id] else np.empty(0, np.float32)
        labels = np.concatenate(scene_labels[scene_id]).astype(bool, copy=False) if scene_labels[scene_id] else np.empty(0, bool)
        pred = scores >= float(threshold)
        positive = int(scene_stats[scene_id]["positive_pixels"])
        valid = int(scene_stats[scene_id]["valid_pixels"])
        tp = int(np.count_nonzero(pred & labels))
        fp = int(np.count_nonzero(pred & ~labels))
        counts = {"tp": tp, "fp": fp, "fn": positive - tp, "tn": valid - tp - fp - (positive - tp)}
        for key in COUNT_KEYS:
            aggregate[key] += counts[key]
        metrics = metrics_from_counts(counts)
        rows[scene_id] = {
            **counts,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
    metrics = metrics_from_counts(aggregate)
    return {**aggregate, **metrics, "predicted_positive_pixels": aggregate["tp"] + aggregate["fp"]}, rows


def event_metrics(records: dict[str, dict[str, list[np.ndarray]]], threshold: float) -> dict[str, Any]:
    total = tile = final = 0
    per_scene: dict[str, dict[str, int]] = {}
    missed: list[str] = []
    for scene_id in sorted(records):
        if not records[scene_id]["coords"]:
            continue
        coords = np.concatenate(records[scene_id]["coords"], axis=0)
        pixel_flags = np.concatenate(records[scene_id]["pixel_gate"], axis=0)
        tile_flags = np.concatenate(records[scene_id]["tile"], axis=0)
        scores = np.concatenate(records[scene_id]["scores"], axis=0)
        unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
        merged_pixel = np.zeros(len(unique_coords), dtype=bool)
        merged_tile = np.zeros(len(unique_coords), dtype=bool)
        merged_final = np.zeros(len(unique_coords), dtype=bool)
        np.logical_or.at(merged_pixel, inverse, pixel_flags)
        np.logical_or.at(merged_tile, inverse, tile_flags)
        np.logical_or.at(merged_final, inverse, scores >= float(threshold))
        # Importing the audited implementation avoids a second connectivity implementation.
        from core.audit_spectral_mlp_gate import SparseEventAccumulator
        components = SparseEventAccumulator._components(unique_coords)
        scene_total = scene_tile = scene_final = 0
        for indices in components:
            scene_total += 1
            if bool(np.any(merged_tile[indices])):
                scene_tile += 1
            if bool(np.any(merged_final[indices])):
                scene_final += 1
            if not bool(np.any(merged_final[indices])):
                anchor = unique_coords[indices].min(axis=0)
                missed.append(f"{scene_id}:event@{int(anchor[0])},{int(anchor[1])}:pixels={len(indices)}")
        total += scene_total
        tile += scene_tile
        final += scene_final
        per_scene[scene_id] = {
            "total_events": scene_total,
            "tile_recalled_events": scene_tile,
            "final_recalled_events": scene_final,
            "tile_event_recall": scene_tile / scene_total if scene_total else float("nan"),
            "final_event_recall": scene_final / scene_total if scene_total else float("nan"),
        }
    return {
        "total_events": total,
        "tile_recalled_events": tile,
        "final_recalled_events": final,
        "tile_event_recall": tile / total if total else float("nan"),
        "final_event_recall": final / total if total else float("nan"),
        "missed_final_events": missed,
        "scene_event_metrics": per_scene,
    }


@torch.inference_mode()
def collect_split(
    gate: torch.nn.Module,
    candidate: torch.nn.Module,
    pgss: torch.nn.Module,
    csv_path: Path,
    split_name: str,
    gate_logit_threshold: float,
    pgss_logit_threshold: float,
    bands: tuple[int, ...],
    batch_size: int,
    workers: int,
    tile_size: int,
    halo: int,
    tile_batch_size: int,
    device: torch.device,
    compute_pr_auc: bool = True,
) -> dict[str, Any]:
    # This helper is also called during training for Val selection.  Enforce
    # inference mode here so BatchNorm statistics are never updated by Val.
    gate.eval()
    candidate.eval()
    pgss.eval()
    _, loader = make_audit_loader(csv_path, bands, batch_size, workers)
    scenes: dict[str, dict[str, int]] = defaultdict(empty_scene)
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    scene_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    scene_labels: dict[str, list[np.ndarray]] = defaultdict(list)
    event_records: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"coords": [], "pixel_gate": [], "tile": [], "scores": []}
    )
    total_valid = total_positive = processed = 0
    started = time.perf_counter()
    for batch in tqdm(loader, desc=f"{split_name} Candidate-PGSS", unit="batch"):
        images, masks, valid, scene_ids_raw, _indices, offsets_y, offsets_x, _coordinate_sources = batch
        scene_ids = list(scene_ids_raw)
        images = images.to(device, non_blocking=True)
        truth = (masks > 0).to(device, non_blocking=True).bool()
        valid = valid.to(device, non_blocking=True).bool()
        gate_logits = gate(prepare_gate_input(images, valid))
        pgss_logits = pgss(images)
        candidate_pixels = valid & (gate_logits >= float(gate_logit_threshold))
        selected, eligible, core, unique_halo = build_tile_masks(candidate_pixels, valid, tile_size, halo)
        candidate_logits = candidate_core_logits(candidate, images, selected, tile_size, halo, tile_batch_size)
        full_prediction = pgss_logits >= float(pgss_logit_threshold)
        gate_prediction = full_prediction & core
        full_counts = metric_tensors(full_prediction, truth, valid)
        gate_counts = metric_tensors(gate_prediction, truth, valid)
        core_valid = core & valid
        candidate_values = candidate_logits[core_valid].float().detach().cpu().numpy().astype(np.float32, copy=False)
        candidate_truth = truth[core_valid].detach().cpu().numpy().astype(bool, copy=False)
        if candidate_values.size:
            score_chunks.append(candidate_values)
            label_chunks.append(candidate_truth)
        for batch_index, scene_id in enumerate(scene_ids):
            scene = scenes[scene_id]
            scene["patches"] += 1
            scene["valid_pixels"] += int(torch.count_nonzero(valid[batch_index]).item())
            scene["positive_pixels"] += int(torch.count_nonzero(truth[batch_index] & valid[batch_index]).item())
            scene["candidate_valid_pixels"] += int(torch.count_nonzero(core_valid[batch_index]).item())
            scene["candidate_positive_pixels"] += int(torch.count_nonzero(core_valid[batch_index] & truth[batch_index]).item())
            scene["candidate_tiles"] += int(torch.count_nonzero(selected[batch_index]).item())
            scene["eligible_tiles"] += int(torch.count_nonzero(eligible[batch_index]).item())
            scene["base_tile_area_pixels"] += int(torch.count_nonzero(core[batch_index]).item())
            scene["base_tile_valid_pixels"] += int(torch.count_nonzero(core_valid[batch_index]).item())
            scene["tile_covered_positive_pixels"] += int(torch.count_nonzero(core_valid[batch_index] & truth[batch_index]).item())
            scene["halo_repeated_input_pixels"] += halo_weight_sum(int(images.shape[-2]), int(images.shape[-1]), selected[batch_index], tile_size, halo)
            scene["halo_unique_input_pixels"] += int(torch.count_nonzero(unique_halo[batch_index]).item())
            scene["all_patch_pixels"] += int(images.shape[-2] * images.shape[-1])
            for key in COUNT_KEYS:
                scene[f"full_{key}"] += int(full_counts[key][batch_index].item())
                scene[f"gate_{key}"] += int(gate_counts[key][batch_index].item())
            mask = core_valid[batch_index, 0]
            scene_scores[scene_id].append(candidate_logits[batch_index, 0][mask].float().detach().cpu().numpy().astype(np.float32, copy=False))
            scene_labels[scene_id].append(truth[batch_index, 0][mask].detach().cpu().numpy().astype(bool, copy=False))
            pos = torch.nonzero((truth[batch_index] & valid[batch_index])[0], as_tuple=False).cpu().numpy().astype(np.int64, copy=False)
            if pos.size:
                pos_t = torch.from_numpy(pos).to(device)
                event_records[scene_id]["coords"].append(pos + np.asarray([int(offsets_y[batch_index]), int(offsets_x[batch_index])], dtype=np.int64))
                event_records[scene_id]["pixel_gate"].append(candidate_pixels[batch_index, 0][pos_t[:, 0], pos_t[:, 1]].detach().cpu().numpy().astype(bool, copy=False))
                event_records[scene_id]["tile"].append(core[batch_index, 0][pos_t[:, 0], pos_t[:, 1]].detach().cpu().numpy().astype(bool, copy=False))
                event_records[scene_id]["scores"].append(candidate_logits[batch_index, 0][pos_t[:, 0], pos_t[:, 1]].detach().cpu().numpy().astype(np.float32, copy=False))
        total_valid += int(torch.count_nonzero(valid).item())
        total_positive += int(torch.count_nonzero(truth & valid).item())
        processed += int(images.shape[0])
    if not score_chunks:
        raise RuntimeError(f"No selected core pixels were produced for {split_name}")
    scores = np.concatenate(score_chunks).astype(np.float32, copy=False)
    labels = np.concatenate(label_chunks).astype(bool, copy=False)
    return {
        "split": split_name,
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path.resolve()),
        "processed_patches": processed,
        "valid_pixels": total_valid,
        "positive_pixels": total_positive,
        "candidate_scored_valid_pixels": int(scores.size),
        "candidate_scored_positive_pixels": int(np.count_nonzero(labels)),
        "scores": scores,
        "labels": labels,
        "scene_scores": scene_scores,
        "scene_labels": scene_labels,
        "scenes": scenes,
        "event_records": event_records,
        "gate_logit_threshold": gate_logit_threshold,
        "pgss_logit_threshold": pgss_logit_threshold,
        "tile_size": tile_size,
        "halo": halo,
        "pr_auc": exact_average_precision(scores, labels, total_positive) if compute_pr_auc else float("nan"),
        "pr_auc_computed": bool(compute_pr_auc),
        "wall_time_s": time.perf_counter() - started,
    }


def metric_tensors(prediction: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    prediction = prediction.bool() & valid.bool()
    truth = truth.bool() & valid.bool()
    valid = valid.bool()
    return {
        "tp": torch.count_nonzero(prediction & truth, dim=(1, 2, 3)),
        "fp": torch.count_nonzero(prediction & ~truth & valid, dim=(1, 2, 3)),
        "fn": torch.count_nonzero(~prediction & truth & valid, dim=(1, 2, 3)),
        "tn": torch.count_nonzero(~prediction & ~truth & valid, dim=(1, 2, 3)),
    }


def halo_weight_sum(height: int, width: int, selected: torch.Tensor, tile_size: int, halo: int) -> int:
    rows, cols = selected.shape[-2:]
    total = 0
    for row in range(int(rows)):
        y0, y1 = row * tile_size, min(height, (row + 1) * tile_size)
        expanded_h = min(height, y1 + halo) - max(0, y0 - halo)
        for col in range(int(cols)):
            if bool(selected[0, row, col]):
                x0, x1 = col * tile_size, min(width, (col + 1) * tile_size)
                expanded_w = min(width, x1 + halo) - max(0, x0 - halo)
                total += expanded_h * expanded_w
    return int(total)


def summarize_protocol(
    collected: dict[str, Any],
    verifier_threshold: dict[str, Any],
    protocol_name: str,
    compute_events: bool = True,
) -> dict[str, Any]:
    threshold = float(verifier_threshold["logit_threshold"])
    metrics, scene_metric_map = candidate_metrics(
        collected["scene_scores"], collected["scene_labels"], collected["scenes"], threshold
    )
    if compute_events:
        event = event_metrics(collected["event_records"], threshold)
    else:
        event = {
            "total_events": 0,
            "tile_recalled_events": 0,
            "final_recalled_events": 0,
            "tile_event_recall": float("nan"),
            "final_event_recall": float("nan"),
            "missed_final_events": [],
            "scene_event_metrics": {},
            "status": "not_computed_for_threshold_curve",
        }
    scene_rows = []
    for scene_id in sorted(collected["scenes"]):
        stats = collected["scenes"][scene_id]
        candidate = scene_metric_map[scene_id]
        full = metrics_from_counts({key: stats[f"full_{key}"] for key in COUNT_KEYS})
        gate = metrics_from_counts({key: stats[f"gate_{key}"] for key in COUNT_KEYS})
        scene_rows.append({
            "dataset": collected["split"],
            "protocol": protocol_name,
            "gate_logit_threshold": collected["gate_logit_threshold"],
            "verifier_logit_threshold": threshold,
            "verifier_probability_threshold": probability_from_logit(threshold),
            "pgss_logit_threshold": collected["pgss_logit_threshold"],
            "tile_size": collected["tile_size"],
            "halo": collected["halo"],
            "scene_id": scene_id,
            **stats,
            "candidate_pixel_ratio": stats["candidate_valid_pixels"] / stats["valid_pixels"] if stats["valid_pixels"] else 0.0,
            "candidate_tile_ratio": stats["candidate_tiles"] / stats["eligible_tiles"] if stats["eligible_tiles"] else 0.0,
            "tile_fire_recall": stats["tile_covered_positive_pixels"] / stats["positive_pixels"] if stats["positive_pixels"] else float("nan"),
            "base_tile_valid_coverage": stats["base_tile_valid_pixels"] / stats["valid_pixels"] if stats["valid_pixels"] else 0.0,
            "halo_repeated_input_ratio": stats["halo_repeated_input_pixels"] / stats["all_patch_pixels"] if stats["all_patch_pixels"] else 0.0,
            "halo_unique_input_ratio": stats["halo_unique_input_pixels"] / stats["all_patch_pixels"] if stats["all_patch_pixels"] else 0.0,
            "full_precision": full["precision"], "full_recall": full["recall"], "full_f1": full["f1"],
            "gate_precision": gate["precision"], "gate_recall": gate["recall"], "gate_f1": gate["f1"],
            "verifier_tp": candidate["tp"], "verifier_fp": candidate["fp"],
            "verifier_fn": candidate["fn"], "verifier_tn": candidate["tn"],
            "verifier_precision": candidate["precision"], "verifier_recall": candidate["recall"], "verifier_f1": candidate["f1"],
            "verifier_added_fn_vs_full": candidate["fn"] - stats["full_fn"],
            "verifier_removed_fp_vs_full": stats["full_fp"] - candidate["fp"],
            **event["scene_event_metrics"].get(scene_id, {}),
        })
    positive_scenes = [row for row in scene_rows if row["positive_pixels"] > 0]
    worst_scene = min(positive_scenes, key=lambda row: row["verifier_recall"]) if positive_scenes else None
    full_counts = {key: sum(int(scene[f"full_{key}"]) for scene in collected["scenes"].values()) for key in COUNT_KEYS}
    gate_counts = {key: sum(int(scene[f"gate_{key}"]) for scene in collected["scenes"].values()) for key in COUNT_KEYS}
    full_metrics = {**full_counts, **metrics_from_counts(full_counts)}
    gate_metrics = {**gate_counts, **metrics_from_counts(gate_counts)}
    return {
        "dataset": collected["split"],
        "protocol": protocol_name,
        "gate_logit_threshold": collected["gate_logit_threshold"],
        "verifier_threshold": verifier_threshold,
        "pgss_logit_threshold": collected["pgss_logit_threshold"],
        "pr_auc": collected["pr_auc"],
        "metrics": metrics,
        "full_pgss_metrics": full_metrics,
        "gate_original_pgss_metrics": gate_metrics,
        "f1_delta_vs_full_pgss_pp": (metrics["f1"] - full_metrics["f1"]) * 100.0,
        "recall_delta_vs_full_pgss_pp": (metrics["recall"] - full_metrics["recall"]) * 100.0,
        "fp_reduction_vs_full_pgss": (
            (full_metrics["fp"] - metrics["fp"]) / full_metrics["fp"] if full_metrics["fp"] else float("nan")
        ),
        "event_metrics": event,
        "candidate_tile_coverage": aggregate_tile_coverage(collected),
        "scene_rows": scene_rows,
        "worst_positive_scene": worst_scene["scene_id"] if worst_scene else None,
        "worst_positive_scene_recall": worst_scene["verifier_recall"] if worst_scene else float("nan"),
        "top_fp_scenes": [
            {"scene_id": row["scene_id"], "fp": row["verifier_fp"], "fn": row["verifier_fn"], "f1": row["verifier_f1"]}
            for row in sorted(scene_rows, key=lambda row: row["verifier_fp"], reverse=True)[:3]
        ],
    }


def aggregate_tile_coverage(collected: dict[str, Any]) -> dict[str, Any]:
    scenes = collected["scenes"]
    keys = ("valid_pixels", "positive_pixels", "candidate_valid_pixels", "candidate_positive_pixels", "candidate_tiles", "eligible_tiles", "base_tile_area_pixels", "base_tile_valid_pixels", "tile_covered_positive_pixels", "halo_repeated_input_pixels", "halo_unique_input_pixels", "all_patch_pixels")
    aggregate = {key: sum(int(scene[key]) for scene in scenes.values()) for key in keys}
    return {
        **aggregate,
        "candidate_pixel_ratio": aggregate["candidate_valid_pixels"] / aggregate["valid_pixels"] if aggregate["valid_pixels"] else 0.0,
        "candidate_tile_ratio": aggregate["candidate_tiles"] / aggregate["eligible_tiles"] if aggregate["eligible_tiles"] else 0.0,
        "tile_fire_recall": aggregate["tile_covered_positive_pixels"] / aggregate["positive_pixels"] if aggregate["positive_pixels"] else 0.0,
        "base_tile_valid_coverage": aggregate["base_tile_valid_pixels"] / aggregate["valid_pixels"] if aggregate["valid_pixels"] else 0.0,
        "halo_repeated_input_ratio": aggregate["halo_repeated_input_pixels"] / aggregate["all_patch_pixels"] if aggregate["all_patch_pixels"] else 0.0,
        "halo_unique_input_ratio": aggregate["halo_unique_input_pixels"] / aggregate["all_patch_pixels"] if aggregate["all_patch_pixels"] else 0.0,
    }


def curve_row(collected: dict[str, Any], selection: dict[str, Any], test_result: dict[str, Any] | None, split_role: str) -> dict[str, Any]:
    row = {
        "selection_split": collected["split"],
        "evaluation_split": split_role,
        "recall_constraint": selection.get("recall_constraint", "formal_f1"),
        "status": selection["status"] if "status" in selection else "feasible",
        "verifier_logit_threshold": selection.get("logit_threshold", ""),
        "verifier_probability_threshold": selection.get("probability_threshold", ""),
        "val_pixel_recall": selection.get("pixel_recall", selection.get("recall", "")),
        "val_precision": selection.get("precision", ""),
        "val_f1": selection.get("f1", ""),
        "val_candidate_pixel_ratio": selection.get("candidate_pixel_ratio", ""),
    }
    if test_result is not None:
        row.update({
            "test_gate_logit_threshold": test_result["gate_logit_threshold"],
            "test_verifier_logit_threshold": test_result["verifier_threshold"]["logit_threshold"],
            "test_pixel_recall": test_result["metrics"]["recall"],
            "test_event_recall": test_result["event_metrics"]["final_event_recall"],
            "test_precision": test_result["metrics"]["precision"],
            "test_f1": test_result["metrics"]["f1"],
            "test_candidate_tile_ratio": test_result["candidate_tile_coverage"]["candidate_tile_ratio"],
            "test_halo_repeated_input_ratio": test_result["candidate_tile_coverage"]["halo_repeated_input_ratio"],
        })
    return row


def all_finite(values: list[dict[str, Any]]) -> bool:
    return all(math.isfinite(float(value)) for row in values for value in row.values() if isinstance(value, (float, int)))


def build_report(result: dict[str, Any], curve_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Candidate-PGSS Formal Test Evaluation",
        "",
        "## Protocol",
        "",
        "- Candidate verifier is the Gate-selected PGSS-W32 model trained from scratch on ActiveFire-Train candidate tiles.",
        "- Gate threshold is selected/frozen by split; verifier threshold is selected only on the corresponding Val split.",
        "- ActiveFire-Val -> ActiveFire-Test and ActiveFire-Val -> Land8Fire-Test zero-shot use the same Gate and verifier thresholds.",
        "- Land8Fire-Val -> Land8Fire-Test target-calibrated is reported separately and Test is never used for selection.",
        "- All metrics use valid pixels only; input is [B7,B6,B5]/65535 and invalid pixels are zeroed.",
        "- No power/energy measurement was performed in this accuracy-only run.",
        "",
        "## Model",
        "",
        f"- Candidate checkpoint: `{result['candidate_checkpoint']}`",
        f"- Parameters: **{result['parameter_info']['total_parameters']:,}**",
        f"- MACs/pixel: **{result['macs_per_pixel']:,}**",
        f"- Tile: **{result['tile_size']}x{result['tile_size']} core + {result['halo']} halo**",
        "",
        "## Formal Frozen Test Results",
        "",
        "| Dataset/protocol | Gate logit | Verifier logit | P | R | F1 | Delta F1 | FP reduction | PR-AUC | Event R | Tile ratio | Halo repeated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("activefire_test_source", "land8fire_test_zero_shot", "land8fire_test_target"):
        item = result[key]
        m, c, e = item["metrics"], item["candidate_tile_coverage"], item["event_metrics"]
        lines.append(
            f"| {item['dataset']} / {item['protocol']} | {item['gate_logit_threshold']:.9g} | {item['verifier_threshold']['logit_threshold']:.9g} | "
            f"{100*m['precision']:.4f}% | {100*m['recall']:.4f}% | {100*m['f1']:.4f}% | {item['f1_delta_vs_full_pgss_pp']:+.4f} pp | "
            f"{100*item['fp_reduction_vs_full_pgss']:.4f}% | {item['pr_auc']:.6f} | {100*e['final_event_recall']:.4f}% | "
            f"{100*c['candidate_tile_ratio']:.4f}% | {100*c['halo_repeated_input_ratio']:.4f}% |"
        )
    lines.extend([
        "",
        "## Verifier Recall Curve",
        "",
        "The curve thresholds are selected on ActiveFire-Val. The formal F1 threshold is kept separately; these rows are diagnostic operating points for deciding whether threshold tuning can recover ActiveFire Recall.",
        "",
        "| Recall constraint | Val logit | Val P/R/F1 | Active Test P/R/F1 | Land Zero-shot P/R/F1 | Land Zero tile ratio |",
        "|---:|---:|---|---|---|---:|",
    ])
    for row in curve_rows:
        if row.get("selection_split") != "ActiveFire-Val" or row.get("evaluation_split") != "ActiveFire-Test":
            continue
        land = next((x for x in curve_rows if x.get("recall_constraint") == row.get("recall_constraint") and x.get("evaluation_split") == "Land8Fire-Test-zero-shot"), None)
        land_metrics = (
            f"{pct(land.get('test_precision'))}/{pct(land.get('test_pixel_recall'))}/{pct(land.get('test_f1'))}"
            if land else "N/A"
        )
        land_tiles = pct(land.get("test_candidate_tile_ratio")) if land else "N/A"
        lines.append(
            f"| {row['recall_constraint']} | {row['verifier_logit_threshold'] if row['status']=='feasible' else 'NO-GO'} | "
            f"{pct(row.get('val_precision'))}/{pct(row.get('val_pixel_recall'))}/{pct(row.get('val_f1'))} | "
            f"{pct(row.get('test_precision'))}/{pct(row.get('test_pixel_recall'))}/{pct(row.get('test_f1'))} | "
            f"{land_metrics} | {land_tiles} |"
        )
    lines.extend(["", "## Hard Scenes", ""])
    for key in ("activefire_test_source", "land8fire_test_zero_shot", "land8fire_test_target"):
        item = result[key]
        lines.append(f"### {item['dataset']} / {item['protocol']}")
        lines.append("")
        lines.append("| Scene | FP | FN | F1 |")
        lines.append("|---|---:|---:|---:|")
        for scene in item["top_fp_scenes"]:
            lines.append(f"| {scene['scene_id']} | {scene['fp']:,} | {scene['fn']:,} | {100*scene['f1']:.4f}% |")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "- The formal Candidate-PGSS Test conclusion must use the three frozen rows above, not the Land8Fire-Val development result.",
        "- The Recall curve determines whether threshold adjustment alone is sufficient. If the ActiveFire Recall-constrained rows still lose the desired F1 or do not preserve the Land8Fire precision benefit, the next intervention should be distillation rather than another threshold-only change.",
        "",
        "## Output Files",
        "",
    ])
    for key, value in result["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def pct(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    return f"{100*float(value):.4f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/eval/configs/formal_eval_765.json")
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--tile-batch-size", type=int, default=0)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if torch.cuda.is_available() and not args.device:
        device = torch.device(config["runtime"]["requested_device"])
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config["runtime"]["fallback_device"])
        print("WARNING: CUDA unavailable; running accuracy evaluation on CPU. No energy ranking is produced.")
    bands = tuple(int(value) for value in config["input"]["bands"])
    tile_size = int(config["tile"]["size"])
    halo = int(config["tile"]["halo"])
    batch_size = int(args.batch_size or config["runtime"]["batch_size"])
    workers = int(config["runtime"]["workers"] if args.workers < 0 else args.workers)
    tile_batch_size = int(args.tile_batch_size or config["runtime"]["candidate_tile_batch_size"])
    gate, gate_checkpoint = load_gate(Path(config["gate"]["checkpoint"]).resolve(), device)
    candidate, candidate_checkpoint = load_candidate_model(Path(config["models"]["candidate_checkpoint"]).resolve(), device)
    pgss, pgss_checkpoint = load_pgss(Path(config["models"]["full_pgss_checkpoint"]).resolve(), device)
    for model in (gate, candidate, pgss):
        model.eval()
    active_gate = float(config["gate"]["activefire_val_logit_threshold"])
    # Some derived formal configs store the frozen Land gate under the
    # canonical *_frozen key. It is the same ActiveFire-Val-selected value;
    # accept both spellings so a derived config cannot stop evaluation.
    land_gate = float(
        config["gate"].get(
            "land8fire_val_logit_threshold",
            config["gate"]["land8fire_frozen_logit_threshold"],
        )
    )
    active_pgss = float(config["models"]["full_pgss_activefire_val_logit_threshold"])
    land_pgss = float(config["models"]["full_pgss_land8fire_val_logit_threshold"])
    curve_constraints = [float(x) for x in config["threshold_curve"]["activefire_recall_constraints"]]

    active_val = collect_split(gate, candidate, pgss, Path(config["data"]["activefire_val"]), "ActiveFire-Val", active_gate, active_pgss, bands, batch_size, workers, tile_size, halo, tile_batch_size, device)
    land_val = collect_split(gate, candidate, pgss, Path(config["data"]["land8fire_val"]), "Land8Fire-Val", land_gate, land_pgss, bands, batch_size, workers, tile_size, halo, tile_batch_size, device)
    active_f1 = exact_f1_threshold(active_val["scores"], active_val["labels"], active_val["positive_pixels"], active_val["valid_pixels"])
    land_f1 = exact_f1_threshold(land_val["scores"], land_val["labels"], land_val["positive_pixels"], land_val["valid_pixels"])
    active_curve = [exact_recall_threshold(active_val["scores"], active_val["labels"], active_val["positive_pixels"], active_val["valid_pixels"], c) for c in curve_constraints]
    active_test = collect_split(gate, candidate, pgss, Path(config["data"]["activefire_test"]), "ActiveFire-Test", active_gate, active_pgss, bands, batch_size, workers, tile_size, halo, tile_batch_size, device)
    land_test_zero = collect_split(
        gate, candidate, pgss, Path(config["data"]["land8fire_test"]),
        "Land8Fire-Test", active_gate, active_pgss, bands, batch_size, workers,
        tile_size, halo, tile_batch_size, device,
    )
    land_test_target = collect_split(
        gate, candidate, pgss, Path(config["data"]["land8fire_test"]),
        "Land8Fire-Test", land_gate, land_pgss, bands, batch_size, workers,
        tile_size, halo, tile_batch_size, device,
    )

    active_formal = summarize_protocol(active_test, active_f1, "source_frozen")
    land_zero = summarize_protocol(
        land_test_zero,
        active_f1,
        "zero_shot_source_frozen",
    )
    land_target = summarize_protocol(land_test_target, land_f1, "target_calibrated")
    result: dict[str, Any] = {
        "algorithm": "Gate-selected candidate-region PGSS-W32",
        "protocol": "formal Test freeze; Test never used for threshold selection",
        "device": str(device),
        "candidate_checkpoint": str(Path(config["models"]["candidate_checkpoint"]).resolve()),
        "candidate_checkpoint_sha256": sha256_file(Path(config["models"]["candidate_checkpoint"]).resolve()),
        "gate_checkpoint": str(Path(config["gate"]["checkpoint"]).resolve()),
        "gate_checkpoint_sha256": sha256_file(Path(config["gate"]["checkpoint"]).resolve()),
        "full_pgss_checkpoint": str(Path(config["models"]["full_pgss_checkpoint"]).resolve()),
        "full_pgss_checkpoint_sha256": sha256_file(Path(config["models"]["full_pgss_checkpoint"]).resolve()),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "parameter_info": candidate_checkpoint.get("parameter_info", {"total_parameters": 4401, "trainable_parameters": 4401}),
        "macs_per_pixel": candidate_checkpoint.get("macs_per_pixel", 4008),
        "bands": list(bands),
        "normalization": "/65535",
        "tile_size": tile_size,
        "halo": halo,
        "land8fire_val_development": {
            "activefire_source_gate": active_gate,
            "activefire_source_verifier_f1_selection": active_f1,
            "land8fire_target_gate": land_gate,
            "land8fire_target_verifier_f1_selection": land_f1,
        },
        "activefire_test_source": active_formal,
        "land8fire_test_zero_shot": land_zero,
        "land8fire_test_target": land_target,
        "outputs": config["outputs"],
        "runtime": {
            "device": str(device),
            "gpu_state": config["runtime"].get("gpu_state", "UNKNOWN"),
            "power_energy_policy": config["runtime"]["power_energy_policy"],
            "test_used_for_selection": False,
        },
    }

    output_dir = Path(config["outputs"]["run_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    difficult_scene_id = config["evaluation"]["difficult_land8fire_scene_id"]
    result["difficult_land8fire_scene"] = {
        "scene_id": difficult_scene_id,
        "zero_shot": next((row for row in land_zero["scene_rows"] if row["scene_id"] == difficult_scene_id), None),
        "target_calibrated": next((row for row in land_target["scene_rows"] if row["scene_id"] == difficult_scene_id), None),
    }
    # Persist the formal frozen-Test result before optional diagnostic curve work.
    write_json(Path(config["outputs"]["result_json"]), result)

    curve_rows: list[dict[str, Any]] = []
    for selection in active_curve:
        if selection.get("status") != "feasible":
            curve_rows.append(curve_row(active_val, selection, None, "ActiveFire-Test"))
            continue
        active_curve_test = summarize_protocol(
            active_test,
            selection,
            "source_frozen_recall_constraint",
            compute_events=False,
        )
        land_curve_test = summarize_protocol(
            land_test_zero,
            selection,
            "zero_shot_source_frozen_recall_constraint",
            compute_events=False,
        )
        curve_rows.append(curve_row(active_val, selection, active_curve_test, "ActiveFire-Test"))
        curve_rows.append(curve_row(active_val, selection, land_curve_test, "Land8Fire-Test-zero-shot"))
    curve_rows.append(curve_row(active_val, active_f1, active_formal, "ActiveFire-Test-formal_f1"))
    curve_rows.append(curve_row(land_val, land_f1, land_target, "Land8Fire-Test-target_calibrated"))

    scene_rows = active_formal["scene_rows"] + land_zero["scene_rows"] + land_target["scene_rows"]
    tile_rows = []
    for item in (active_formal, land_zero, land_target):
        row = {
            "dataset": item["dataset"],
            "protocol": item["protocol"],
            "gate_logit_threshold": item["gate_logit_threshold"],
            "verifier_logit_threshold": item["verifier_threshold"]["logit_threshold"],
            "verifier_probability_threshold": item["verifier_threshold"]["probability_threshold"],
            "pgss_logit_threshold": item["pgss_logit_threshold"],
            **item["candidate_tile_coverage"],
            "event_recall": item["event_metrics"]["final_event_recall"],
            "pr_auc": item["pr_auc"],
            "precision": item["metrics"]["precision"],
            "recall": item["metrics"]["recall"],
            "f1": item["metrics"]["f1"],
            "top_fp_scenes": json.dumps(item["top_fp_scenes"], ensure_ascii=False),
        }
        tile_rows.append(row)
    comparison_rows = []
    for item in (active_formal, land_zero, land_target):
        m = item["metrics"]
        comparison_rows.append({
            "dataset": item["dataset"], "protocol": item["protocol"],
            "gate_logit_threshold": item["gate_logit_threshold"],
            "verifier_logit_threshold": item["verifier_threshold"]["logit_threshold"],
            "verifier_probability_threshold": item["verifier_threshold"]["probability_threshold"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "pr_auc": item["pr_auc"],
            "event_recall": item["event_metrics"]["final_event_recall"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "candidate_tile_ratio": item["candidate_tile_coverage"]["candidate_tile_ratio"],
            "halo_repeated_input_ratio": item["candidate_tile_coverage"]["halo_repeated_input_ratio"],
            "full_pgss_f1": item["full_pgss_metrics"]["f1"],
            "full_pgss_fp": item["full_pgss_metrics"]["fp"],
            "f1_delta_vs_full_pgss_pp": item["f1_delta_vs_full_pgss_pp"],
            "recall_delta_vs_full_pgss_pp": item["recall_delta_vs_full_pgss_pp"],
            "fp_reduction_vs_full_pgss": item["fp_reduction_vs_full_pgss"],
            "worst_positive_scene": item["worst_positive_scene"],
            "worst_positive_scene_recall": item["worst_positive_scene_recall"],
            "top_fp_scenes": json.dumps(item["top_fp_scenes"], ensure_ascii=False),
        })

    write_json(Path(config["outputs"]["result_json"]), result)
    write_csv(Path(config["outputs"]["comparison_csv"]), comparison_rows)
    write_csv(Path(config["outputs"]["threshold_curve_csv"]), curve_rows)
    write_csv(Path(config["outputs"]["scene_metrics_csv"]), scene_rows)
    write_csv(Path(config["outputs"]["tile_coverage_csv"]), tile_rows)
    Path(config["outputs"]["report_md"]).write_text(build_report(result, curve_rows), encoding="utf-8")
    print(json.dumps({"result": config["outputs"]["result_json"], "report": config["outputs"]["report_md"], "device": str(device)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
