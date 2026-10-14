#!/usr/bin/env python3
"""Exact logit-threshold evaluation and score diagnostics for PGSS-Tiny V1.

The F1/AP search is exact for the float32 logits emitted by the model. It uses
all valid pixels and evaluates every distinct positive-pixel logit as a
candidate threshold. Negative quantiles are estimated with a high-resolution
streaming logit histogram so the full background tensor is never materialized.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# code/ root on sys.path so the core/models/train/eval packages resolve
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.pgss_common import FireSegmentationCsvDataset, metrics_from_counts, sha256_file, write_json
from models.pgss_model import PGSSTinyV1


def probability_from_logit(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


def probabilities_from_logits(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def parse_named_thresholds(items: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=LOGIT for --fixed-logit-threshold, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name or name == "oracle":
            raise ValueError("Threshold name must be non-empty and cannot be 'oracle'")
        result[name] = float(value)
    return result


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    bands = tuple(checkpoint.get("bands", (7, 6, 2)))
    model = PGSSTinyV1(
        in_channels=len(bands),
        width=int(checkpoint.get("width", 16)),
        head_width=int(checkpoint.get("head_width", 8)),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model, checkpoint, bands


def quantiles(values: np.ndarray, requested: tuple[float, ...]) -> dict[str, float]:
    if values.size == 0:
        return {f"p{100 * q:g}": float("nan") for q in requested}
    probs = probabilities_from_logits(np.quantile(values.astype(np.float64), requested))
    return {f"p{100 * q:g}": float(value) for q, value in zip(requested, probs)}


def histogram_quantiles(
    histogram: np.ndarray,
    minimum: float,
    maximum: float,
    requested: tuple[float, ...],
) -> dict[str, float]:
    total = int(histogram.sum())
    if total == 0:
        return {f"p{100 * q:g}": float("nan") for q in requested}
    cumulative = np.cumsum(histogram, dtype=np.int64)
    width = (maximum - minimum) / len(histogram)
    output = {}
    for q in requested:
        rank = max(1, int(math.ceil(q * total)))
        index = int(np.searchsorted(cumulative, rank, side="left"))
        logit = minimum + (index + 0.5) * width
        output[f"p{100 * q:g}"] = probability_from_logit(logit)
    return output


def exact_search(
    thresholds: np.ndarray,
    positive_logits: np.ndarray,
    fp_counts: np.ndarray,
) -> tuple[dict, float]:
    positive_sorted = np.sort(positive_logits.astype(np.float32, copy=False))
    total_positive = int(positive_sorted.size)
    tp = total_positive - np.searchsorted(positive_sorted, thresholds, side="left")
    fp = fp_counts.astype(np.int64, copy=False)
    fn = total_positive - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    recall = tp.astype(np.float64) / total_positive
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_f1 = float(np.max(f1))
    tied = np.flatnonzero(f1 == best_f1)
    best_index = int(tied[-1])  # Prefer the higher threshold when F1 is exactly tied.
    best_logit = float(thresholds[best_index])
    best = {
        "probability_threshold": probability_from_logit(best_logit),
        "logit_threshold": best_logit,
        "tp": int(tp[best_index]),
        "fp": int(fp[best_index]),
        "fn": int(fn[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": best_f1,
        "predicted_positive_pixels": int(tp[best_index] + fp[best_index]),
    }

    descending_precision = precision[::-1]
    descending_recall = recall[::-1]
    recall_increment = np.diff(np.concatenate(([0.0], descending_recall)))
    average_precision = float(np.sum(recall_increment * descending_precision))
    return best, average_precision


def counts_for_prediction(prediction: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> dict[str, int]:
    prediction = prediction.bool() & valid
    truth = truth.bool() & valid
    return {
        "tp": int(torch.count_nonzero(prediction & truth).item()),
        "fp": int(torch.count_nonzero(prediction & ~truth).item()),
        "fn": int(torch.count_nonzero(~prediction & truth & valid).item()),
        "tn": int(torch.count_nonzero(~prediction & ~truth & valid).item()),
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for exact PGSS evaluation")
    device = torch.device("cuda")
    model_path = Path(args.model).resolve()
    csv_path = Path(args.csv).resolve()
    model, checkpoint, bands = load_model(model_path, device)
    if model.training:
        raise AssertionError("Model must be in eval mode during inference")
    dataset = FireSegmentationCsvDataset(
        csv_path,
        bands=bands,
        max_patches=args.max_patches,
        return_scene_id=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )

    positive_chunks: list[np.ndarray] = []
    total_valid = total_positive = processed = nonfinite = 0
    minimum_logit = math.inf
    maximum_logit = -math.inf
    started = time.perf_counter()
    for images, masks, valid, _scene_ids in tqdm(loader, desc=f"{args.split_name} pass1", unit="batch"):
        images = images.to(device, non_blocking=True)
        masks = (masks > 0).to(device, non_blocking=True).bool()
        valid = valid.to(device, non_blocking=True).bool()
        logits = model(images)
        valid_logits = logits[valid]
        finite = torch.isfinite(valid_logits)
        nonfinite += int(torch.count_nonzero(~finite).item())
        if not bool(torch.all(finite)):
            raise RuntimeError("Non-finite valid-pixel logits encountered")
        positive = logits[masks & valid]
        positive_chunks.append(positive.float().cpu().numpy())
        if valid_logits.numel():
            minimum_logit = min(minimum_logit, float(valid_logits.min().item()))
            maximum_logit = max(maximum_logit, float(valid_logits.max().item()))
        total_valid += int(valid_logits.numel())
        total_positive += int(positive.numel())
        processed += int(images.shape[0])
    if total_positive == 0:
        raise RuntimeError("Exact F1 threshold search requires at least one positive valid pixel")

    positive_logits = np.concatenate(positive_chunks).astype(np.float32, copy=False)
    thresholds = np.unique(positive_logits)
    threshold_tensor = torch.from_numpy(thresholds).to(device)
    fp_counts = torch.zeros(len(thresholds), dtype=torch.int64, device=device)
    histogram_min = float(minimum_logit)
    histogram_max = float(maximum_logit)
    if not histogram_max > histogram_min:
        histogram_max = histogram_min + 1e-6
    negative_hist = torch.zeros(args.quantile_bins, dtype=torch.int64, device=device)

    named_thresholds = parse_named_thresholds(args.fixed_logit_threshold)
    provisional_counts = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in named_thresholds}
    scene_stats: dict[str, dict] = defaultdict(
        lambda: {"patches": 0, "valid_pixels": 0, "positive_pixels": 0, "fp": defaultdict(int)}
    )

    for images, masks, valid, scene_ids in tqdm(loader, desc=f"{args.split_name} pass2", unit="batch"):
        images = images.to(device, non_blocking=True)
        masks = (masks > 0).to(device, non_blocking=True).bool()
        valid = valid.to(device, non_blocking=True).bool()
        logits = model(images)
        negative_logits = logits[valid & ~masks].float()
        if negative_logits.numel():
            sorted_negative = torch.sort(negative_logits).values
            first_ge = torch.searchsorted(sorted_negative, threshold_tensor, right=False)
            fp_counts += sorted_negative.numel() - first_ge
            negative_hist += torch.histc(
                negative_logits,
                bins=args.quantile_bins,
                min=histogram_min,
                max=histogram_max,
            ).to(torch.int64)

        for name, threshold in named_thresholds.items():
            prediction = logits >= threshold
            batch_counts = counts_for_prediction(prediction, masks, valid)
            for key, value in batch_counts.items():
                provisional_counts[name][key] += value
            fp_per_patch = torch.count_nonzero(prediction & ~masks & valid, dim=(1, 2, 3)).cpu().tolist()
            for scene_id, fp_value in zip(scene_ids, fp_per_patch):
                scene_stats[scene_id]["fp"][name] += int(fp_value)

        valid_per_patch = torch.count_nonzero(valid, dim=(1, 2, 3)).cpu().tolist()
        positive_per_patch = torch.count_nonzero(masks & valid, dim=(1, 2, 3)).cpu().tolist()
        for scene_id, valid_value, positive_value in zip(scene_ids, valid_per_patch, positive_per_patch):
            scene_stats[scene_id]["patches"] += 1
            scene_stats[scene_id]["valid_pixels"] += int(valid_value)
            scene_stats[scene_id]["positive_pixels"] += int(positive_value)

    fp_numpy = fp_counts.cpu().numpy()
    best, average_precision = exact_search(thresholds, positive_logits, fp_numpy)
    named_thresholds["oracle"] = float(best["logit_threshold"])

    # Validation has no frozen threshold, so its selected oracle threshold needs
    # one final pass for per-scene FP. Test splits already collect frozen-threshold
    # scene counts during pass two; their oracle remains aggregate diagnostics only.
    oracle_counts = {
        "tp": best["tp"],
        "fp": best["fp"],
        "fn": best["fn"],
        "tn": total_valid - best["tp"] - best["fp"] - best["fn"],
    }
    if not provisional_counts:
        for images, masks, valid, scene_ids in tqdm(loader, desc=f"{args.split_name} oracle-scenes", unit="batch"):
            images = images.to(device, non_blocking=True)
            masks = (masks > 0).to(device, non_blocking=True).bool()
            valid = valid.to(device, non_blocking=True).bool()
            prediction = model(images) >= best["logit_threshold"]
            fp_per_patch = torch.count_nonzero(prediction & ~masks & valid, dim=(1, 2, 3)).cpu().tolist()
            for scene_id, fp_value in zip(scene_ids, fp_per_patch):
                scene_stats[scene_id]["fp"]["oracle"] += int(fp_value)

    fixed_results = {}
    for name, counts in provisional_counts.items():
        metrics = metrics_from_counts(counts)
        fixed_results[name] = {
            "probability_threshold": probability_from_logit(named_thresholds[name]),
            "logit_threshold": named_thresholds[name],
            **metrics,
            "predicted_positive_pixels": int(metrics["tp"] + metrics["fp"]),
        }
    fixed_results["oracle"] = {
        "probability_threshold": best["probability_threshold"],
        "logit_threshold": best["logit_threshold"],
        **metrics_from_counts(oracle_counts),
        "predicted_positive_pixels": int(best["tp"] + best["fp"]),
    }

    per_scene_path = Path(args.scene_fp_output).resolve() if args.scene_fp_output else Path(args.output).with_suffix(".scene_fp.csv").resolve()
    per_scene_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_names = list(provisional_counts) or ["oracle"]
    with per_scene_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["scene_id", "patches", "valid_pixels", "positive_pixels"] + [
            f"fp_{name}" for name in threshold_names
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scene_id in sorted(scene_stats):
            values = scene_stats[scene_id]
            row = {
                "scene_id": scene_id,
                "patches": values["patches"],
                "valid_pixels": values["valid_pixels"],
                "positive_pixels": values["positive_pixels"],
            }
            row.update({f"fp_{name}": values["fp"].get(name, 0) for name in threshold_names})
            writer.writerow(row)

    negative_hist_numpy = negative_hist.cpu().numpy()
    report = {
        "algorithm": "pgss_tiny_v1",
        "split": args.split_name,
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "architecture": checkpoint.get("architecture", "PGSSTinyV1"),
        "loss": checkpoint.get("loss"),
        "width": int(checkpoint.get("width", 16)),
        "head_width": int(checkpoint.get("head_width", 8)),
        "bands": list(bands),
        "normalization": "/65535",
        "invalid_input_policy": "all input channels explicitly set to zero where all source bands are zero",
        "model_eval_mode": not model.training,
        "processed_patches": processed,
        "valid_pixels": total_valid,
        "positive_pixels": total_positive,
        "negative_pixels": total_valid - total_positive,
        "nonfinite_valid_logits": nonfinite,
        "exact_search": {
            "domain": "float32_logit",
            "candidate_rule": "every distinct positive valid-pixel logit",
            "candidate_count": int(len(thresholds)),
            "tie_break": "highest logit threshold",
            "best": best,
        },
        "pr_auc": average_precision,
        "pr_auc_definition": "average precision (step-wise area under the exact pixel PR curve)",
        "probability_distribution": {
            "positive_exact": quantiles(positive_logits, (0.01, 0.05, 0.5, 0.95, 0.99)),
            "negative_histogram_approx": histogram_quantiles(
                negative_hist_numpy,
                histogram_min,
                histogram_max,
                (0.5, 0.95, 0.99, 0.999),
            ),
            "negative_quantile_method": {
                "domain": "logit",
                "bins": args.quantile_bins,
                "minimum_logit": histogram_min,
                "maximum_logit": histogram_max,
            },
        },
        "threshold_results": fixed_results,
        "per_scene_fp_csv": str(per_scene_path),
        "wall_time_s": time.perf_counter() - started,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--fixed-logit-threshold", action="append", default=[], metavar="NAME=LOGIT")
    parser.add_argument("--scene-fp-output", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=0)
    parser.add_argument("--quantile-bins", type=int, default=65536)
    args = parser.parse_args()
    report = evaluate(args)
    write_json(Path(args.output).resolve(), report)
    best = report["exact_search"]["best"]
    print(
        f"{args.split_name}: probability={best['probability_threshold']:.12g} "
        f"logit={best['logit_threshold']:.9g} P={best['precision']:.6f} "
        f"R={best['recall']:.6f} F1={best['f1']:.6f} AP={report['pr_auc']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
