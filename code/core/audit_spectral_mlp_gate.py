#!/usr/bin/env python3
"""Exact threshold, tile, event, and frozen-PGSS audit for SpectralMLPGate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# code/ root on sys.path so the core/models/train/eval packages resolve
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.exact_eval_pgss import exact_search
from core.pgss_common import (
    metrics_from_counts,
    read_mask,
    read_rows,
    scene_id_from_row,
    sha256_file,
    write_json,
)
from models.pgss_model import PGSSTinyV1
from models.spectral_mlp_gate import SpectralMLPGate, count_parameters, estimate_macs_per_pixel, prepare_gate_input


COUNT_KEYS = ("tp", "fp", "fn", "tn")
PATCH_COORD_RE = re.compile(r"_patch_(\d+)_(\d+)$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def probability_from_logit(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class GateAuditDataset(Dataset):
    """Unchanged PGSS samples plus sparse scene coordinates for event merging."""

    def __init__(self, csv_path: str | Path, bands: tuple[int, ...] = (7, 6, 2)) -> None:
        self.csv_path = Path(csv_path).resolve()
        self.rows = read_rows(self.csv_path)
        self.bands = tuple(int(value) for value in bands)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = Path(row["data_file"])
        with rasterio.open(image_path) as source:
            image = source.read(self.bands).astype(np.float32) / 65535.0
            valid_source = source.read().astype(np.float32, copy=False)
            transform = source.transform
        truth = read_mask(row["mask_file"])
        valid = np.any(valid_source != 0, axis=0)
        image[:, ~valid] = 0.0
        if image.shape[1:] != truth.shape:
            raise ValueError(f"Image/mask mismatch for {image_path}: {image.shape[1:]} != {truth.shape}")

        match = PATCH_COORD_RE.search(image_path.stem)
        if match:
            offset_y, offset_x = int(match.group(1)), int(match.group(2))
            coordinate_source = "filename_patch_y_x"
        elif transform.a and transform.e:
            offset_x = int(round(transform.c / abs(transform.a)))
            offset_y = int(round(-transform.f / abs(transform.e)))
            coordinate_source = "geotransform_pixel_grid"
        else:
            # Keep unlocatable patches separated so events are never falsely merged.
            offset_y, offset_x = index * (truth.shape[0] + 1), 0
            coordinate_source = "isolated_patch_fallback"
        return (
            torch.from_numpy(image),
            torch.from_numpy(truth[None].astype(np.float32)),
            torch.from_numpy(valid[None].astype(np.bool_)),
            scene_id_from_row(row),
            index,
            offset_y,
            offset_x,
            coordinate_source,
        )


def make_audit_loader(
    csv_path: Path,
    bands: tuple[int, ...],
    batch_size: int,
    workers: int,
) -> tuple[GateAuditDataset, DataLoader]:
    dataset = GateAuditDataset(csv_path, bands)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    return dataset, loader


def load_gate(path: Path, device: torch.device) -> tuple[SpectralMLPGate, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SpectralMLPGate(
        in_channels=int(checkpoint.get("in_channels", 3)),
        hidden1=int(checkpoint.get("hidden1", 16)),
        hidden2=int(checkpoint.get("hidden2", 8)),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model, checkpoint


def load_pgss(path: Path, device: torch.device) -> tuple[PGSSTinyV1, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    bands = tuple(checkpoint.get("bands", (7, 6, 2)))
    model = PGSSTinyV1(
        in_channels=len(bands),
        width=int(checkpoint.get("width", 16)),
        head_width=int(checkpoint.get("head_width", 8)),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model, checkpoint


def build_tile_masks(
    candidate: torch.Tensor,
    valid: torch.Tensor,
    tile_size: int,
    halo: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected grid, eligible grid, base-tile core, and unique halo masks."""
    if candidate.shape != valid.shape or candidate.ndim != 4 or candidate.shape[1] != 1:
        raise ValueError("candidate and valid must both have shape [B,1,H,W]")
    selected = F.max_pool2d(candidate.float(), tile_size, stride=tile_size, ceil_mode=True) > 0
    eligible = F.max_pool2d(valid.float(), tile_size, stride=tile_size, ceil_mode=True) > 0
    core = selected.repeat_interleave(tile_size, dim=2).repeat_interleave(tile_size, dim=3)
    core = core[:, :, : candidate.shape[2], : candidate.shape[3]]
    unique_halo = F.max_pool2d(core.float(), 2 * halo + 1, stride=1, padding=halo) > 0
    return selected, eligible, core, unique_halo


def repeated_halo_weights(
    height: int,
    width: int,
    tile_size: int,
    halo: int,
    device: torch.device,
) -> torch.Tensor:
    rows = math.ceil(height / tile_size)
    cols = math.ceil(width / tile_size)
    weights = torch.empty((1, 1, rows, cols), dtype=torch.int64, device=device)
    for row in range(rows):
        y0, y1 = row * tile_size, min((row + 1) * tile_size, height)
        expanded_h = min(height, y1 + halo) - max(0, y0 - halo)
        for col in range(cols):
            x0, x1 = col * tile_size, min((col + 1) * tile_size, width)
            expanded_w = min(width, x1 + halo) - max(0, x0 - halo)
            weights[0, 0, row, col] = expanded_h * expanded_w
    return weights


def sum_scene_counts(scenes: dict[str, dict[str, int]], keys: Iterable[str]) -> dict[str, int]:
    return {key: sum(int(scene.get(key, 0)) for scene in scenes.values()) for key in keys}


def prediction_count_tensors(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction = prediction.bool() & valid.bool()
    truth = truth.bool() & valid.bool()
    valid = valid.bool()
    return {
        "tp": torch.count_nonzero(prediction & truth, dim=(1, 2, 3)),
        "fp": torch.count_nonzero(prediction & ~truth & valid, dim=(1, 2, 3)),
        "fn": torch.count_nonzero(~prediction & truth & valid, dim=(1, 2, 3)),
        "tn": torch.count_nonzero(~prediction & ~truth & valid, dim=(1, 2, 3)),
    }


def empty_scene_stats() -> dict[str, int]:
    return {
        "patches": 0,
        "valid_pixels": 0,
        "positive_pixels": 0,
        "candidate_pixels": 0,
        "candidate_positive_pixels": 0,
        "eligible_tiles": 0,
        "candidate_tiles": 0,
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
        "gated_tp": 0,
        "gated_fp": 0,
        "gated_fn": 0,
        "gated_tn": 0,
        "added_fn": 0,
        "removed_fp": 0,
    }


def add_per_patch(
    scene_stats: dict[str, dict[str, int]],
    scene_ids: list[str],
    values: dict[str, torch.Tensor],
) -> None:
    cpu_values = {key: tensor.detach().cpu().tolist() for key, tensor in values.items()}
    for index, scene_id in enumerate(scene_ids):
        scene = scene_stats[scene_id]
        for key, items in cpu_values.items():
            scene[key] += int(items[index])


class SparseEventAccumulator:
    """Merge positive pixels into scene-level 8-connected components on a sparse grid."""

    def __init__(self, combination_keys: list[str]) -> None:
        self.keys = combination_keys
        self.coords: dict[str, list[np.ndarray]] = defaultdict(list)
        self.flags: dict[str, dict[str, dict[str, list[np.ndarray]]]] = defaultdict(
            lambda: {
                key: {"pixel_gate": [], "tile": [], "final": []}
                for key in self.keys
            }
        )
        self.coordinate_sources: set[str] = set()

    def add_batch(
        self,
        scene_ids: list[str],
        offsets_y: list[int],
        offsets_x: list[int],
        coordinate_sources: list[str],
        truth_valid: torch.Tensor,
        flags: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        positions = torch.nonzero(truth_valid[:, 0], as_tuple=False).cpu().numpy()
        counts = torch.count_nonzero(truth_valid, dim=(1, 2, 3)).cpu().tolist()
        flat_flags = {
            key: {
                name: tensor[truth_valid].detach().cpu().numpy().astype(np.bool_, copy=False)
                for name, tensor in values.items()
            }
            for key, values in flags.items()
        }
        cursor = 0
        for batch_index, count in enumerate(counts):
            self.coordinate_sources.add(coordinate_sources[batch_index])
            if not count:
                continue
            local = positions[cursor : cursor + count, 1:3].astype(np.int64, copy=True)
            local[:, 0] += int(offsets_y[batch_index])
            local[:, 1] += int(offsets_x[batch_index])
            scene_id = scene_ids[batch_index]
            self.coords[scene_id].append(local)
            for key in self.keys:
                for name in ("pixel_gate", "tile", "final"):
                    self.flags[scene_id][key][name].append(
                        flat_flags[key][name][cursor : cursor + count]
                    )
            cursor += count

    @staticmethod
    def _components(coords: np.ndarray) -> list[np.ndarray]:
        parent = np.arange(len(coords), dtype=np.int64)
        rank = np.zeros(len(coords), dtype=np.int8)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left == root_right:
                return
            if rank[root_left] < rank[root_right]:
                root_left, root_right = root_right, root_left
            parent[root_right] = root_left
            if rank[root_left] == rank[root_right]:
                rank[root_left] += 1

        lookup: dict[tuple[int, int], int] = {}
        for index, (y_value, x_value) in enumerate(coords.tolist()):
            y_coord, x_coord = int(y_value), int(x_value)
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
                neighbor = lookup.get((y_coord + dy, x_coord + dx))
                if neighbor is not None:
                    union(index, neighbor)
            lookup[(y_coord, x_coord)] = index
        groups: dict[int, list[int]] = defaultdict(list)
        for index in range(len(coords)):
            groups[find(index)].append(index)
        return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]

    def finalize(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, int]]]]:
        aggregate = {
            key: {
                "total_events": 0,
                "tiny_events": 0,
                "pixel_gate_recalled_events": 0,
                "tile_recalled_events": 0,
                "final_recalled_events": 0,
                "tiny_pixel_gate_recalled_events": 0,
                "tiny_tile_recalled_events": 0,
                "tiny_final_recalled_events": 0,
                "missed_pixel_gate_events": [],
                "missed_tile_events": [],
                "missed_final_events": [],
                "missed_tiny_pixel_gate_events": [],
                "missed_tiny_tile_events": [],
                "missed_tiny_final_events": [],
            }
            for key in self.keys
        }
        per_scene: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        for scene_id in sorted(self.coords):
            raw_coords = np.concatenate(self.coords[scene_id], axis=0)
            unique_coords, inverse = np.unique(raw_coords, axis=0, return_inverse=True)
            unique_flags: dict[str, dict[str, np.ndarray]] = {}
            for key in self.keys:
                unique_flags[key] = {}
                for name in ("pixel_gate", "tile", "final"):
                    raw_flags = np.concatenate(self.flags[scene_id][key][name])
                    merged = np.zeros(len(unique_coords), dtype=np.bool_)
                    np.logical_or.at(merged, inverse, raw_flags)
                    unique_flags[key][name] = merged
            components = self._components(unique_coords)
            components.sort(key=lambda indices: tuple(unique_coords[indices].min(axis=0).tolist()))
            for key in self.keys:
                scene_counts = {
                    "total_events": len(components),
                    "tiny_events": sum(len(indices) <= 2 for indices in components),
                    "pixel_gate_recalled_events": 0,
                    "tile_recalled_events": 0,
                    "final_recalled_events": 0,
                    "tiny_pixel_gate_recalled_events": 0,
                    "tiny_tile_recalled_events": 0,
                    "tiny_final_recalled_events": 0,
                }
                for event_index, indices in enumerate(components, start=1):
                    is_tiny = len(indices) <= 2
                    anchor = unique_coords[indices].min(axis=0)
                    event_id = (
                        f"{scene_id}:event_{event_index:05d}@{int(anchor[0])},{int(anchor[1])}"
                        f":pixels={len(indices)}"
                    )
                    for name, count_key, missed_key in (
                        ("pixel_gate", "pixel_gate_recalled_events", "missed_pixel_gate_events"),
                        ("tile", "tile_recalled_events", "missed_tile_events"),
                        ("final", "final_recalled_events", "missed_final_events"),
                    ):
                        recalled = bool(np.any(unique_flags[key][name][indices]))
                        if recalled:
                            scene_counts[count_key] += 1
                        else:
                            aggregate[key][missed_key].append(event_id)
                        if is_tiny:
                            tiny_count_key = f"tiny_{count_key}"
                            if recalled:
                                scene_counts[tiny_count_key] += 1
                            else:
                                aggregate[key][f"missed_tiny_{name}_events"].append(event_id)
                per_scene[scene_id][key] = scene_counts
                for count_key in (
                    "total_events",
                    "tiny_events",
                    "pixel_gate_recalled_events",
                    "tile_recalled_events",
                    "final_recalled_events",
                    "tiny_pixel_gate_recalled_events",
                    "tiny_tile_recalled_events",
                    "tiny_final_recalled_events",
                ):
                    aggregate[key][count_key] += scene_counts[count_key]
        for values in aggregate.values():
            total = values["total_events"]
            tiny_total = values["tiny_events"]
            values["pixel_gate_event_recall"] = values["pixel_gate_recalled_events"] / total if total else float("nan")
            values["tile_event_recall"] = values["tile_recalled_events"] / total if total else float("nan")
            values["final_event_recall"] = values["final_recalled_events"] / total if total else float("nan")
            values["tiny_pixel_gate_event_recall"] = values["tiny_pixel_gate_recalled_events"] / tiny_total if tiny_total else float("nan")
            values["tiny_tile_event_recall"] = values["tiny_tile_recalled_events"] / tiny_total if tiny_total else float("nan")
            values["tiny_final_event_recall"] = values["tiny_final_recalled_events"] / tiny_total if tiny_total else float("nan")
        return aggregate, per_scene


def select_recall_thresholds(
    positive_logits: np.ndarray,
    constraints: list[float],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    positive_sorted = np.sort(positive_logits.astype(np.float32, copy=False))
    thresholds = np.unique(positive_sorted)
    tp = len(positive_sorted) - np.searchsorted(positive_sorted, thresholds, side="left")
    recall = tp.astype(np.float64) / len(positive_sorted)
    selections = []
    for constraint in constraints:
        feasible = np.flatnonzero(recall >= constraint)
        if feasible.size == 0:
            raise RuntimeError(f"No exact logit threshold satisfies recall >= {constraint}")
        index = int(feasible[-1])
        logit = float(thresholds[index])
        selections.append(
            {
                "recall_constraint": float(constraint),
                "logit_threshold": logit,
                "probability_threshold": probability_from_logit(logit),
                "positive_pixels": int(len(positive_sorted)),
                "retained_positive_pixels": int(tp[index]),
                "pixel_gate_recall": float(recall[index]),
                "threshold_index": index,
            }
        )
    return thresholds, tp, selections


def _validation_scene_template() -> dict[str, int]:
    return {
        "patches": 0,
        "valid_pixels": 0,
        "positive_pixels": 0,
        "candidate_pixels": 0,
        "candidate_positive_pixels": 0,
        "eligible_tiles": 0,
        "candidate_tiles": 0,
        "base_tile_valid_pixels": 0,
        "tile_covered_positive_pixels": 0,
    }


@torch.inference_mode()
def validation_exact_audit(
    model: SpectralMLPGate,
    csv_path: Path,
    split_name: str,
    bands: tuple[int, ...],
    constraints: list[float],
    tile_specs: list[dict[str, int]],
    batch_size: int,
    workers: int,
    device: torch.device,
    collect_curve: bool = False,
    collect_events: bool = False,
) -> dict[str, Any]:
    model.eval()
    dataset, loader = make_audit_loader(csv_path, bands, batch_size, workers)
    positive_chunks: list[np.ndarray] = []
    total_valid = total_positive = processed = 0
    started = time.perf_counter()
    for images, masks, valid, *_rest in tqdm(loader, desc=f"{split_name} gate pass1", unit="batch", leave=False):
        images = images.to(device, non_blocking=True)
        masks = (masks > 0).to(device, non_blocking=True).bool()
        valid = valid.to(device, non_blocking=True).bool()
        logits = model(prepare_gate_input(images, valid))
        if not bool(torch.all(torch.isfinite(logits[valid]))):
            raise RuntimeError(f"Non-finite Gate logits in {split_name}")
        positive = logits[masks & valid].float()
        positive_chunks.append(positive.cpu().numpy())
        total_valid += int(torch.count_nonzero(valid).item())
        total_positive += int(positive.numel())
        processed += int(images.shape[0])
    positive_logits = np.concatenate(positive_chunks).astype(np.float32, copy=False)
    thresholds, tp_by_threshold, selections = select_recall_thresholds(positive_logits, constraints)
    threshold_tensor = torch.from_numpy(thresholds).to(device)
    fp_counts = torch.zeros(len(thresholds), dtype=torch.int64, device=device)

    combinations = []
    for selection in selections:
        for tile in tile_specs:
            key = f"{selection['recall_constraint']:.4f}|{tile['size']}"
            combinations.append(
                {
                    "key": key,
                    "selection": selection,
                    "tile": tile,
                    "scenes": defaultdict(_validation_scene_template),
                }
            )
    event_accumulator = SparseEventAccumulator([item["key"] for item in combinations]) if collect_events else None

    for batch in tqdm(loader, desc=f"{split_name} gate pass2", unit="batch", leave=False):
        images, masks, valid, scene_ids_raw, _indices, offsets_y, offsets_x, coordinate_sources_raw = batch
        scene_ids = list(scene_ids_raw)
        coordinate_sources = list(coordinate_sources_raw)
        images = images.to(device, non_blocking=True)
        masks = (masks > 0).to(device, non_blocking=True).bool()
        valid = valid.to(device, non_blocking=True).bool()
        logits = model(prepare_gate_input(images, valid))
        negative = logits[~masks & valid].float()
        if negative.numel():
            sorted_negative = torch.sort(negative).values
            first_ge = torch.searchsorted(sorted_negative, threshold_tensor, right=False)
            fp_counts += sorted_negative.numel() - first_ge

        event_flags: dict[str, dict[str, torch.Tensor]] = {}
        for combination in combinations:
            selection, tile = combination["selection"], combination["tile"]
            candidate = valid & (logits >= float(selection["logit_threshold"]))
            selected, eligible, core, _unique_halo = build_tile_masks(
                candidate, valid, int(tile["size"]), int(tile["halo"])
            )
            values = {
                "patches": torch.ones(images.shape[0], dtype=torch.int64, device=device),
                "valid_pixels": torch.count_nonzero(valid, dim=(1, 2, 3)),
                "positive_pixels": torch.count_nonzero(masks & valid, dim=(1, 2, 3)),
                "candidate_pixels": torch.count_nonzero(candidate, dim=(1, 2, 3)),
                "candidate_positive_pixels": torch.count_nonzero(candidate & masks, dim=(1, 2, 3)),
                "eligible_tiles": torch.count_nonzero(eligible, dim=(1, 2, 3)),
                "candidate_tiles": torch.count_nonzero(selected, dim=(1, 2, 3)),
                "base_tile_valid_pixels": torch.count_nonzero(core & valid, dim=(1, 2, 3)),
                "tile_covered_positive_pixels": torch.count_nonzero(core & masks & valid, dim=(1, 2, 3)),
            }
            add_per_patch(combination["scenes"], scene_ids, values)
            if event_accumulator is not None:
                event_flags[combination["key"]] = {
                    "pixel_gate": candidate,
                    "tile": core,
                    "final": core,
                }
        if event_accumulator is not None:
            event_accumulator.add_batch(
                scene_ids,
                offsets_y.cpu().tolist(),
                offsets_x.cpu().tolist(),
                coordinate_sources,
                masks & valid,
                event_flags,
            )

    fp_numpy = fp_counts.cpu().numpy()
    _best_f1, average_precision = exact_search(thresholds, positive_logits, fp_numpy)
    total_negative = total_valid - total_positive
    for selection in selections:
        index = int(selection.pop("threshold_index"))
        fp = int(fp_numpy[index])
        tp = int(tp_by_threshold[index])
        selection.update(
            {
                "selection_split": split_name,
                "fp": fp,
                "fn": total_positive - tp,
                "tn": total_negative - fp,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "candidate_pixels": tp + fp,
                "candidate_pixel_ratio": (tp + fp) / total_valid,
            }
        )

    event_aggregate, event_scene = event_accumulator.finalize() if event_accumulator else ({}, {})
    scene_rows = []
    tile_rows = []
    for combination in combinations:
        selection, tile, scenes = combination["selection"], combination["tile"], combination["scenes"]
        aggregate = sum_scene_counts(scenes, _validation_scene_template().keys())
        positive_scene_pixel_recalls = []
        positive_scene_tile_recalls = []
        missed_positive_scenes = []
        for scene_id in sorted(scenes):
            scene = scenes[scene_id]
            pixel_recall = scene["candidate_positive_pixels"] / scene["positive_pixels"] if scene["positive_pixels"] else float("nan")
            tile_recall = scene["tile_covered_positive_pixels"] / scene["positive_pixels"] if scene["positive_pixels"] else float("nan")
            if scene["positive_pixels"]:
                positive_scene_pixel_recalls.append((pixel_recall, scene_id))
                positive_scene_tile_recalls.append((tile_recall, scene_id))
                if scene["tile_covered_positive_pixels"] == 0:
                    missed_positive_scenes.append(scene_id)
            events = event_scene.get(scene_id, {}).get(combination["key"], {})
            scene_rows.append(
                {
                    "dataset": split_name,
                    "protocol": "threshold_selection",
                    "recall_constraint": selection["recall_constraint"],
                    "logit_threshold": selection["logit_threshold"],
                    "tile_size": tile["size"],
                    "halo": tile["halo"],
                    "scene_id": scene_id,
                    **scene,
                    "candidate_pixel_ratio": scene["candidate_pixels"] / scene["valid_pixels"] if scene["valid_pixels"] else 0.0,
                    "pixel_gate_recall": pixel_recall,
                    "candidate_tile_ratio": scene["candidate_tiles"] / scene["eligible_tiles"] if scene["eligible_tiles"] else 0.0,
                    "base_tile_valid_coverage": scene["base_tile_valid_pixels"] / scene["valid_pixels"] if scene["valid_pixels"] else 0.0,
                    "tile_fire_recall": tile_recall,
                    **events,
                }
            )
        worst_pixel = min(positive_scene_pixel_recalls) if positive_scene_pixel_recalls else (float("nan"), "")
        worst_tile = min(positive_scene_tile_recalls) if positive_scene_tile_recalls else (float("nan"), "")
        events = event_aggregate.get(combination["key"], {})
        tile_rows.append(
            {
                "dataset": split_name,
                "protocol": "threshold_selection",
                "recall_constraint": selection["recall_constraint"],
                "logit_threshold": selection["logit_threshold"],
                "probability_threshold": selection["probability_threshold"],
                "tile_size": tile["size"],
                "halo": tile["halo"],
                **aggregate,
                "candidate_pixel_ratio": aggregate["candidate_pixels"] / aggregate["valid_pixels"],
                "pixel_gate_recall": aggregate["candidate_positive_pixels"] / aggregate["positive_pixels"],
                "candidate_tile_ratio": aggregate["candidate_tiles"] / aggregate["eligible_tiles"],
                "base_tile_valid_coverage": aggregate["base_tile_valid_pixels"] / aggregate["valid_pixels"],
                "tile_fire_recall": aggregate["tile_covered_positive_pixels"] / aggregate["positive_pixels"],
                "worst_positive_scene_pixel_recall": worst_pixel[0],
                "worst_positive_scene_pixel_recall_id": worst_pixel[1],
                "worst_positive_scene_tile_recall": worst_tile[0],
                "worst_positive_scene_tile_recall_id": worst_tile[1],
                "positive_scenes": len(positive_scene_tile_recalls),
                "retained_positive_scenes": len(positive_scene_tile_recalls) - len(missed_positive_scenes),
                "retained_positive_scene_ratio": (
                    (len(positive_scene_tile_recalls) - len(missed_positive_scenes)) / len(positive_scene_tile_recalls)
                    if positive_scene_tile_recalls else float("nan")
                ),
                "missed_positive_scenes": missed_positive_scenes,
                **events,
            }
        )

    curve_rows = []
    if collect_curve:
        total_positive = len(positive_logits)
        fn = total_positive - tp_by_threshold
        precision = np.divide(
            tp_by_threshold,
            tp_by_threshold + fp_numpy,
            out=np.zeros_like(tp_by_threshold, dtype=np.float64),
            where=(tp_by_threshold + fp_numpy) > 0,
        )
        recall = tp_by_threshold.astype(np.float64) / total_positive
        selected_lookup = {
            float(item["logit_threshold"]): item["recall_constraint"] for item in selections
        }
        for index, threshold in enumerate(thresholds):
            curve_rows.append(
                {
                    "selection_split": split_name,
                    "logit_threshold": float(threshold),
                    "probability_threshold": probability_from_logit(float(threshold)),
                    "tp": int(tp_by_threshold[index]),
                    "fp": int(fp_numpy[index]),
                    "fn": int(fn[index]),
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "candidate_pixel_ratio": float((tp_by_threshold[index] + fp_numpy[index]) / total_valid),
                    "selected_for_recall_constraint": selected_lookup.get(float(threshold), ""),
                }
            )

    return {
        "split": split_name,
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "processed_patches": processed,
        "valid_pixels": total_valid,
        "positive_pixels": total_positive,
        "pr_auc": average_precision,
        "selections": selections,
        "tile_rows": tile_rows,
        "scene_rows": scene_rows,
        "curve_rows": curve_rows,
        "event_coordinate_sources": sorted(event_accumulator.coordinate_sources) if event_accumulator else [],
        "wall_time_s": time.perf_counter() - started,
    }


def _metrics_with_prefix(counts: dict[str, int], prefix: str) -> dict[str, Any]:
    raw = {key: int(counts[f"{prefix}_{key}"]) for key in COUNT_KEYS}
    return metrics_from_counts(raw)


@torch.inference_mode()
def evaluate_test_split(
    gate: SpectralMLPGate,
    pgss: PGSSTinyV1,
    csv_path: Path,
    split_name: str,
    bands: tuple[int, ...],
    protocols: list[dict[str, Any]],
    tile_specs: list[dict[str, int]],
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, Any]:
    gate.eval()
    pgss.eval()
    dataset, loader = make_audit_loader(csv_path, bands, batch_size, workers)
    combinations = []
    for protocol in protocols:
        for selection in protocol["gate_selections"]:
            for tile in tile_specs:
                key = f"{protocol['name']}|{selection['recall_constraint']:.4f}|{tile['size']}"
                combinations.append(
                    {
                        "key": key,
                        "protocol": protocol,
                        "selection": selection,
                        "tile": tile,
                        "scenes": defaultdict(empty_scene_stats),
                    }
                )
    event_accumulator = SparseEventAccumulator([item["key"] for item in combinations])
    repeated_weight_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}
    processed = 0
    started = time.perf_counter()

    for batch in tqdm(loader, desc=f"{split_name} MLP Gate+PGSS", unit="batch"):
        images, masks, valid, scene_ids_raw, _indices, offsets_y, offsets_x, coordinate_sources_raw = batch
        scene_ids = list(scene_ids_raw)
        coordinate_sources = list(coordinate_sources_raw)
        images = images.to(device, non_blocking=True)
        masks = (masks > 0).to(device, non_blocking=True).bool()
        valid = valid.to(device, non_blocking=True).bool()
        gate_logits = gate(prepare_gate_input(images, valid))
        pgss_logits = pgss(images)
        if not bool(torch.all(torch.isfinite(gate_logits[valid]))):
            raise RuntimeError(f"Non-finite Gate logits in {split_name}")
        if not bool(torch.all(torch.isfinite(pgss_logits[valid]))):
            raise RuntimeError(f"Non-finite PGSS logits in {split_name}")
        height, width = images.shape[-2:]
        full_predictions = {
            protocol["name"]: pgss_logits >= float(protocol["pgss_logit_threshold"])
            for protocol in protocols
        }
        event_flags: dict[str, dict[str, torch.Tensor]] = {}

        for combination in combinations:
            protocol = combination["protocol"]
            selection = combination["selection"]
            tile = combination["tile"]
            tile_size, halo = int(tile["size"]), int(tile["halo"])
            candidate = valid & (gate_logits >= float(selection["logit_threshold"]))
            selected, eligible, core, unique_halo = build_tile_masks(
                candidate, valid, tile_size, halo
            )
            full_prediction = full_predictions[protocol["name"]]
            gated_prediction = full_prediction & core
            full_counts = prediction_count_tensors(full_prediction, masks, valid)
            gated_counts = prediction_count_tensors(gated_prediction, masks, valid)
            weight_key = (height, width, tile_size, halo)
            if weight_key not in repeated_weight_cache:
                repeated_weight_cache[weight_key] = repeated_halo_weights(
                    height, width, tile_size, halo, device
                )
            halo_weights = repeated_weight_cache[weight_key]
            values: dict[str, torch.Tensor] = {
                "patches": torch.ones(images.shape[0], dtype=torch.int64, device=device),
                "valid_pixels": torch.count_nonzero(valid, dim=(1, 2, 3)),
                "positive_pixels": torch.count_nonzero(masks & valid, dim=(1, 2, 3)),
                "candidate_pixels": torch.count_nonzero(candidate, dim=(1, 2, 3)),
                "candidate_positive_pixels": torch.count_nonzero(candidate & masks, dim=(1, 2, 3)),
                "eligible_tiles": torch.count_nonzero(eligible, dim=(1, 2, 3)),
                "candidate_tiles": torch.count_nonzero(selected, dim=(1, 2, 3)),
                "base_tile_area_pixels": torch.count_nonzero(core, dim=(1, 2, 3)),
                "base_tile_valid_pixels": torch.count_nonzero(core & valid, dim=(1, 2, 3)),
                "tile_covered_positive_pixels": torch.count_nonzero(core & masks & valid, dim=(1, 2, 3)),
                "halo_repeated_input_pixels": torch.sum(
                    selected.to(torch.int64) * halo_weights, dim=(1, 2, 3)
                ),
                "halo_unique_input_pixels": torch.count_nonzero(unique_halo, dim=(1, 2, 3)),
                "all_patch_pixels": torch.full(
                    (images.shape[0],), height * width, dtype=torch.int64, device=device
                ),
                "added_fn": torch.count_nonzero(
                    full_prediction & masks & valid & ~core, dim=(1, 2, 3)
                ),
                "removed_fp": torch.count_nonzero(
                    full_prediction & ~masks & valid & ~core, dim=(1, 2, 3)
                ),
            }
            values.update({f"full_{key}": value for key, value in full_counts.items()})
            values.update({f"gated_{key}": value for key, value in gated_counts.items()})
            add_per_patch(combination["scenes"], scene_ids, values)
            event_flags[combination["key"]] = {
                "pixel_gate": candidate,
                "tile": core,
                "final": gated_prediction,
            }
        event_accumulator.add_batch(
            scene_ids,
            offsets_y.cpu().tolist(),
            offsets_x.cpu().tolist(),
            coordinate_sources,
            masks & valid,
            event_flags,
        )
        processed += int(images.shape[0])

    event_aggregate, event_scene = event_accumulator.finalize()
    scene_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    for combination in combinations:
        protocol = combination["protocol"]
        selection = combination["selection"]
        tile = combination["tile"]
        scenes = combination["scenes"]
        aggregate = sum_scene_counts(scenes, empty_scene_stats().keys())
        positive_pixel_recalls = []
        positive_tile_recalls = []
        scenes_with_candidate_tile = []
        scenes_with_fire_covered = []
        missed_positive_scenes = []
        for scene_id in sorted(scenes):
            scene = scenes[scene_id]
            pixel_recall = scene["candidate_positive_pixels"] / scene["positive_pixels"] if scene["positive_pixels"] else float("nan")
            tile_recall = scene["tile_covered_positive_pixels"] / scene["positive_pixels"] if scene["positive_pixels"] else float("nan")
            if scene["positive_pixels"]:
                positive_pixel_recalls.append((pixel_recall, scene_id))
                positive_tile_recalls.append((tile_recall, scene_id))
                if scene["candidate_tiles"]:
                    scenes_with_candidate_tile.append(scene_id)
                if scene["tile_covered_positive_pixels"]:
                    scenes_with_fire_covered.append(scene_id)
                else:
                    missed_positive_scenes.append(scene_id)
            full_metrics = _metrics_with_prefix(scene, "full")
            gated_metrics = _metrics_with_prefix(scene, "gated")
            events = event_scene.get(scene_id, {}).get(combination["key"], {})
            scene_rows.append(
                {
                    "dataset": split_name,
                    "protocol": protocol["name"],
                    "gate_selection_split": selection["selection_split"],
                    "recall_constraint": selection["recall_constraint"],
                    "logit_threshold": selection["logit_threshold"],
                    "probability_threshold": selection["probability_threshold"],
                    "tile_size": tile["size"],
                    "halo": tile["halo"],
                    "scene_id": scene_id,
                    **scene,
                    "candidate_pixel_ratio": scene["candidate_pixels"] / scene["valid_pixels"] if scene["valid_pixels"] else 0.0,
                    "pixel_gate_recall": pixel_recall,
                    "candidate_tile_ratio": scene["candidate_tiles"] / scene["eligible_tiles"] if scene["eligible_tiles"] else 0.0,
                    "base_tile_area_ratio": scene["base_tile_area_pixels"] / scene["all_patch_pixels"] if scene["all_patch_pixels"] else 0.0,
                    "base_tile_valid_coverage": scene["base_tile_valid_pixels"] / scene["valid_pixels"] if scene["valid_pixels"] else 0.0,
                    "tile_fire_recall": tile_recall,
                    "halo_repeated_input_ratio": scene["halo_repeated_input_pixels"] / scene["all_patch_pixels"] if scene["all_patch_pixels"] else 0.0,
                    "halo_unique_input_ratio": scene["halo_unique_input_pixels"] / scene["all_patch_pixels"] if scene["all_patch_pixels"] else 0.0,
                    "full_precision": full_metrics["precision"],
                    "full_recall": full_metrics["recall"],
                    "full_f1": full_metrics["f1"],
                    "gated_precision": gated_metrics["precision"],
                    "gated_recall": gated_metrics["recall"],
                    "gated_f1": gated_metrics["f1"],
                    "f1_delta_pp": (gated_metrics["f1"] - full_metrics["f1"]) * 100.0,
                    **events,
                }
            )

        full_metrics = _metrics_with_prefix(aggregate, "full")
        gated_metrics = _metrics_with_prefix(aggregate, "gated")
        worst_pixel = min(positive_pixel_recalls) if positive_pixel_recalls else (float("nan"), "")
        worst_tile = min(positive_tile_recalls) if positive_tile_recalls else (float("nan"), "")
        events = event_aggregate[combination["key"]]
        tile_rows.append(
            {
                "dataset": split_name,
                "protocol": protocol["name"],
                "gate_selection_split": selection["selection_split"],
                "recall_constraint": selection["recall_constraint"],
                "logit_threshold": selection["logit_threshold"],
                "probability_threshold": selection["probability_threshold"],
                "tile_size": tile["size"],
                "halo": tile["halo"],
                **aggregate,
                "candidate_pixel_ratio": aggregate["candidate_pixels"] / aggregate["valid_pixels"],
                "pixel_gate_recall": aggregate["candidate_positive_pixels"] / aggregate["positive_pixels"],
                "candidate_tile_ratio": aggregate["candidate_tiles"] / aggregate["eligible_tiles"],
                "base_tile_area_ratio": aggregate["base_tile_area_pixels"] / aggregate["all_patch_pixels"],
                "base_tile_valid_coverage": aggregate["base_tile_valid_pixels"] / aggregate["valid_pixels"],
                "tile_fire_recall": aggregate["tile_covered_positive_pixels"] / aggregate["positive_pixels"],
                "halo_repeated_input_ratio": aggregate["halo_repeated_input_pixels"] / aggregate["all_patch_pixels"],
                "halo_unique_input_ratio": aggregate["halo_unique_input_pixels"] / aggregate["all_patch_pixels"],
                "worst_positive_scene_pixel_recall": worst_pixel[0],
                "worst_positive_scene_pixel_recall_id": worst_pixel[1],
                "worst_positive_scene_tile_recall": worst_tile[0],
                "worst_positive_scene_tile_recall_id": worst_tile[1],
                "positive_scenes": len(positive_tile_recalls),
                "positive_scenes_with_candidate_tile": len(scenes_with_candidate_tile),
                "positive_scene_candidate_tile_ratio": len(scenes_with_candidate_tile) / len(positive_tile_recalls) if positive_tile_recalls else float("nan"),
                "positive_scenes_with_fire_covered": len(scenes_with_fire_covered),
                "positive_scene_fire_covered_ratio": len(scenes_with_fire_covered) / len(positive_tile_recalls) if positive_tile_recalls else float("nan"),
                "missed_positive_scene_count": len(missed_positive_scenes),
                "missed_positive_scenes": missed_positive_scenes,
                "pgss_logit_threshold": protocol["pgss_logit_threshold"],
                "full_tp": full_metrics["tp"],
                "full_fp": full_metrics["fp"],
                "full_fn": full_metrics["fn"],
                "full_tn": full_metrics["tn"],
                "full_precision": full_metrics["precision"],
                "full_recall": full_metrics["recall"],
                "full_f1": full_metrics["f1"],
                "gated_tp": gated_metrics["tp"],
                "gated_fp": gated_metrics["fp"],
                "gated_fn": gated_metrics["fn"],
                "gated_tn": gated_metrics["tn"],
                "gated_precision": gated_metrics["precision"],
                "gated_recall": gated_metrics["recall"],
                "gated_f1": gated_metrics["f1"],
                "f1_delta_pp": (gated_metrics["f1"] - full_metrics["f1"]) * 100.0,
                "total_events": events["total_events"],
                "pixel_gate_recalled_events": events["pixel_gate_recalled_events"],
                "pixel_gate_event_recall": events["pixel_gate_event_recall"],
                "tile_recalled_events": events["tile_recalled_events"],
                "tile_event_recall": events["tile_event_recall"],
                "final_recalled_events": events["final_recalled_events"],
                "final_event_recall": events["final_event_recall"],
                "tiny_events": events["tiny_events"],
                "tiny_pixel_gate_recalled_events": events["tiny_pixel_gate_recalled_events"],
                "tiny_pixel_gate_event_recall": events["tiny_pixel_gate_event_recall"],
                "tiny_tile_recalled_events": events["tiny_tile_recalled_events"],
                "tiny_tile_event_recall": events["tiny_tile_event_recall"],
                "tiny_final_recalled_events": events["tiny_final_recalled_events"],
                "tiny_final_event_recall": events["tiny_final_event_recall"],
            }
        )
    return {
        "split": split_name,
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "processed_patches": processed,
        "tile_rows": tile_rows,
        "scene_rows": scene_rows,
        "event_audit": event_aggregate,
        "event_coordinate_sources": sorted(event_accumulator.coordinate_sources),
        "wall_time_s": time.perf_counter() - started,
    }


def _union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _find_row(
    rows: list[dict[str, Any]],
    dataset: str,
    protocol: str,
    constraint: float,
    tile_size: int,
) -> dict[str, Any]:
    for row in rows:
        if (
            row["dataset"] == dataset
            and row["protocol"] == protocol
            and math.isclose(float(row["recall_constraint"]), constraint)
            and int(row["tile_size"]) == tile_size
        ):
            return row
    raise KeyError((dataset, protocol, constraint, tile_size))


def _find_event_audit(
    audits: dict[str, dict[str, Any]],
    protocol: str,
    constraint: float,
    tile_size: int,
) -> dict[str, Any]:
    """Resolve event keys without depending on one formatted float string."""
    requested = f"{protocol}|{constraint:.4f}|{tile_size}"
    if requested in audits:
        return audits[requested]
    for key, value in audits.items():
        parts = str(key).split("|")
        if len(parts) != 3:
            continue
        try:
            key_constraint = float(parts[1])
            key_tile = int(parts[2])
        except ValueError:
            continue
        if parts[0] == protocol and math.isclose(key_constraint, constraint) and key_tile == tile_size:
            return value
    return {
        "status": "MISSING_EVENT_AUDIT_KEY",
        "requested_key": requested,
        "available_keys": sorted(str(key) for key in audits),
    }


def _fmt_pct(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.{digits}f}%"


def checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Keep reproducibility metadata while excluding non-JSON model tensors."""
    return {key: value for key, value in checkpoint.items() if key != "model_state_dict"}


def build_comparison_rows(
    spectral_rows: list[dict[str, Any]],
    b7_csv: Path,
    difficult_scene_id: str,
    scene_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    b7_rows = list(csv.DictReader(b7_csv.open(newline="", encoding="utf-8"))) if b7_csv.exists() else []
    b7_lookup = {
        (row["dataset"], row["protocol"], float(row["recall_constraint"]), int(row["tile_size"])): row
        for row in b7_rows
    }
    difficult_lookup = {
        (
            row["dataset"],
            row["protocol"],
            float(row["recall_constraint"]),
            int(row["tile_size"]),
        ): row
        for row in scene_rows
        if row.get("scene_id") == difficult_scene_id
    }
    output = []
    for row in spectral_rows:
        if row["protocol"] == "threshold_selection":
            continue
        key = (
            row["dataset"],
            row["protocol"],
            float(row["recall_constraint"]),
            int(row["tile_size"]),
        )
        b7 = b7_lookup.get(key)
        difficult = difficult_lookup.get(key)
        output.append(
            {
                "dataset": row["dataset"],
                "protocol": row["protocol"],
                "recall_constraint": row["recall_constraint"],
                "tile_size": row["tile_size"],
                "spectral_pixel_gate_recall": row["pixel_gate_recall"],
                "spectral_candidate_pixel_ratio": row["candidate_pixel_ratio"],
                "spectral_candidate_tile_ratio": row["candidate_tile_ratio"],
                "spectral_halo_repeated_input_ratio": row["halo_repeated_input_ratio"],
                "spectral_halo_unique_input_ratio": row["halo_unique_input_ratio"],
                "spectral_gate_pgss_f1": row["gated_f1"],
                "spectral_f1_delta_pp": row["f1_delta_pp"],
                "spectral_added_fn": row["added_fn"],
                "spectral_removed_fp": row["removed_fp"],
                "spectral_difficult_scene_fp": difficult["gated_fp"] if difficult else "",
                "b7_candidate_pixel_ratio": float(b7["candidate_pixel_ratio"]) if b7 else "",
                "b7_candidate_tile_ratio": float(b7["candidate_tile_ratio"]) if b7 else "",
                "b7_halo_repeated_input_ratio": float(b7["halo_repeated_input_ratio"]) if b7 else "",
                "b7_gate_pgss_f1": float(b7["gated_f1"]) if b7 else "",
                "b7_f1_delta_pp": float(b7["f1_delta_pp"]) if b7 else "",
                "tile_ratio_delta_vs_b7_pp": (
                    (float(row["candidate_tile_ratio"]) - float(b7["candidate_tile_ratio"])) * 100.0
                    if b7 else ""
                ),
            }
        )
    return output


def build_report(
    config: dict[str, Any],
    config_path: Path,
    checkpoint: dict[str, Any],
    train_result: dict[str, Any],
    active_val: dict[str, Any],
    land_val: dict[str, Any],
    test_rows: list[dict[str, Any]],
    scene_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    event_audits: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Spectral MLP Gate Feasibility Report",
        "",
        "## Protocol",
        "",
        "- Gate: `B7/B6/B2 -> Conv1x1(3,16) -> SiLU -> Conv1x1(16,8) -> SiLU -> Conv1x1(8,1)`.",
        "- The Gate was trained from scratch on ActiveFire-Train. Checkpoint and source thresholds use ActiveFire-Val only.",
        "- Land8Fire-Val is used only for the separately labeled target-calibrated diagnostic.",
        "- Test was not used for model, checkpoint, or threshold selection.",
        "- Events are scene-level sparse 8-connected components merged across patch boundaries using TIFF geotransforms or filename patch coordinates.",
        f"- GPU state: `{config['runtime']['gpu_state']}` on physical GPU `{config['runtime']['physical_gpu_id']}`; power and energy were not collected or ranked.",
        "",
        "## Model And Training",
        "",
        f"- Exact parameters: **{checkpoint['parameter_info']['total_parameters']}**.",
        f"- MACs/pixel: **{checkpoint['macs_per_pixel']}**; 256x256 MACs: **{checkpoint['macs_per_pixel'] * 256 * 256:,}**.",
        f"- Selected epoch: **{checkpoint['epoch']} / {config['training']['epochs']}**.",
        f"- Checkpoint criterion: {config['training']['checkpoint_tie_break']}.",
        f"- Training wall time: {train_result['wall_time_s']:.1f}s.",
        "",
        "## Exact Validation Thresholds",
        "",
        "| Selection split | Constraint | Logit | Probability | Pixel recall | Candidate pixels | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in (active_val, land_val):
        for selection in audit["selections"]:
            lines.append(
                f"| {audit['split']} | {_fmt_pct(selection['recall_constraint'], 2)} | "
                f"{selection['logit_threshold']:.9g} | {selection['probability_threshold']:.9g} | "
                f"{_fmt_pct(selection['pixel_gate_recall'])} | {_fmt_pct(selection['candidate_pixel_ratio'])} | "
                f"{audit['pr_auc']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Frozen Test Results",
            "",
            "| Dataset/protocol | Constraint | Tile/halo | Pixel recall | Event recall | Candidate tiles | Base area | Halo repeated | Halo unique | Gate+PGSS P/R/F1 | Delta F1 | Added FN | Removed FP | Missed scenes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    order = [
        ("ActiveFire-Test", "source_frozen"),
        ("Land8Fire-Test", "zero_shot_source_frozen"),
        ("Land8Fire-Test", "target_calibrated"),
    ]
    for dataset, protocol in order:
        for constraint in config["thresholds"]["recall_constraints"]:
            for tile in config["tiles"]:
                row = _find_row(test_rows, dataset, protocol, float(constraint), int(tile["size"]))
                lines.append(
                    f"| {dataset} / {protocol} | {_fmt_pct(constraint, 2)} | {tile['size']}/{tile['halo']} | "
                    f"{_fmt_pct(row['pixel_gate_recall'])} | {_fmt_pct(row['tile_event_recall'])} | "
                    f"{row['candidate_tiles']:,}/{row['eligible_tiles']:,} ({_fmt_pct(row['candidate_tile_ratio'])}) | "
                    f"{_fmt_pct(row['base_tile_area_ratio'])} | {_fmt_pct(row['halo_repeated_input_ratio'])} | "
                    f"{_fmt_pct(row['halo_unique_input_ratio'])} | "
                    f"{_fmt_pct(row['gated_precision'])}/{_fmt_pct(row['gated_recall'])}/{_fmt_pct(row['gated_f1'])} | "
                    f"{row['f1_delta_pp']:+.4f} pp | {row['added_fn']:,} | {row['removed_fp']:,} | "
                    f"{row['missed_positive_scene_count']} |"
                )

    active = _find_row(test_rows, "ActiveFire-Test", "source_frozen", 0.995, 32)
    land_zero = _find_row(test_rows, "Land8Fire-Test", "zero_shot_source_frozen", 0.995, 32)
    land_target = _find_row(test_rows, "Land8Fire-Test", "target_calibrated", 0.995, 32)
    go = config["go_no_go"]
    checks = {
        "ActiveFire pixel Gate Recall": active["pixel_gate_recall"] >= go["activefire_pixel_gate_recall_min"],
        "Land8Fire zero-shot pixel Gate Recall": land_zero["pixel_gate_recall"] >= go["land_zero_shot_pixel_gate_recall_min"],
        "ActiveFire tile event Recall": active["tile_event_recall"] >= go["event_recall_min"],
        "Land8Fire tile event Recall": land_zero["tile_event_recall"] >= go["event_recall_min"],
        "No missed positive scenes": active["missed_positive_scene_count"] <= go["missed_positive_scenes_max"] and land_zero["missed_positive_scene_count"] <= go["missed_positive_scenes_max"],
        "ActiveFire 32x32 tile ratio": active["candidate_tile_ratio"] <= go["tile32_candidate_ratio_max"],
        "Land8Fire 32x32 tile ratio": land_zero["candidate_tile_ratio"] <= go["tile32_candidate_ratio_max"],
        "ActiveFire F1 drop": active["f1_delta_pp"] >= -go["gate_pgss_f1_drop_pp_max"],
        "Land8Fire F1 drop": land_zero["f1_delta_pp"] >= -go["gate_pgss_f1_drop_pp_max"],
    }
    decision = "GO" if all(checks.values()) else "NO-GO"
    lines.extend(["", "## Go/No-Go", "", f"**Decision: {decision}.**", ""])
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}.")

    difficult_id = config["baselines"]["difficult_scene_id"]
    difficult = [
        row
        for row in scene_rows
        if row.get("dataset") == "Land8Fire-Test"
        and row.get("protocol") == "target_calibrated"
        and row.get("scene_id") == difficult_id
        and math.isclose(float(row.get("recall_constraint", 0)), 0.995)
        and int(row.get("tile_size", 0)) == 32
    ][0]
    lines.extend(
        [
            "",
            "## B7 Gate Comparison And Error Concentration",
            "",
            f"- At 99.5% / 32x32, Spectral MLP candidate tiles are Active `{_fmt_pct(active['candidate_tile_ratio'])}`, Land zero-shot `{_fmt_pct(land_zero['candidate_tile_ratio'])}`, and Land target `{_fmt_pct(land_target['candidate_tile_ratio'])}`.",
            f"- B7 baselines are Active `85.4605%`, Land zero-shot `71.1344%`, and Land target `72.7074%`.",
            f"- Difficult Land scene `{difficult_id}` FP: {difficult['full_fp']:,} -> {difficult['gated_fp']:,}; removed {difficult['removed_fp']:,}, added FN {difficult['added_fn']:,}.",
        ]
    )
    for dataset, protocol in (("ActiveFire-Test", "source_frozen"), ("Land8Fire-Test", "zero_shot_source_frozen")):
        rows = sorted(
            (
                row for row in scene_rows
                if row.get("dataset") == dataset
                and row.get("protocol") == protocol
                and math.isclose(float(row.get("recall_constraint", 0)), 0.995)
                and int(row.get("tile_size", 0)) == 32
            ),
            key=lambda row: (-int(row["added_fn"]), row["scene_id"]),
        )
        top = "; ".join(f"{row['scene_id']}={row['added_fn']}" for row in rows[:3] if int(row["added_fn"]) > 0) or "none"
        lines.append(f"- {dataset} top added-FN scenes: {top}.")

    lines.extend(["", "## Completely Missed Scenes And Events", ""])
    for label, row, audit_key in (
        ("ActiveFire source-frozen", active, "active_source_995_tile32"),
        ("Land8Fire zero-shot", land_zero, "land_zero_995_tile32"),
        ("Land8Fire target-calibrated", land_target, "land_target_995_tile32"),
    ):
        event_values = event_audits[audit_key]
        missed_scenes = row.get("missed_positive_scenes", [])
        missed_tile_events = event_values.get("missed_tile_events", [])
        missed_final_events = event_values.get("missed_final_events", [])
        lines.extend(
            [
                f"- {label} fully missed positive scenes ({len(missed_scenes)}): "
                f"{', '.join(missed_scenes) if missed_scenes else 'none'}.",
                f"- {label} Gate/tile-missed events ({len(missed_tile_events)}): "
                f"{'; '.join(missed_tile_events) if missed_tile_events else 'none'}.",
                f"- {label} final Gate+PGSS-missed events ({len(missed_final_events)}): "
                f"{'; '.join(missed_final_events) if missed_final_events else 'none'}.",
            ]
        )

    clearly_better = active["candidate_tile_ratio"] < 0.854605 and land_zero["candidate_tile_ratio"] < 0.711344
    halo_benefit = active["halo_repeated_input_ratio"] <= go["halo_repeated_input_ratio_preferred_max"] and land_zero["halo_repeated_input_ratio"] <= go["halo_repeated_input_ratio_preferred_max"]
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. **Does Spectral MLP clearly beat B7?** {'Yes' if clearly_better else 'No'} under the joint Active/Land 99.5%-Recall 32x32 criterion.",
            f"2. **Is the 32x32 candidate tile ratio below 40%?** {'Yes' if active['candidate_tile_ratio'] <= 0.4 and land_zero['candidate_tile_ratio'] <= 0.4 else 'No'}; Active/Land are {_fmt_pct(active['candidate_tile_ratio'])}/{_fmt_pct(land_zero['candidate_tile_ratio'])}.",
            f"3. **Are any positive scenes completely missed?** {'No' if active['missed_positive_scene_count'] == 0 and land_zero['missed_positive_scene_count'] == 0 else 'Yes'}; counts are Active `{active['missed_positive_scene_count']}`, Land `{land_zero['missed_positive_scene_count']}`.",
            f"4. **Does halo-adjusted work indicate useful compute savings?** {'Yes' if halo_benefit else 'No'}; repeated input ratios are Active/Land {_fmt_pct(active['halo_repeated_input_ratio'])}/{_fmt_pct(land_zero['halo_repeated_input_ratio'])}.",
            f"5. **Proceed to a real second-stage spatial Verifier?** {'Yes' if decision == 'GO' else 'No'} under the specified criteria.",
            "",
            "## Artifacts And Reproducibility",
            "",
            f"- Config: `{config_path}` SHA-256 `{sha256_file(config_path)}`.",
            f"- Best checkpoint: `{config['outputs']['best_checkpoint']}` SHA-256 `{sha256_file(Path(config['outputs']['best_checkpoint']))}`.",
            f"- ActiveFire-Val exact PR-AUC: `{active_val['pr_auc']:.8f}`; Land8Fire-Val exact PR-AUC: `{land_val['pr_auc']:.8f}`.",
            f"- Event coordinate sources: Active `{active_val['event_coordinate_sources']}`, Land `{land_val['event_coordinate_sources']}`.",
            f"- Comparison rows: `{len(comparison_rows)}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="audit protocol config; the historical default was removed from this release",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Gate+PGSS audit")
    device = torch.device("cuda")
    bands = tuple(int(value) for value in config["input"]["bands"])
    gate_path = Path(config["outputs"]["best_checkpoint"]).resolve()
    gate, gate_checkpoint = load_gate(gate_path, device)
    pgss, pgss_checkpoint = load_pgss(Path(config["pgss"]["checkpoint"]).resolve(), device)
    if gate.training or pgss.training:
        raise AssertionError("Both frozen models must be in eval mode")
    if count_parameters(gate)["total_parameters"] != 209 or estimate_macs_per_pixel() != 184:
        raise AssertionError("Spectral Gate architecture drift detected")

    thresholds = [float(value) for value in config["thresholds"]["recall_constraints"]]
    tile_specs = [{"size": int(item["size"]), "halo": int(item["halo"])} for item in config["tiles"]]
    batch_size = int(config["training"]["batch_size"])
    workers = max(1, int(config["training"]["workers"]) // 2)
    active_val = validation_exact_audit(
        gate, Path(config["data"]["activefire_val"]), "ActiveFire-Val", bands,
        thresholds, tile_specs, batch_size, workers, device,
        collect_curve=True, collect_events=True,
    )
    land_val = validation_exact_audit(
        gate, Path(config["data"]["land8fire_val"]), "Land8Fire-Val", bands,
        thresholds, tile_specs, batch_size, workers, device,
        collect_curve=True, collect_events=True,
    )
    active_protocols = [{
        "name": "source_frozen",
        "gate_selections": active_val["selections"],
        "pgss_logit_threshold": float(config["pgss"]["activefire_frozen_logit_threshold"]),
    }]
    land_protocols = [
        {
            "name": "zero_shot_source_frozen",
            "gate_selections": active_val["selections"],
            "pgss_logit_threshold": float(config["pgss"]["activefire_frozen_logit_threshold"]),
        },
        {
            "name": "target_calibrated",
            "gate_selections": land_val["selections"],
            "pgss_logit_threshold": float(config["pgss"]["land8fire_frozen_logit_threshold"]),
        },
    ]
    active_test = evaluate_test_split(
        gate, pgss, Path(config["data"]["activefire_test"]), "ActiveFire-Test", bands,
        active_protocols, tile_specs, batch_size, workers, device,
    )
    land_test = evaluate_test_split(
        gate, pgss, Path(config["data"]["land8fire_test"]), "Land8Fire-Test", bands,
        land_protocols, tile_specs, batch_size, workers, device,
    )

    curve_rows = active_val["curve_rows"] + land_val["curve_rows"]
    scene_rows = active_val["scene_rows"] + land_val["scene_rows"] + active_test["scene_rows"] + land_test["scene_rows"]
    tile_rows = active_val["tile_rows"] + land_val["tile_rows"] + active_test["tile_rows"] + land_test["tile_rows"]
    write_csv(Path(config["outputs"]["threshold_curve_csv"]), curve_rows, _union_fieldnames(curve_rows))
    write_csv(Path(config["outputs"]["scene_metrics_csv"]), scene_rows, _union_fieldnames(scene_rows))
    write_csv(Path(config["outputs"]["tile_coverage_csv"]), tile_rows, _union_fieldnames(tile_rows))
    comparison_rows = build_comparison_rows(
        active_test["tile_rows"] + land_test["tile_rows"],
        Path(config["baselines"].get("b7_gate_tile_coverage_csv",
                                   "output/B7_GATE_TILE_COVERAGE.csv")),
        config["baselines"]["difficult_scene_id"],
        active_test["scene_rows"] + land_test["scene_rows"],
    )
    write_csv(Path(config["outputs"]["comparison_csv"]), comparison_rows, _union_fieldnames(comparison_rows))
    train_result = load_json(Path(config["outputs"]["train_result_json"]))
    audit_result = {
        "algorithm": "SpectralMLPGate_3_16_8_1",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "gate_checkpoint": str(gate_path),
        "gate_checkpoint_sha256": sha256_file(gate_path),
        "pgss_checkpoint": config["pgss"]["checkpoint"],
        "pgss_checkpoint_sha256": sha256_file(Path(config["pgss"]["checkpoint"])),
        "gate_checkpoint_metadata": checkpoint_metadata(gate_checkpoint),
        "pgss_width": pgss_checkpoint.get("width"),
        "test_used_for_selection": False,
        "activefire_val": {key: value for key, value in active_val.items() if key not in ("curve_rows", "scene_rows")},
        "land8fire_val": {key: value for key, value in land_val.items() if key not in ("curve_rows", "scene_rows")},
        "activefire_test": {key: value for key, value in active_test.items() if key != "scene_rows"},
        "land8fire_test": {key: value for key, value in land_test.items() if key != "scene_rows"},
        "outputs": config["outputs"],
        "gpu_state": config["runtime"]["gpu_state"],
        "power_energy_policy": config["runtime"]["power_energy_policy"],
    }
    write_json(Path(config["outputs"]["audit_result_json"]), audit_result)
    report = build_report(
        config, config_path, gate_checkpoint, train_result, active_val, land_val,
        active_test["tile_rows"] + land_test["tile_rows"],
        active_test["scene_rows"] + land_test["scene_rows"],
        comparison_rows,
        {
            "active_source_995_tile32": _find_event_audit(active_test["event_audit"], "source_frozen", 0.995, 32),
            "land_zero_995_tile32": _find_event_audit(land_test["event_audit"], "zero_shot_source_frozen", 0.995, 32),
            "land_target_995_tile32": _find_event_audit(land_test["event_audit"], "target_calibrated", 0.995, 32),
        },
    )
    Path(config["outputs"]["report_md"]).write_text(report, encoding="utf-8")
    print(f"Wrote {config['outputs']['report_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
