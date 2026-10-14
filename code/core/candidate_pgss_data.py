#!/usr/bin/env python3
"""Build and materialize Gate-selected PGSS verifier training tiles."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from core.audit_spectral_mlp_gate import make_audit_loader
from core.pgss_common import read_mask, read_rows, scene_id_from_row, sha256_file
from models.spectral_mlp_gate import prepare_gate_input

CATEGORIES = (
    "true_fire",
    "pgss_false_positive",
    "gate_background",
    "ordinary_background",
)


def read_full_sample(row: dict[str, str], bands: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one source patch using the same normalization and valid-mask rule."""
    image_path = Path(row["data_file"])
    with rasterio.open(image_path) as source:
        image = source.read(bands).astype(np.float32) / 65535.0
        valid_source = source.read().astype(np.float32, copy=False)
    truth = read_mask(row["mask_file"])
    valid = np.any(valid_source != 0, axis=0)
    image[:, ~valid] = 0.0
    if image.shape[1:] != truth.shape:
        raise ValueError(
            f"Image/mask mismatch for {image_path}: {image.shape[1:]} != {truth.shape}"
        )
    return image, truth.astype(bool, copy=False), valid.astype(bool, copy=False)


def _sample_category(records: list[dict[str, Any]], cap: int, rng: random.Random) -> list[dict[str, Any]]:
    if cap <= 0 or len(records) <= cap:
        return records
    indices = sorted(rng.sample(range(len(records)), cap))
    return [records[index] for index in indices]


@torch.inference_mode()
def build_candidate_manifest(
    csv_path: Path,
    output_manifest: Path,
    gate: torch.nn.Module,
    pgss: torch.nn.Module,
    bands: tuple[int, ...],
    gate_logit_threshold: float,
    pgss_logit_threshold: float,
    tile_size: int,
    halo: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
    caps: dict[str, int],
) -> dict[str, Any]:
    dataset, loader = make_audit_loader(csv_path, bands, batch_size, workers)
    records_by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    total_tiles = 0
    gate_selected_tiles = 0
    positive_tiles = 0
    missed_positive_tiles = 0
    processed_patches = 0

    gate.eval()
    pgss.eval()
    for batch in tqdm(loader, desc="ActiveFire-Train candidate scan", unit="batch"):
        images, masks, valid, scene_ids_raw, indices_raw, *_rest = batch
        scene_ids = list(scene_ids_raw)
        indices = [int(value) for value in indices_raw]
        images = images.to(device, non_blocking=True)
        truth = (masks > 0).to(device, non_blocking=True).bool()
        valid = valid.to(device, non_blocking=True).bool()
        gate_logits = gate(prepare_gate_input(images, valid))
        pgss_logits = pgss(images)
        if not bool(torch.all(torch.isfinite(gate_logits[valid]))):
            raise RuntimeError("Non-finite Gate logits during candidate scan")
        if not bool(torch.all(torch.isfinite(pgss_logits[valid]))):
            raise RuntimeError("Non-finite PGSS logits during candidate scan")

        gate_pixels = valid & (gate_logits >= gate_logit_threshold)
        selected_grid = F.max_pool2d(
            gate_pixels.float(), tile_size, stride=tile_size, ceil_mode=True
        ) > 0
        pgss_prediction = pgss_logits >= pgss_logit_threshold
        height, width = images.shape[-2:]
        rows = (height + tile_size - 1) // tile_size
        cols = (width + tile_size - 1) // tile_size
        truth_grid = F.max_pool2d(
            (truth & valid).float(), tile_size, stride=tile_size, ceil_mode=True
        ) > 0
        pgss_fp_grid = F.max_pool2d(
            (pgss_prediction & ~truth & valid).float(),
            tile_size,
            stride=tile_size,
            ceil_mode=True,
        ) > 0
        total_tiles += int(images.shape[0] * rows * cols)
        gate_selected_tiles += int(torch.count_nonzero(selected_grid).item())
        positive_tiles += int(torch.count_nonzero(truth_grid).item())
        missed_positive_tiles += int(torch.count_nonzero(truth_grid & ~selected_grid).item())

        category_masks = {
            "true_fire": selected_grid & truth_grid,
            "pgss_false_positive": selected_grid & ~truth_grid & pgss_fp_grid,
            "gate_background": selected_grid & ~truth_grid & ~pgss_fp_grid,
            "ordinary_background": ~selected_grid & ~truth_grid,
        }
        # Transfer only selected coordinates once per category instead of
        # synchronizing once for every tile.
        for category, category_mask in category_masks.items():
            coordinates = torch.nonzero(category_mask[:, 0], as_tuple=False).cpu().tolist()
            for batch_index, grid_y, grid_x in coordinates:
                row_index = indices[batch_index]
                row = dataset.rows[row_index]
                records_by_category[category].append(
                    {
                        "data_file": row["data_file"],
                        "mask_file": row["mask_file"],
                        "scene_id": scene_ids[batch_index],
                        "source_row_index": row_index,
                        "tile_y": int(grid_y) * tile_size,
                        "tile_x": int(grid_x) * tile_size,
                        "tile_size": tile_size,
                        "halo": halo,
                        "category": category,
                    }
                )
        processed_patches += int(images.shape[0])

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    category_counts_before = {category: len(records) for category, records in records_by_category.items()}
    category_counts_after = {}
    for category in CATEGORIES:
        picked = _sample_category(records_by_category[category], int(caps.get(category, 0)), rng)
        category_counts_after[category] = len(picked)
        sampled.extend(picked)
    sampled.sort(key=lambda item: (int(item["source_row_index"]), int(item["tile_y"]), int(item["tile_x"])))

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "data_file", "mask_file", "scene_id", "source_row_index", "tile_y", "tile_x",
        "tile_size", "halo", "category",
    ]
    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sampled)

    return {
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path.resolve()),
        "manifest": str(output_manifest.resolve()),
        "manifest_sha256": sha256_file(output_manifest.resolve()),
        "processed_patches": processed_patches,
        "total_tiles": total_tiles,
        "gate_selected_tiles": gate_selected_tiles,
        "positive_tiles": positive_tiles,
        "missed_positive_tiles": missed_positive_tiles,
        "missed_positive_tile_recall": (
            1.0 - missed_positive_tiles / positive_tiles if positive_tiles else float("nan")
        ),
        "category_counts_before_sampling": category_counts_before,
        "category_counts_after_sampling": category_counts_after,
        "total_training_tiles": len(sampled),
        "sampling_seed": seed,
        "caps": caps,
        "gate_logit_threshold": gate_logit_threshold,
        "pgss_logit_threshold": pgss_logit_threshold,
        "tile_size": tile_size,
        "halo": halo,
        "bands": list(bands),
        "normalization": "/65535",
    }


def materialize_candidate_tiles(
    manifest_path: Path,
    output_path: Path,
    bands: tuple[int, ...],
    tile_size: int,
    halo: int,
) -> dict[str, Any]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"Candidate manifest is empty: {manifest_path}")

    images_out = np.empty((len(records), len(bands), tile_size + 2 * halo, tile_size + 2 * halo), dtype=np.float16)
    targets_out = np.zeros((len(records), 1, tile_size, tile_size), dtype=np.uint8)
    valid_out = np.zeros((len(records), 1, tile_size, tile_size), dtype=np.uint8)
    category_ids = np.empty((len(records),), dtype=np.int64)
    categories = {category: index for index, category in enumerate(CATEGORIES)}
    current_key = None
    current_sample: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for index, record in enumerate(tqdm(records, desc="Materialize candidate tiles", unit="tile")):
        key = record["data_file"]
        if key != current_key:
            current_sample = read_full_sample(record, bands)
            current_key = key
        assert current_sample is not None
        image, truth, valid = current_sample
        y0 = int(record["tile_y"])
        x0 = int(record["tile_x"])
        image_pad = np.pad(image, ((0, 0), (halo, halo), (halo, halo)), mode="constant")
        truth_pad = np.pad(truth, ((halo, halo), (halo, halo)), mode="constant")
        valid_pad = np.pad(valid, ((halo, halo), (halo, halo)), mode="constant")
        images_out[index] = image_pad[:, y0:y0 + tile_size + 2 * halo, x0:x0 + tile_size + 2 * halo]
        targets_out[index, 0] = truth_pad[y0 + halo:y0 + halo + tile_size, x0 + halo:x0 + halo + tile_size]
        valid_out[index, 0] = valid_pad[y0 + halo:y0 + halo + tile_size, x0 + halo:x0 + halo + tile_size]
        category_ids[index] = categories[record["category"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "images": torch.from_numpy(images_out),
            "targets": torch.from_numpy(targets_out),
            "valid": torch.from_numpy(valid_out),
            "category_ids": torch.from_numpy(category_ids),
            "categories": list(CATEGORIES),
            "bands": list(bands),
            "normalization": "/65535",
            "tile_size": tile_size,
            "halo": halo,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path.resolve()),
        },
        output_path,
    )
    return {
        "tensor_file": str(output_path.resolve()),
        "tensor_sha256": sha256_file(output_path.resolve()),
        "tiles": len(records),
        "image_shape": list(images_out.shape[1:]),
        "target_shape": list(targets_out.shape[1:]),
        "category_counts": {
            category: int(np.count_nonzero(category_ids == category_id))
            for category, category_id in categories.items()
        },
    }


class CandidateTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_path: str | Path) -> None:
        payload = torch.load(Path(tensor_path).resolve(), map_location="cpu", weights_only=False)
        self.images = payload["images"]
        self.targets = payload["targets"]
        self.valid = payload["valid"]
        self.category_ids = payload["category_ids"]
        self.categories = payload.get("categories", list(CATEGORIES))

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index], self.valid[index], self.category_ids[index]
