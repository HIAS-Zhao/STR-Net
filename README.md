# STR-Net

STR-Net (Spectral-Gated Tile Routing Network) is a lightweight framework for active-fire segmentation in multispectral Landsat 8 imagery. A 209-parameter spectral gate screens pixels at original resolution, and single-pixel-triggered tile routing invokes a compact spatial verifier only on selected regions, skipping redundant background computation.

This repository provides training, evaluation, and deployment workflows; the runnable configuration currently covers seed `20260827`.

## Related Resources

The dataset preparation utilities, fixed data splits, and baseline implementations associated with our experiments are maintained in our team's related repository:

- **Repository:** [fire-segmentation](https://github.com/HIAS-Zhao/fire-segmentation)
- **Related work:** DualSight-Fire (dual-decoder fire segmentation). That repository also hosts the shared ActiveFire/Land8Fire split CSVs used here, plus baseline implementations (U-Net, Attention U-Net, U-Net++, FDE-UNet, FPS-U2Net).

Please refer to the related repository for dataset preparation, split definitions, and baseline-specific training and evaluation instructions. This repository focuses on STR-Net and its spectral gating, conditional tile routing, spatial verification, and sparse inference pipeline.

## 1. Repository Structure

(As implemented in the current codebase)

```text
.
├── README.md
├── requirements.txt                  # Tested versions (Python 3.10, CUDA 12.4)
├── .gitignore
├── LICENSE
├── code/
│   ├── models/                       # SG (gate) and SV (verifier) model definitions (+ __init__.py)
│   │   ├── spectral_mlp_gate.py      #   SG: SpectralMLPGate (209 params, 1x1 convs)
│   │   └── pgss_model.py             #   SV: PGSSTinyV1(width=32, head_width=8),
│   │                                 #       4,401 params + boundary-aligned tile forward
│   ├── core/                         # Data loading, losses, metrics, routing, sparse inference (+ __init__.py)
│   │   ├── pgss_common.py            #   Dataset, masked focal-Tversky loss, metrics
│   │   ├── audit_spectral_mlp_gate.py#   Routing (build_tile_masks), gate loader/audit, event accumulator
│   │   ├── exact_eval_pgss.py        #   Dense full-scene exact-threshold evaluator (library + CLI)
│   │   ├── candidate_pgss_eval.py    #   Sparse tiled inference (candidate_core_logits)
│   │   └── candidate_pgss_data.py    #   Candidate-tile mining + fp16 tensor materialization
│   ├── train/                        # Training entries + frozen configs (+ __init__.py)
│   │   ├── train_spectral_mlp_gate.py    # Stage-1 gate training (12 epochs)
│   │   ├── build_candidate_pgss_tiles.py # Stage-2 candidate-tile tensor build
│   │   ├── train_candidate_pgss.py       # Stage-3 candidate pretraining (40 epochs)
│   │   ├── train_candidate_nokd.py       # Stage-4 SupInit / Random-init refinement (40 epochs)
│   │   └── configs/                      # gate_765_train / candidate_765 / supinit_nokd_40e
│   └── eval/                         # Formal frozen-Test evaluation (+ __init__.py)
│       ├── evaluate_candidate_pgss_formal_test.py  # Formal evaluation entry
│       └── configs/                    # formal_eval_765 / FROZEN_THRESHOLDS record
├── scripts/                          # Stage wrappers; run from the repository root
│   ├── run_gate.sh                   # Gate training
│   ├── run_candidate_pretrain.sh     # Candidate construction + pretraining
│   ├── run_supinit_nokd.sh           # Refinement (SupInit No-KD)
│   └── run_formal_eval.sh            # Frozen-threshold evaluation
├── data/
│   └── README.md                     # Data sources, format, split distribution and placement
│                                     #   (create data/splits_activefire/ and data/splits_land8fire/
│                                     #    here and drop the five fixed split CSVs in, per section 3)
└── deployment/                       # STR-Net ONNX export and deployment documentation
    ├── tools/
    │   └── export_pgss_tile_seed20260827_onnx.py  # Exports the SV verifier to the
    │                                              #   48x48-tile ONNX deployment graph
    ├── onnx_pgss/
    │   ├── deployment_manifest.json   #   Deployment contract: graphs, shapes, thresholds,
    │   │                              #   pipeline wiring, provenance, validation status
    │   └── README.md
    └── docs/
        └── ONNX_QUICK_CHECK_PROTOCOL.md  # Pre-TensorRT export acceptance standard
```

## 2. Environment Setup

Conda is recommended (more reliable for `rasterio/gdal`).

```bash
conda create -n strnet python=3.10 -y
conda activate strnet
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

- A CUDA GPU is **required** for tile construction and all training stages
  (the scripts raise if `torch.cuda.is_available()` is false).
- Tested versions: torch 2.5.1+cu124, numpy 1.26.4, rasterio 1.4.4,
  Python 3.10. Newer versions are likely compatible but have not been tested.
- The ONNX export tool additionally needs `onnx` and `onnxruntime`; they are
  intentionally not in `requirements.txt` because core training and evaluation
  do not require them. See section 6 for the tool's current validation status.

## 3. Data Preparation

The fixed splits are **not bundled** here. They are maintained, together with
the dataset preparation utilities, in the related
[fire-segmentation](https://github.com/HIAS-Zhao/fire-segmentation) repository
(see Related Resources). Obtain the five fixed split CSVs and place
them under `data/` exactly as below - the configs read them from these paths
(`data/splits_activefire/voting_{train,val,test}.csv` and
`data/splits_land8fire/land8fire_{val,test}.csv`). See `data/README.md` for
the full distribution description.

```text
data/splits_activefire/voting_train.csv   # 50,285 patches / 2,270 scenes
data/splits_activefire/voting_val.csv     # 16,760 patches /   757 scenes
data/splits_activefire/voting_test.csv    # 16,760 patches /   757 scenes
data/splits_land8fire/land8fire_val.csv   # 11,050 patches /    12 scenes
data/splits_land8fire/land8fire_test.csv  # 12,042 patches /    13 scenes
```

- The splits are scene-disjoint and frozen; nothing in this repository
  regenerates or re-shuffles them. Recreate the two directories if missing:
  `mkdir -p data/splits_activefire data/splits_land8fire`.
- CSV format: first row is a header; each row contains an image path
  (10-band uint16 GeoTIFF, `256x256`) and a mask path (labels `mask > 0`).
  Split CSV paths may need to be updated for your local environment.
  Preserve the original scene membership when adjusting paths or
  regenerating the CSVs.
- Band notes: rasterio bands `7 6 5` (B7, B6, B5) scaled `float32 / 65535.0`;
  a pixel is valid iff any of the 10 source bands is nonzero.

STR-Net-specific input preprocessing (bands 7/6/5, `/65535`, validity rule)
and the evaluation protocol are defined in this repository and take precedence
over the related repository's defaults; do not substitute the other
repository's files for the local paths above.

**Storage/memory note (stages 2-4):** Stage 2 materializes every candidate
tile into one float16 tensor: 3,217,974 tiles of 3x48x48 (~41 GiB) plus uint8
core labels and validity masks (~6 GiB) - roughly **47 GiB of disk and host
RAM**. Both the materialization step and the training dataset load the full
tensor into memory. This is a host-memory requirement, not a GPU-memory
requirement; Stage 1 alone needs none of it.

Stages 2-4 additionally need the frozen full-image PGSS scanner checkpoint
(hard-negative mining and comparison baseline):

```text
output/frozen/full_pgss_scanner_model_best.pth
```

TODO(release): public download link. It is a PGSSTinyV1 (width 32) trained in
a previous internal run; its ActiveFire logit threshold is frozen at
`2.344194173812866` in the shipped configs. Stage 1 is fully runnable without
it; stages 2-4 require it (stage 3 loads it for its frozen-protocol validation
even though the training loss itself does not use it). Its location is set in
the shipped configs (`pgss.checkpoint` / `sources.scanner_checkpoint` /
`models.full_pgss_checkpoint`).

## 4. Training

All training stages support:

- Single GPU: run with `bash scripts/...` (no DDP code path in this project)

Run every command from the repository root.

### 4.1 Stage 1 - Spectral Gate (SG)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gate.sh
```

Masked focal-Tversky loss, AdamW lr 1e-3, cosine schedule, batch 128,
12 epochs. Checkpoint selection on ActiveFire-Val: the largest logit threshold
with pixel recall >= 99.5%, then the epoch minimizing the 32x32 candidate-tile
ratio (PR-AUC tie-break).

Default output directory:

- `output/SEED_20260827/gate/` (frozen gate logit threshold `-6.143380165100098`)

### 4.2 Stages 2+3 - Candidate Tiles and Candidate Pretraining

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_candidate_pretrain.sh
```

Stage 2 scans the training patches with the frozen gate and the frozen scanner
and materializes a float16 tensor of 48x48 windows (32 core + 8 halo; ~3.22M
tiles in four categories; the tensor needs ~47 GiB of disk and host RAM, see
section 3). The categories are `true_fire`, `pgss_false_positive`,
`gate_background`, and `ordinary_background` (fire-free tiles the gate did not
select; this category is not filtered by tile validity and makes up ~95% of
the tensor). Stage 3 trains the verifier from scratch with a
`WeightedRandomSampler` (category fractions 0.45/0.25/0.20/0.10), AdamW lr
1e-4, cosine schedule, batch 512, 40 epochs; the loss is computed on the 32x32
core only. Best epoch = max ActiveFire-Val candidate-pipeline F1, where the
candidate threshold is re-optimized on that split at every epoch.

Default output directory:

- `output/SEED_20260827/candidate/` (tile tensor + `model/model_best.pth`)

### 4.3 Stage 4 - SupInit No-KD Refinement (Main Method)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_supinit_nokd.sh
```

Identical objective and data to stage 3, but initialized from the stage-3
checkpoint (strict weight load; optimizer state is fresh) and trained with a
plain shuffled pass over all tiles (no sampler). Checkpoint selection priority
on ActiveFire-Val: (final event recall, pixel F1, PR-AUC). The small-event
recall term that appears in the training code's selection string is currently
inert - it reads a field the evaluator never produces - and does not influence
selection; see section 9.

For the random-init ablation, set `training.initialization` to `"random"` and
`sources.initialization_checkpoint` to `null` in
`code/train/configs/supinit_nokd_40e.json` first.

Default output directory:

- `output/SEED_20260827/supinit_nokd_40E/model/`

All stages are seeded (`random`/`numpy`/`torch`/`torch.cuda`); stage 4
additionally seeds its DataLoader shuffle generator. `cudnn.benchmark=True`.

## 5. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_formal_eval.sh
```

By default, weights are loaded from the paths in the config
(`output/SEED_20260827/supinit_nokd_40E/model/model_best.pth`); you can point
`code/eval/configs/formal_eval_765.json` at a different checkpoint.

The evaluator performs, in a fixed order:

1. **ActiveFire-Val** - re-selects the verifier threshold by exact max-F1 over
   the candidate scores (reproduces `2.225609064102173` for the released
   seed); the gate threshold stays frozen at its training value.
2. **ActiveFire-Test** (source-frozen) - gate + verifier with the frozen
   ActiveFire-Val thresholds.
3. **Land8Fire-Test zero-shot** - the same ActiveFire-Val thresholds.
4. **Land8Fire-Test target-calibrated** (diagnostic only) - verifier threshold
   re-selected on Land8Fire-Val. Land8Fire-Val is read only for this
   calibration; the zero-shot pass (item 3) never uses Land8Fire data for
   selection. Target-calibrated numbers are reported separately and must not
   be presented as zero-shot results.

Predictions are `verifier_logit >= threshold` restricted to gate-selected
core & valid pixels; positives outside selected tiles always count as FN
(never truncated). Metrics: precision/recall/F1/IoU, exact PR-AUC, and
8-connected fire-event recall merged across patches per scene. Test data is
read only after thresholds are frozen; the run records
`test_used_for_selection: false` in its result payload.

A standalone single-image visualization script is not included. The sparse
inference path (gate -> tile selection -> batched 48x48 verifier windows ->
core logits) is implemented in `code/core/candidate_pgss_eval.py`
(`candidate_core_logits`).

## 6. Deployment

`deployment/` documents the STR-Net ONNX deployment contract:

- `deployment/onnx_pgss/deployment_manifest.json` - the contract: three graphs
  (gate / dense reference / 48x48 tile verifier), input shapes and dtypes,
  thresholds, pipeline wiring and provenance. Engines are built on the target
  Jetson Orin from this manifest (TensorRT FP16; not pre-built).
  Threshold semantics: the gate uses `-6.143380165100098`; the tile verifier
  and its dense reference graph use the formal verifier threshold
  `2.225609064102173`; `2.344194173812866` belongs to the frozen full-PGSS
  **scanner** - a training-time candidate-mining and comparison model that is
  not part of the deployed head. As shipped, the manifest currently lists the
  scanner value under `models.pgss` and `models.pgss_tile`; this must be
  corrected before any deployment build (section 9, issue 1).
- `deployment/tools/export_pgss_tile_seed20260827_onnx.py` - intended to
  export the seed-20260827 spatial verifier to the 48x48-tile graph
  (`[B,3,48,48] image + [B,1,48,48] valid` -> `[B,1,32,32] core_logit`),
  running onnx_checker and ONNX Runtime numeric checks as it goes.
  **Status: not validated on a clean checkout.** The script still carries
  source paths from the internal directory tree and a module import that does
  not resolve in this repository layout, and `onnx`/`onnxruntime` are not in
  `requirements.txt` (section 9, issue 2). No turn-key export command is
  provided until it is fixed and re-verified.
- ONNX binaries are not part of the Git repository (`.gitignore` excludes
  `*.onnx`). The manifest records earlier internal validation status, but no
  export artifact is bundled here to re-verify it; treat the manifest as an
  interface contract, not as a verified export.
- `deployment/docs/ONNX_QUICK_CHECK_PROTOCOL.md` - the pre-TensorRT ONNX
  acceptance standard applied to the exported graphs.

## 7. Reported Results

Headline numbers as reported in the paper, aggregated over the three training
seeds under the formal frozen-threshold protocol described in section 5:

| Benchmark / protocol | Metric | Value |
| --- | --- | --- |
| ActiveFire-Test (source-frozen) | F1 | 96.73% |
| ActiveFire-Test (source-frozen) | IoU | 93.66% |
| Land8Fire-Test (target-calibrated) | F1 | 90.82% |
| Land8Fire-Test (target-calibrated) | Recall | 96.02% |
| Jetson Orin NX (TensorRT FP16) | Throughput | 115.7 img/s (ActiveFire) / 195.9 img/s (Land8Fire) |
| Jetson Orin NX (TensorRT FP16) | Energy | 38.3 mJ/img (Land8Fire) |
| vs. fastest neural baseline | Throughput | 1.27x (ActiveFire) / 2.12x (Land8Fire) |
| vs. LW-IRSTNet | Energy, Land8Fire | -41.3% |

Notes:

- Pixel-level metrics are seed-aggregated values from the paper. This
  repository ships the runnable configuration of seed `20260827` only; the
  other seeds and the trained weights are not bundled (section 8).
- In this repository's evaluator, Land8Fire-Test is produced twice: a strict
  zero-shot pass (ActiveFire-Val thresholds, section 5 item 3) and a
  target-calibrated diagnostic pass (thresholds from Land8Fire-Val, section 5
  item 4). The paper's Land8Fire figures correspond to the calibrated
  protocol; zero-shot numbers are separate and not interchangeable.
- Throughput and energy were measured on Jetson Orin NX with TensorRT FP16
  engines in the paper's deployment experiments. Engines and measurement
  scripts are not bundled here (section 6), so these figures cannot be
  reproduced from this repository alone.
- No result files are bundled; `scripts/run_formal_eval.sh` regenerates the
  per-seed outputs listed in section 8.

## 8. Outputs and Weights

Training and evaluation output directories typically include:

- `output/SEED_20260827/gate/SpectralMLPGate_best.pth`
- `output/SEED_20260827/candidate/activefire_train_candidate_tiles.{csv,pt}`
- `output/SEED_20260827/candidate/model/model_best.pth`
- `output/SEED_20260827/supinit_nokd_40E/model/model_best.pth`
- `output/SEED_20260827/supinit_nokd_40E/evaluation/FORMAL_RESULT.json`,
  `FORMAL_SUMMARY.csv`, `THRESHOLD_CURVE.csv`, `SCENE_METRICS.csv`,
  `TILE_COVERAGE.csv`, `FORMAL_REPORT.md`

Evaluation results are not bundled with this repository: running
`scripts/run_formal_eval.sh` produces them under
`output/SEED_20260827/supinit_nokd_40E/evaluation/` (`FORMAL_RESULT.json`,
`FORMAL_SUMMARY.csv`, `THRESHOLD_CURVE.csv`, `SCENE_METRICS.csv`,
`TILE_COVERAGE.csv`, `FORMAL_REPORT.md`), as described in section 5.

TODO(release): add download links for the trained seed-20260827 gate/verifier
checkpoints and the frozen scanner checkpoint. No pretrained weight is bundled
with this repository.

## 9. Known Issues

This README describes the repository as it currently is. Documentation
completeness does not imply that the code has been re-validated end-to-end on
a clean checkout, nor that the release is final. The following
documentation/code discrepancies were found while aligning this README with
the code; they are recorded here and are **not** fixed in this pass, and no
experiment was re-run:

1. **Deployment manifest verifier thresholds (P0).**
   `deployment/onnx_pgss/deployment_manifest.json` lists
   `threshold_logit: 2.344194173812866` under both `models.pgss` and
   `models.pgss_tile`. Both graphs are exports of the SV verifier, whose
   formal threshold is `2.225609064102173`
   (`code/eval/configs/FROZEN_THRESHOLDS.json`); the listed value is the
   frozen full-PGSS *scanner* threshold. Correct the manifest before building
   any deployment engine.
2. **ONNX export tool not validated on a clean checkout (P1).**
   `deployment/tools/export_pgss_tile_seed20260827_onnx.py` still carries
   source paths from the internal directory tree and a module import that does
   not resolve in this repository layout; `onnx`/`onnxruntime` are missing
   from `requirements.txt`. The script has not been re-verified in this
   repository.
3. **Stage-4 small-event recall tie-break is inert (P1).**
   `code/train/train_candidate_nokd.py` (`priority()`) reads an
   `event_size_summary` field that the evaluator never produces, so that term
   is always 0. The effective selection order is (final event recall, pixel
   F1, PR-AUC), and the `checkpoint_selection` string written into
   `TRAIN_REPORT.json` overstates it. Fixing this requires a code change; the
   shipped seed-20260827 checkpoint (epoch 30) was selected by the effective
   order.
4. **Default band lists disagree with the configs (P2).**
   `FireSegmentationCsvDataset`, `GateAuditDataset` and `load_pgss` default to
   bands `(7, 6, 2)`, while every shipped config and checkpoint uses
   `[7, 6, 5]`. All entry points pass bands explicitly, so runs are
   unaffected, but the defaults are misleading.
5. **`ordinary_background` tiles are not validity-filtered (P2).** The Stage-2
   category mask is `~selected & ~truth`, so fully invalid tiles are included
   in the training tensor (~95% of all tiles).
6. **Train/inference border semantics differ slightly (P3).** Stages 3-4 run
   a plain `model(images)` forward on the 48x48 tiles, while inference and the
   ONNX graph use `forward_boundary_aligned_tile`, which masks out-of-image
   features after the stem and each residual block. The difference is
   confined to patch borders.
7. **Probability thresholds in `FROZEN_THRESHOLDS.json` (P3).** The recorded
   `probability_threshold` values for the gate (0.002149229375984409) and the
   scanner (0.912168) do not equal the sigmoid of their logits
   (0.0021430494 and 0.9124716 respectively). The logit values are
   authoritative.
8. **Random-init ablation reporting (P3).** If only
   `training.initialization` is set to `"random"`, `TRAIN_REPORT.json` still
   records and hashes the stage-3 checkpoint as `pretrain_checkpoint`; also
   set `sources.initialization_checkpoint` to `null`, as described in
   section 4.3.

## 10. License and Citation

This project is intended for academic research and teaching. If you use it in
a paper or project, please cite this repository and acknowledge the authors.

A BibTeX entry will be added with the paper publication. The two networks in
`code/models/spectral_mlp_gate.py` and `code/models/pgss_model.py` are
original to this project; the frozen full-PGSS scanner used for candidate
mining originates from a previous internal experiment of the same project
line and is not a third-party work.
