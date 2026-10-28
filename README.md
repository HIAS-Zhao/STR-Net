# STR-Net

STR-Net (Spectral-Gated Tile Routing Network) is a lightweight framework for active-fire segmentation in multispectral imagery, such as Landsat 8 imagery. A 209-parameter spectral gate screens pixels at original resolution, and single-pixel-triggered tile routing invokes a compact spatial verifier only on selected regions, skipping redundant background computation.

This repository provides training, evaluation, and deployment workflows; the runnable configuration currently covers seed `20260827`.

## Related Resources

The dataset preparation utilities, fixed data splits, and baseline implementations associated with our experiments are maintained in our team's related repository:

- **Repository:** [fire-segmentation](https://github.com/HIAS-Zhao/fire-segmentation)
- **Related work:** DualSight-Fire (dual-decoder fire segmentation). That repository also hosts the shared ActiveFire/Land8Fire split CSVs used here and the baseline implementations.

Please refer to the related repository for dataset preparation, split definitions, and baseline-specific training and evaluation instructions. This repository focuses on STR-Net itself - spectral gating, conditional tile routing, spatial verification, and sparse inference; STR-Net-specific input preprocessing and the evaluation protocol are defined in this repository and take precedence.

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
- The ONNX export tool additionally needs `onnx` and `onnxruntime`
  (intentionally not in `requirements.txt`); see section 6 for its status.

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

## 4. Training

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

Candidate-tile construction uses the frozen spectral gate and a pretrained
PGSS scanner (threshold: 2.344194173812866) for hard-negative mining. The
scanner checkpoint is specified in the training configuration. Stage 2 scans
the training patches and materializes a float16 tensor of 48x48 windows
(32 core + 8 halo; ~3.22M tiles in four categories). The categories are
`true_fire`, `pgss_false_positive`, `gate_background`, and
`ordinary_background` (fire-free tiles the gate did not select). Stage 3
trains the verifier from scratch with a
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
on ActiveFire-Val: (final event recall, pixel-level F1, PR-AUC).

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
  **scanner**, a training-time candidate-mining and comparison model that is
  not part of the deployed head.
- `deployment/tools/export_pgss_tile_seed20260827_onnx.py` - intended to
  export the seed-20260827 spatial verifier to the 48x48-tile graph
  (`[B,3,48,48] image + [B,1,48,48] valid` -> `[B,1,32,32] core_logit`),
  running onnx_checker and ONNX Runtime numeric checks as it goes.
- ONNX binaries are not part of the Git repository (`.gitignore` excludes
  `*.onnx`); the manifest is an interface contract, not a bundled export.
- `deployment/docs/ONNX_QUICK_CHECK_PROTOCOL.md` - the pre-TensorRT ONNX
  acceptance standard applied to the exported graphs.

**Status:** the ONNX export workflow has not yet been fully validated on a
clean clone of this repository (`onnx`/`onnxruntime` are additionally
required, and are intentionally not in `requirements.txt`). Before any
deployment build, verify the exported models, thresholds, and numerical
consistency against the PyTorch reference.

## 7. Results

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
  protocol.
- Throughput and energy were measured on Jetson Orin NX with TensorRT FP16
  engines in the paper's deployment experiments.

## 8. Outputs and Weights

Training and evaluation output directories typically include:

- `output/SEED_20260827/gate/SpectralMLPGate_best.pth`
- `output/SEED_20260827/candidate/activefire_train_candidate_tiles.{csv,pt}`
- `output/SEED_20260827/candidate/model/model_best.pth`
- `output/SEED_20260827/supinit_nokd_40E/model/model_best.pth`
- `output/SEED_20260827/supinit_nokd_40E/evaluation/FORMAL_RESULT.json`,
  `FORMAL_SUMMARY.csv`, `THRESHOLD_CURVE.csv`, `SCENE_METRICS.csv`,
  `TILE_COVERAGE.csv`, `FORMAL_REPORT.md`

## 9. License and Citation

This project is intended for academic research and teaching. If you use it in
a paper or project, please cite this repository and acknowledge the authors.
