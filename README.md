# STR-Net

STR-Net: a two-stage ultra-lightweight active-fire detection pipeline for Landsat 8, in which a 209-parameter pointwise spectral gate (SG) proposes candidate pixels and a 4,401-parameter spatial verifier (SV) is run only on the selected 32x32 tiles plus an 8-pixel halo.

This repository contains the training, candidate-tile construction, formal frozen-threshold evaluation and deployment assets of the STR-Net experiment (three seeds).

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
│   │   └── pgss_model.py             #   SV: PGSSTinyV1 (4,401 params) + boundary-aligned tile forward
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
        ├── ONNX_QUICK_CHECK_PROTOCOL.md  # Pre-TensorRT export acceptance standard
        └── PGSS_PACKAGE_SCOPE.md         # What the minimal ONNX package contains
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

## 3. Data Preparation

The splits are **not bundled**. Obtain the five fixed split CSVs and place
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
  The paths inside the shipped CSVs are absolute paths of the original
  machine - update them to your local dataset location, or regenerate the
  CSVs with the same scene membership.
- Band notes: rasterio bands `7 6 5` (B7, B6, B5) scaled `float32 / 65535.0`;
  a pixel is valid iff any of the 10 source bands is nonzero.

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
tiles in four categories). Stage 3 trains the verifier from scratch with a
`WeightedRandomSampler` (category fractions 0.45/0.25/0.20/0.10), AdamW lr
1e-4, cosine schedule, batch 512, 40 epochs; the loss is computed on the 32x32
core only. Best epoch = max ActiveFire-Val candidate-pipeline F1.

Default output directory:

- `output/SEED_20260827/candidate/` (tile tensor + `model/model_best.pth`)

### 4.3 Stage 4 - SupInit No-KD Refinement (Main Method)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_supinit_nokd.sh
```

Identical objective and data to stage 3, but initialized from the stage-3
checkpoint (strict weight load; optimizer state is fresh) and trained with a
plain shuffled pass over all tiles (no sampler). Checkpoint selection priority
on ActiveFire-Val: (final event recall, small-event recall, pixel F1, PR-AUC).

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
   re-selected on Land8Fire-Val.

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

`deployment/` contains the STR-Net ONNX export tool and its contract:

- `deployment/tools/export_pgss_tile_seed20260827_onnx.py` - exports the
  seed-20260827 spatial verifier to the 48x48-tile deployment graph
  (`[B,3,48,48] image + [B,1,48,48] valid` -> `[B,1,32,32] core_logit`),
  running onnx_checker and ONNX Runtime numeric checks as it goes.
- `deployment/onnx_pgss/deployment_manifest.json` - the deployment contract:
  the three graphs (gate / dense reference / tile verifier), input shapes and
  dtypes, frozen thresholds, pipeline wiring, provenance and validation
  status. Engines are built on the target Jetson Orin from this manifest
  (TensorRT FP16; not pre-built).
- `deployment/docs/` - the pre-TensorRT ONNX quick-check protocol and the
  scope note of the minimal ONNX package.

## 7. Outputs and Weights

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

## 8. License and Citation

This project is intended for academic research and teaching. If you use it in
a paper or project, please cite this repository and acknowledge the authors.

TODO: the LICENSE file currently contains a placeholder MIT text - confirm the
final license with the authors before any public release; the BibTeX entry
will be added with the paper publication. The two networks in
`code/models/spectral_mlp_gate.py` and `code/models/pgss_model.py` are
original to this project; the frozen full-PGSS scanner used for candidate
mining originates from a previous internal experiment of the same project
line and is not a third-party work.
