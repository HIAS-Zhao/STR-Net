# Data

The train/val/test splits are **not bundled** with this repository. This file
describes how the data is distributed so the splits can be reconstructed or
obtained; place the five CSV files in the layout below before running
anything.

```text
data/splits_activefire/voting_train.csv   # ActiveFire-Train: 50,285 patches / 2,270 scenes
data/splits_activefire/voting_val.csv     # ActiveFire-Val:   16,760 patches /   757 scenes
data/splits_activefire/voting_test.csv    # ActiveFire-Test:  16,760 patches /   757 scenes
data/splits_land8fire/land8fire_val.csv   # Land8Fire-Val:    11,050 patches /    12 scenes
data/splits_land8fire/land8fire_test.csv  # Land8Fire-Test:   12,042 patches /    13 scenes
```

## Split distribution

- Two Landsat 8 datasets, both cut into `256x256` pixel patches.
- **ActiveFire** (in-domain): 83,805 patches over 3,784 scenes, split
  scene-disjoint into train 50,285 / val 16,760 / test 16,760 patches
  (2,270 / 757 / 757 scenes). A scene never appears in two splits.
- **Land8Fire** (cross-dataset, evaluation only): val 11,050 patches / 12
  scenes, test 12,042 patches / 13 scenes.
- Pixel-level workload of the evaluation sets: ActiveFire-Test 1,064,704,488
  valid pixels with 131,989 fire pixels; Land8Fire-Test 539,934,909 valid
  pixels with 109,897 fire pixels (0.02% positive - extremely imbalanced).
- The ActiveFire labels are majority/"voting" masks aggregated from three
  reference fire-detection algorithms (Kumar-Roy / Murphy / Schroeder);
  Land8Fire labels are PNG masks.
- The splits are frozen. Never re-shuffle or re-split: all thresholds in this
  repository were frozen on ActiveFire-Val, and Test was read exactly once.

## CSV format

- First row is a header (`data_file,mask_file`, plus
  `scene_id,patch_id` for Land8Fire).
- `data_file`: path to one 10-band uint16 GeoTIFF patch.
- `mask_file`: path to the mask; labels are `mask > 0`.
- The original CSVs used absolute paths of the authoring machine - replace
  them with the paths of your local dataset copy (keep the same rows and
  scene membership).

## Preprocessing (fixed in code)

- Bands: rasterio bands `7 6 5` (B7, B6, B5), scaled `float32 / 65535.0`.
- Valid mask: a pixel is valid iff any of the 10 source bands is nonzero;
  invalid pixels are zeroed (the mask is not fed to the network).

TODO(dataset release): add the public download instructions / DOI for
ActiveFire and Land8Fire once confirmed.
