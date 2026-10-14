# Gate-PGSS-765 SupInit No-KD — seed 20260827

This package is exported from the completed seed20260827 run in `V1_SUPINIT_NOKD_765_20260908`. It contains the 12-epoch supervised Gate and the final 40-epoch No-KD PGSS model initialized from the 40-epoch KD-free Candidate supervised checkpoint. No teacher output, distillation loss, optimizer state or TensorRT engine is included.

- `models/gate_765_seed20260827.onnx`: Gate raw logit, input `[B,3,256,256]`, output `[B,1,256,256]`.
- `models/pgss_w32_765_seed20260827.onnx`: dense reference PGSS-W32-765 raw logit, input `[B,3,256,256]`, output `[B,1,256,256]`.
- `models/pgss_w32_765_seed20260827_tile48.onnx`: Conditional Sparse verifier graph, input image `[B,3,48,48]` plus valid mask `[B,1,48,48]`, output core logit `[B,1,32,32]`.
- `deployment_manifest.json`: Orin TensorRT FP16 input, tiling, threshold and provenance contract.
- `config/`: frozen training/inference configuration and source records.
- `audit/EXPORT_AUDIT.json`: checkpoint/ONNX hashes and checker/ORT results.
- `audit/PREDICTION_PARITY_AUDIT.json` and `audit/PREDICTION_PARITY_AUDIT.md`: fixed ActiveFire-Val PyTorch–ONNX prediction parity, Gate candidate/tile index comparison, dense logits, boundary-aligned sparse tile batches 1/3/17, and end-to-end event metrics.

The formal deployed PGSS threshold is logit `2.344194173812866`; `2.225609064102173` is the checkpoint's ActiveFire-Val threshold record and is retained separately for traceability. Build engines on Orin from the two ONNX files; this host did not build TensorRT.
