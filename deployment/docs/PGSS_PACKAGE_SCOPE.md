# Scope of the copied ORIN_TRT_FP16_MINIMAL_PACKAGE

The source package is not a generic single U-Net ONNX directory. It is the deployment package for the V1 Gate/PGSS path. Its ONNX files are:

- `spectral_mlp_gate_765.onnx`: Gate model;
- `selective_kd_pgss_w32_765_48x48.onnx`: selective PGSS verifier.

The package does not contain a Full-U-Net ONNX. Its deployment manifest, frozen thresholds, audits and README were copied under `config/pgss_package/` and `docs/pgss_package/` for unified management.
