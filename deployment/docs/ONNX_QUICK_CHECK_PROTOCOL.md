# ONNX quick-check protocol

This is a pre-TensorRT export check. It is not final FP16 acceptance.

For every model, freeze one shared 32-image manifest before export: 8 small-fire, 8 other-fire, and 16 background images. Use the same checkpoint, preprocessing, output head, valid mask, and frozen threshold as the PyTorch reference. Use fixed input `[1,C,256,256]`; only batch may remain dynamic.

Required checks:

1. `onnx.checker.check_model()` passes.
2. ONNX Runtime executes successfully, output shape is `[1,1,256,256]`, and PyTorch/ORT outputs are independently finite.
3. Thresholded foreground agreement in valid pixels is Dice >= 0.995. If both masks are empty, agreement Dice is 1.0.
4. Existing event matching is used to check that no sampled small-fire event becomes a new miss.

`max_abs` and `mean_abs` are diagnostic fields only. They do not independently reject the export. A finite output check must be computed as `np.isfinite(torch_output).all()` and `np.isfinite(onnx_output).all()`; do not derive it from an error list.

Statuses:

- `ONNX_INVALID`: checker/runtime/shape/non-finite failure.
- `ONNX_REVIEW`: runnable but foreground Dice or tiny-event condition fails.
- `ONNX_QUICK_CHECK_PASS`: all three pre-check groups pass; this only permits TensorRT construction.

The final decision is made once on the TensorRT FP16 engine using the complete frozen validation protocol. `ONNX_QUICK_CHECK_PASS` must never be labeled `FP16_ACCEPTED` or strict numerical parity.

